import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .tools import ToolContext, dispatch_tool_call, tool_schemas

log = logging.getLogger(__name__)

class ChatClient(Protocol):
    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]: ...

    def extract_text(self, resp: dict[str, Any]) -> str: ...


@dataclass(frozen=True)
class AgentResult:
    text: str
    raw_last_response: dict[str, Any]
    steps: int


def _extract_message(resp: dict[str, Any]) -> dict[str, Any]:
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Unexpected response: missing choices")
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        raise RuntimeError("Unexpected response: missing message")
    return msg


def _is_json_object(s: str) -> bool:
    try:
        v = json.loads(s)
    except Exception:
        return False
    return isinstance(v, dict)


def _sanitize_tool_calls(tool_calls: list[Any]) -> list[dict[str, Any]]:
    """Sanitize tool_calls to be safe to send back to OpenAI-compatible APIs.

    Some providers validate that tool_call.function.arguments is valid JSON, and will 400
    if the model returned malformed JSON. To keep the agent resilient, we replace any
    malformed/non-object arguments with "{}" (still dispatching with the raw arguments).
    """
    out: list[dict[str, Any]] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        tc_id = tc.get("id")
        fn = tc.get("function")
        if not isinstance(tc_id, str) or not tc_id:
            continue
        if not isinstance(fn, dict):
            continue
        nm = fn.get("name")
        if not isinstance(nm, str) or not nm.strip():
            continue
        args = fn.get("arguments", "{}")
        if not isinstance(args, str) or not _is_json_object(args):
            tc2 = dict(tc)
            fn2 = dict(fn)
            fn2["arguments"] = "{}"
            tc2["function"] = fn2
            out.append(tc2)
        else:
            out.append(tc)
    return out


def _extract_finish_reason(resp: dict[str, Any]) -> str | None:
    try:
        choices = resp.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        c0 = choices[0] if isinstance(choices[0], dict) else None
        if not isinstance(c0, dict):
            return None
        fr = c0.get("finish_reason") or c0.get("stop_reason") or c0.get("finishReason")
        return fr.strip() if isinstance(fr, str) and fr.strip() else None
    except Exception:
        return None


def _is_tool_grammar_error(exc: Exception) -> bool:
    s = str(exc or "")
    if not s:
        return False
    m = s.lower()
    return (
        ("invalid grammar request" in m)
        or ("structural_tag" in m)
        or ("tool_calls_section_begin" in m)
    )


def run_agent(
    *,
    client: ChatClient,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    enable_tools: bool,
    tool_ctx: ToolContext | None,
    max_tool_steps: int,
    verbose_tools: bool = False,
    tools_override: list[dict[str, Any]] | None = None,
    tools_builder: Callable[[], list[dict[str, Any]]] | None = None,
) -> AgentResult:
    log.debug(
        "agent start model=%s temperature=%.3f max_tokens=%d enable_tools=%s max_tool_steps=%d messages=%d",
        getattr(client, "model", None),
        float(temperature),
        int(max_tokens),
        bool(enable_tools),
        int(max_tool_steps),
        len(messages),
    )
    steps = 0
    last_resp: dict[str, Any] = {}

    for _ in range(max_tool_steps + 1):
        tools = (
            tools_builder()
            if tools_builder is not None
            else (tools_override if tools_override is not None else (tool_schemas() if enable_tools else None))
        )
        steps += 1
        try:
            last_resp = client.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
            )
        except Exception as e:
            # Some providers (notably NIM) can intermittently reject tool schemas
            # with grammar/cache errors. Retry once without tools for this turn.
            if tools and _is_tool_grammar_error(e):
                log.warning("provider rejected tool grammar; retrying this turn without tools: %s", e)
                last_resp = client.chat(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=None,
                )
            else:
                raise
        log.debug("agent step=%d finish_reason=%s", steps, (_extract_finish_reason(last_resp) or "unknown"))
        msg = _extract_message(last_resp)
        raw_tool_calls = msg.get("tool_calls") if isinstance(msg.get("tool_calls"), list) else []
        tool_calls = _sanitize_tool_calls(raw_tool_calls)

        assistant_entry: dict[str, Any] = {"role": "assistant"}
        # Some providers are picky about null content; omit it when absent (common for tool-call turns).
        if msg.get("content") is not None:
            assistant_entry["content"] = msg.get("content")
        if tool_calls:
            assistant_entry["tool_calls"] = tool_calls
        messages.append(assistant_entry)

        if not tool_calls:
            text = client.extract_text(last_resp)
            if str(text or "").strip():
                log.info("agent completed steps=%d finish_reason=%s", steps, (_extract_finish_reason(last_resp) or "unknown"))
                return AgentResult(
                    text=text,
                    raw_last_response=last_resp,
                    steps=steps,
                )

            # Some providers/models occasionally return an empty assistant message (often after tool use).
            # Retry once without tools while preserving tool-result context, so UIs (Discord) don't
            # send an empty message.
            fr = _extract_finish_reason(last_resp) or "unknown"
            if tools:
                # Remove the empty assistant message we just appended above before retrying.
                try:
                    if messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "assistant":
                        messages.pop()
                except Exception:
                    pass
                try:
                    steps += 1
                    last_resp = client.chat(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=None,
                    )
                    msg2 = _extract_message(last_resp)
                    raw_tool_calls2 = (
                        msg2.get("tool_calls") if isinstance(msg2.get("tool_calls"), list) else []
                    )
                    tool_calls2 = _sanitize_tool_calls(raw_tool_calls2)
                    assistant_entry2: dict[str, Any] = {"role": "assistant"}
                    if msg2.get("content") is not None:
                        assistant_entry2["content"] = msg2.get("content")
                    if tool_calls2:
                        assistant_entry2["tool_calls"] = tool_calls2
                    messages.append(assistant_entry2)

                    if tool_calls2:
                        return AgentResult(
                            text="[tool call requested; tools disabled in empty-response recovery]",
                            raw_last_response=last_resp,
                            steps=steps,
                        )

                    text2 = client.extract_text(last_resp)
                    if str(text2 or "").strip():
                        log.info("agent completed via empty-response recovery steps=%d", steps)
                        return AgentResult(
                            text=text2,
                            raw_last_response=last_resp,
                            steps=steps,
                        )
                except Exception as e:
                    log.warning("empty response recovery failed: %s", e, exc_info=True)

            return AgentResult(
                text=f"[empty response from model (finish_reason={fr})]",
                raw_last_response=last_resp,
                steps=steps,
            )

        if not enable_tools or tool_ctx is None:
            return AgentResult(
                text="[tool call requested; tools disabled in this run]",
                raw_last_response=last_resp,
                steps=steps,
            )

        for tc in raw_tool_calls:
            if not isinstance(tc, dict):
                continue
            tc_id = tc.get("id")
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = fn.get("name")
            args_json = fn.get("arguments", "{}")
            if not isinstance(tc_id, str) or not isinstance(name, str) or not isinstance(args_json, str):
                continue

            if verbose_tools:
                log.info("[tool] %s %s", name, args_json)

            try:
                result = dispatch_tool_call(tool_ctx, name, args_json)
                content = json.dumps(result, ensure_ascii=True)
            except Exception as e:
                err: dict[str, Any] = {"ok": False, "tool": name, "error": str(e)}
                # Attach raw arguments to help the model self-correct (bounded).
                if args_json and len(args_json) <= 4000:
                    err["arguments_json"] = args_json
                content = json.dumps(err, ensure_ascii=True)

            if verbose_tools:
                log.info("[tool-result] %s", content[:4000])

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": content,
                }
            )

    return AgentResult(
        text="[max tool steps exceeded]",
        raw_last_response=last_resp,
        steps=steps,
    )
