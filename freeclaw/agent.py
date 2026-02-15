import json
import sys
from dataclasses import dataclass
from typing import Any, Callable

from .providers.nim import NimChatClient
from .tools import ToolContext, dispatch_tool_call, tool_schemas


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


def run_agent(
    *,
    client: NimChatClient,
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
    steps = 0
    last_resp: dict[str, Any] = {}

    for _ in range(max_tool_steps + 1):
        tools = (
            tools_builder()
            if tools_builder is not None
            else (tools_override if tools_override is not None else (tool_schemas() if enable_tools else None))
        )
        steps += 1
        last_resp = client.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )
        msg = _extract_message(last_resp)
        tool_calls = msg.get("tool_calls") if isinstance(msg.get("tool_calls"), list) else []

        assistant_entry: dict[str, Any] = {"role": "assistant", "content": msg.get("content")}
        if tool_calls:
            assistant_entry["tool_calls"] = tool_calls
        messages.append(assistant_entry)

        if not tool_calls:
            return AgentResult(
                text=NimChatClient.extract_text(last_resp),
                raw_last_response=last_resp,
                steps=steps,
            )

        if not enable_tools or tool_ctx is None:
            return AgentResult(
                text="[tool call requested; tools disabled in this run]",
                raw_last_response=last_resp,
                steps=steps,
            )

        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            tc_id = tc.get("id")
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = fn.get("name")
            args_json = fn.get("arguments", "{}")
            if not isinstance(tc_id, str) or not isinstance(name, str) or not isinstance(args_json, str):
                continue

            if verbose_tools:
                print(f"[tool] {name} {args_json}", file=sys.stderr, flush=True)

            try:
                result = dispatch_tool_call(tool_ctx, name, args_json)
                content = json.dumps(result, ensure_ascii=True)
            except Exception as e:
                content = json.dumps({"ok": False, "tool": name, "error": str(e)}, ensure_ascii=True)

            if verbose_tools:
                print(f"[tool-result] {content[:4000]}", file=sys.stderr, flush=True)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": name,
                    "content": content,
                }
            )

    return AgentResult(
        text="[max tool steps exceeded]",
        raw_last_response=last_resp,
        steps=steps,
    )
