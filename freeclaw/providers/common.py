from __future__ import annotations

from typing import Any

from ..http_client import get_json


def fetch_openai_models(
    *,
    base_url: str,
    headers: dict[str, str],
    timeout_s: float,
) -> list[dict[str, Any]]:
    url = base_url.rstrip("/") + "/models"
    resp = get_json(url, headers=headers, timeout_s=float(timeout_s)).json
    data = resp.get("data")
    if not isinstance(data, list):
        return []
    out: list[dict[str, Any]] = []
    for m in data:
        if isinstance(m, dict):
            out.append(m)
    return out


def model_ids_from_entries(models: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for m in models:
        mid = m.get("id")
        if isinstance(mid, str) and mid.strip():
            ids.append(mid.strip())
    return sorted(set(ids), key=str.lower)


def build_chat_payload(
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": str(model),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    return payload


def extract_chat_text(resp: dict[str, Any]) -> str:
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Unexpected response: missing choices")
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        raise RuntimeError("Unexpected response: missing message")
    content = msg.get("content")
    if content is None:
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            return "[tool call requested; tool execution not implemented in freeclaw yet]"
        return ""
    if not isinstance(content, str):
        return str(content)
    return content
