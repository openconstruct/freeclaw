import os
from dataclasses import dataclass, field
from typing import Any

from ..config import ClawConfig
from ..http_client import get_json, post_json


def _first_env(*names: str) -> str | None:
    for n in names:
        v = os.getenv(n)
        if v and v.strip():
            return v.strip()
    return None


@dataclass(frozen=True)
class NimChatClient:
    base_url: str
    api_key: str
    model: str | None
    timeout_s: float = 120.0
    _resolved_model: str | None = field(default=None, init=False, repr=False, compare=False)

    @staticmethod
    def from_config(cfg: ClawConfig) -> "NimChatClient":
        api_key = _first_env("NVIDIA_API_KEY", "NIM_API_KEY", "NVIDIA_NIM_API_KEY")
        if not api_key:
            raise SystemExit(
                "Missing NVIDIA API key. Set NVIDIA_API_KEY (or NIM_API_KEY / NVIDIA_NIM_API_KEY)."
            )
        return NimChatClient(base_url=cfg.base_url.rstrip("/"), api_key=api_key, model=cfg.model)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "User-Agent": "freeclaw/0.1.0",
        }

    def _resolve_model(self) -> str:
        if self.model:
            return self.model
        if self._resolved_model:
            return self._resolved_model
        # Try OpenAI-compatible model listing; pick a reasonable default.
        models_url = f"{self.base_url}/models"
        resp = get_json(models_url, headers=self._headers(), timeout_s=self.timeout_s).json
        data = resp.get("data")
        if not isinstance(data, list) or not data:
            raise SystemExit(
                "No model configured and /models did not return a non-empty list. "
                "Set FREECLAW_MODEL."
            )

        ids: list[str] = []
        for m in data:
            mid = m.get("id") if isinstance(m, dict) else None
            if isinstance(mid, str) and mid.strip():
                ids.append(mid.strip())
        if not ids:
            raise SystemExit("No model ids found in /models response. Set FREECLAW_MODEL.")

        # Prefer common instruct models if present.
        preferred = [
            "meta/llama-3.1-8b-instruct",
            "meta/llama-3.1-70b-instruct",
            "meta/llama-3.1-405b-instruct",
        ]
        for p in preferred:
            if p in ids:
                object.__setattr__(self, "_resolved_model", p)
                return p
        for mid in ids:
            if "llama" in mid.lower() and "instruct" in mid.lower():
                object.__setattr__(self, "_resolved_model", mid)
                return mid
        object.__setattr__(self, "_resolved_model", ids[0])
        return ids[0]

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self._resolve_model(),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return post_json(url, headers=self._headers(), payload=payload, timeout_s=self.timeout_s).json

    @staticmethod
    def extract_text(resp: dict[str, Any]) -> str:
        # OpenAI chat.completions shape.
        choices = resp.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("Unexpected response: missing choices")
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(msg, dict):
            raise RuntimeError("Unexpected response: missing message")
        content = msg.get("content")
        if content is None:
            # Some providers may return tool calls without content.
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                return "[tool call requested; tool execution not implemented in freeclaw yet]"
            return ""
        if not isinstance(content, str):
            return str(content)
        return content

    @staticmethod
    def extract_tool_calls(resp: dict[str, Any]) -> list[dict[str, Any]]:
        choices = resp.get("choices")
        if not isinstance(choices, list) or not choices:
            return []
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(msg, dict):
            return []
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            return []
        out: list[dict[str, Any]] = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                out.append(tc)
        return out
