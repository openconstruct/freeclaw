from __future__ import annotations

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


def fetch_models(*, base_url: str, api_key: str, timeout_s: float = 30.0) -> list[dict[str, Any]]:
    """
    Fetch Groq /models entries.

    Returns a list of dict entries (raw objects), best-effort.
    """
    url = base_url.rstrip("/") + "/models"
    resp = get_json(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "freeclaw/0.1.0",
        },
        timeout_s=float(timeout_s),
    ).json
    data = resp.get("data")
    if not isinstance(data, list):
        return []
    out: list[dict[str, Any]] = []
    for m in data:
        if isinstance(m, dict):
            out.append(m)
    return out


def model_ids(models: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for m in models:
        mid = m.get("id")
        if isinstance(mid, str) and mid.strip():
            ids.append(mid.strip())
    return sorted(set(ids), key=str.lower)


@dataclass(frozen=True)
class GroqChatClient:
    base_url: str
    api_key: str
    model: str | None
    timeout_s: float = 120.0
    _resolved_model: str | None = field(default=None, init=False, repr=False, compare=False)

    @staticmethod
    def from_config(cfg: ClawConfig) -> "GroqChatClient":
        api_key = _first_env("GROQ_API_KEY", "GROQ_KEY")
        if not api_key:
            raise SystemExit("Missing Groq API key. Set GROQ_API_KEY.")

        # If the user flips provider but forgets base_url, avoid accidentally talking to another provider.
        base_url = (cfg.base_url or "").strip().rstrip("/")
        if not base_url or base_url in {
            "https://integrate.api.nvidia.com/v1",
            "https://openrouter.ai/api/v1",
        }:
            base_url = "https://api.groq.com/openai/v1"

        return GroqChatClient(base_url=base_url, api_key=api_key, model=cfg.model)

    def with_model(self, model: str | None) -> "GroqChatClient":
        if model == self.model:
            return self
        return GroqChatClient(
            base_url=self.base_url,
            api_key=self.api_key,
            model=model,
            timeout_s=self.timeout_s,
        )

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

        models = fetch_models(base_url=self.base_url, api_key=self.api_key, timeout_s=min(self.timeout_s, 30.0))
        ids = model_ids(models)
        if not ids:
            raise SystemExit(
                "No model configured and /models did not return a non-empty list. Set FREECLAW_MODEL."
            )

        # Prefer current common Groq defaults if present.
        preferred = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
        ]
        for p in preferred:
            if p in ids:
                object.__setattr__(self, "_resolved_model", p)
                return p

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
        url = f"{self.base_url.rstrip('/')}/chat/completions"
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

