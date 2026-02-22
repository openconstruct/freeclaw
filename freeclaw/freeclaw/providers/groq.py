from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..common import first_env
from ..config import ClawConfig
from ..http_client import post_json
from .common import build_chat_payload, extract_chat_text, fetch_openai_models, model_ids_from_entries


def fetch_models(*, base_url: str, api_key: str, timeout_s: float = 30.0) -> list[dict[str, Any]]:
    """
    Fetch Groq /models entries.

    Returns a list of dict entries (raw objects), best-effort.
    """
    return fetch_openai_models(
        base_url=base_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "freeclaw/0.1.0",
        },
        timeout_s=float(timeout_s),
    )


def model_ids(models: list[dict[str, Any]]) -> list[str]:
    return model_ids_from_entries(models)


@dataclass(frozen=True)
class GroqChatClient:
    base_url: str
    api_key: str
    model: str | None
    timeout_s: float = 120.0
    _resolved_model: str | None = field(default=None, init=False, repr=False, compare=False)

    @staticmethod
    def from_config(cfg: ClawConfig) -> "GroqChatClient":
        api_key = first_env("GROQ_API_KEY", "GROQ_KEY")
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
        payload = build_chat_payload(
            model=self._resolve_model(),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )
        return post_json(url, headers=self._headers(), payload=payload, timeout_s=self.timeout_s).json

    @staticmethod
    def extract_text(resp: dict[str, Any]) -> str:
        return extract_chat_text(resp)
