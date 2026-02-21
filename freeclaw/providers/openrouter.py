from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..common import first_env
from ..config import ClawConfig
from ..http_client import post_json
from .common import build_chat_payload, extract_chat_text, fetch_openai_models


def _as_price(v: Any) -> float | None:
    # OpenRouter returns prices as strings; keep parsing permissive.
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None
    return None


def is_free_model(model_obj: Any) -> bool:
    """
    Best-effort "free" detection for OpenRouter model entries.

    Heuristics:
    - id ends with ':free'
    - pricing.prompt == 0 and pricing.completion == 0 (when present)
    """
    if not isinstance(model_obj, dict):
        return False
    mid = model_obj.get("id")
    if isinstance(mid, str) and mid.strip().lower().endswith(":free"):
        return True
    pricing = model_obj.get("pricing")
    if isinstance(pricing, dict):
        p = _as_price(pricing.get("prompt"))
        c = _as_price(pricing.get("completion"))
        if p is not None and c is not None and p == 0.0 and c == 0.0:
            return True
    return False


def fetch_models(*, base_url: str, api_key: str | None = None, timeout_s: float = 30.0) -> list[dict[str, Any]]:
    """
    Fetch OpenRouter /models entries.

    Returns a list of dict entries (raw objects from OpenRouter), best-effort.
    """
    headers: dict[str, str] = {"Accept": "application/json", "User-Agent": "freeclaw/0.1.0"}
    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    return fetch_openai_models(base_url=base_url, headers=headers, timeout_s=float(timeout_s))


def model_ids(models: list[dict[str, Any]], *, free_only: bool = False) -> list[str]:
    ids: list[str] = []
    for m in models:
        mid = m.get("id")
        if not isinstance(mid, str) or not mid.strip():
            continue
        if free_only and not is_free_model(m):
            continue
        ids.append(mid.strip())
    return sorted(set(ids), key=str.lower)


@dataclass(frozen=True)
class OpenRouterChatClient:
    base_url: str
    api_key: str
    model: str | None
    timeout_s: float = 120.0
    _resolved_model: str | None = field(default=None, init=False, repr=False, compare=False)

    @staticmethod
    def from_config(cfg: ClawConfig) -> "OpenRouterChatClient":
        api_key = first_env("OPENROUTER_API_KEY", "OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("Missing OpenRouter API key. Set OPENROUTER_API_KEY.")

        # If the user flips provider but forgets base_url, avoid accidentally talking to NIM.
        base_url = (cfg.base_url or "").strip().rstrip("/")
        if not base_url or base_url == "https://integrate.api.nvidia.com/v1":
            base_url = "https://openrouter.ai/api/v1"

        return OpenRouterChatClient(base_url=base_url, api_key=api_key, model=cfg.model)

    def with_model(self, model: str | None) -> "OpenRouterChatClient":
        if model == self.model:
            return self
        return OpenRouterChatClient(
            base_url=self.base_url,
            api_key=self.api_key,
            model=model,
            timeout_s=self.timeout_s,
        )

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "User-Agent": "freeclaw/0.1.0",
        }
        # Optional (recommended by OpenRouter): identify your app.
        referer = first_env("OPENROUTER_HTTP_REFERER", "FREECLAW_OPENROUTER_HTTP_REFERER")
        title = first_env("OPENROUTER_X_TITLE", "FREECLAW_OPENROUTER_X_TITLE")
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
        return headers

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

        # Prefer free models if available.
        free_ids = model_ids(models, free_only=True)
        if free_ids:
            object.__setattr__(self, "_resolved_model", free_ids[0])
            return free_ids[0]

        # Otherwise: just pick a stable first entry.
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
