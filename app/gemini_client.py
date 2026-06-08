"""Gemini client — offline preprocessing only (never used at rank time).

Authenticates via Application Default Credentials (OAuth refresh token JSON) when
``GOOGLE_APPLICATION_CREDENTIALS`` is set, otherwise falls back to ``GEMINI_API_KEY``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google import genai

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_PRO_MODEL = "gemini-2.5-pro"
DEFAULT_GEMINI_FLASH_MODEL = "gemini-3.5-flash"

FLASH_MODEL_FALLBACKS: tuple[str, ...] = (
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-latest",
)
PRO_MODEL_FALLBACKS: tuple[str, ...] = (
    "gemini-3.5-pro",
    "gemini-2.5-pro",
    "gemini-pro-latest",
    "gemini-2.0-flash",
)


def get_api_key(settings) -> str | None:
    """Return API key from settings or GEMINI_API_KEY env (optional fallback)."""
    key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key and key.strip() and key.strip() != "your_gemini_api_key_here":
        return key.strip()
    return None


def resolve_flash_model(settings) -> str:
    return getattr(settings, "gemini_flash_model", None) or DEFAULT_GEMINI_FLASH_MODEL


def resolve_pro_model(settings) -> str:
    return getattr(settings, "gemini_pro_model", None) or DEFAULT_GEMINI_PRO_MODEL


def has_gemini_auth(settings) -> bool:
    """True when an API key or Application Default Credentials are available."""
    if get_api_key(settings):
        return True
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path and Path(creds_path).is_file():
        return True
    try:
        import google.auth

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        return credentials is not None
    except Exception:
        return False


@lru_cache(maxsize=1)
def get_genai_client(api_key: str | None) -> genai.Client:
    """Cached Gemini client — ADC when no API key is provided."""
    from google import genai

    if api_key:
        logger.info("Gemini auth: API key")
        return genai.Client(api_key=api_key)
    logger.info("Gemini auth: Application Default Credentials")
    return genai.Client()


def generate_json(
    prompt: str,
    *,
    settings,
    model: str = DEFAULT_GEMINI_FLASH_MODEL,
    temperature: float = 0.0,
    model_fallbacks: tuple[str, ...] | None = None,
) -> dict[str, Any] | list[Any]:
    """Call Gemini and parse JSON response, trying fallback models on 404."""
    from google.genai import types

    client = get_genai_client(get_api_key(settings))
    candidates = _candidate_models(model, model_fallbacks or FLASH_MODEL_FALLBACKS)
    last_error: Exception | None = None

    for model_name in candidates:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                ),
            )
            text = response.text or ""
            if model_name != model:
                logger.info("Gemini call succeeded with fallback model %s", model_name)
            return parse_json_response(text)
        except Exception as exc:
            last_error = exc
            if _is_model_not_found(exc):
                logger.warning("Gemini model unavailable: %s — trying next fallback", model_name)
                continue
            raise

    raise RuntimeError(
        f"All Gemini models failed. Tried: {', '.join(candidates)}. Last error: {last_error}"
    )


def _candidate_models(primary: str, fallbacks: tuple[str, ...]) -> list[str]:
    ordered: list[str] = []
    for name in (primary, *fallbacks):
        if name not in ordered:
            ordered.append(name)
    return ordered


def _is_model_not_found(exc: Exception) -> bool:
    message = str(exc).lower()
    return "404" in message or "not found" in message or "not supported" in message


def parse_json_response(text: str) -> dict[str, Any] | list[Any]:
    """Parse JSON from model output, stripping optional markdown fences."""
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    return json.loads(cleaned)


def load_prompt_template(name: str) -> str:
    """Load prompt template from prompts/ directory."""
    from app.config import PROJECT_ROOT

    path = PROJECT_ROOT / "prompts" / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")
