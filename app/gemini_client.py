"""Gemini API client — offline preprocessing only (never used at rank time)."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# Defaults updated for current Google AI API (1.5 models retired on many keys).
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
    """Return configured API key from settings or GEMINI_API_KEY env var."""
    key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
    if key and key.strip() and key.strip() != "your_gemini_api_key_here":
        return key.strip()
    return None


def resolve_flash_model(settings) -> str:
    return getattr(settings, "gemini_flash_model", None) or DEFAULT_GEMINI_FLASH_MODEL


def resolve_pro_model(settings) -> str:
    return getattr(settings, "gemini_pro_model", None) or DEFAULT_GEMINI_PRO_MODEL


def generate_json(
    prompt: str,
    *,
    api_key: str,
    model: str = DEFAULT_GEMINI_FLASH_MODEL,
    temperature: float = 0.0,
    model_fallbacks: tuple[str, ...] | None = None,
) -> dict[str, Any] | list[Any]:
    """Call Gemini and parse JSON response, trying fallback models on 404."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    candidates = _candidate_models(model, model_fallbacks or FLASH_MODEL_FALLBACKS)
    last_error: Exception | None = None

    for model_name in candidates:
        try:
            gemini_model = genai.GenerativeModel(
                model_name=model_name,
                generation_config={
                    "temperature": temperature,
                    "response_mime_type": "application/json",
                },
            )
            response = gemini_model.generate_content(prompt)
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
