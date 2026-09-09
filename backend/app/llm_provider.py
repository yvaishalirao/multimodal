"""Swappable LLM provider interface (S3-T1, INV-14).

One abstract interface, one concrete provider (Gemini) wired by default.
Selection comes from env config, and every path that could reach a paid
model is closed off unless explicitly opened:

- LLM_PROVIDER: only "gemini" is accepted; an unknown name is a config
  error, never a silent fallback to some other provider.
- GEMINI_MODEL: defaults to a free-tier model; any model outside
  FREE_TIER_MODELS is a config error unless ALLOW_PAID_TIER=true.

validate_llm_config() runs at app startup (see main.py lifespan), so a
misconfiguration stops the service from starting instead of surfacing as a
surprise bill on the first extraction.
"""
import json
import os
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

from pydantic import BaseModel

DEFAULT_PROVIDER = "gemini"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"

# Text+vision models with a Gemini API free tier, per
# https://ai.google.dev/gemini-api/docs/pricing (checked 2026-10-03).
# Image/TTS/live/audio-only models are excluded: they can't do this job.
FREE_TIER_MODELS = frozenset(
    {
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    }
)


class LLMConfigError(Exception):
    pass


class LLMConfig(BaseModel):
    provider: str
    model: str
    allow_paid_tier: bool


class RawExtractionResult(BaseModel):
    """Unvalidated structured output, exactly as the model returned it.
    Schema validation happens downstream (S3-T3), not here."""

    provider: str
    model: str
    raw_output: dict[str, Any]


class LLMProvider(ABC):
    name: str
    model: str

    @abstractmethod
    async def extract_fields(
        self, images: list[bytes], text: str, schema: type[BaseModel]
    ) -> RawExtractionResult:
        """images: page images as PNG bytes; text: Docling text/table
        content; schema: the Pydantic model the output should follow."""


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, model: str, api_key: str | None):
        self.model = model
        self._api_key = api_key
        self._client = None

    def _get_client(self):
        # Built lazily so the stack starts (and tests run) without a key;
        # only an actual LLM call requires one.
        if self._client is None:
            if not self._api_key:
                raise LLMConfigError("GEMINI_API_KEY is not set.")
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    async def extract_fields(
        self, images: list[bytes], text: str, schema: type[BaseModel]
    ) -> RawExtractionResult:
        from google.genai import types

        contents: list[Any] = [
            types.Part.from_bytes(data=image, mime_type="image/png") for image in images
        ]
        contents.append(text)

        response = await self._get_client().aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        return RawExtractionResult(
            provider=self.name, model=self.model, raw_output=json.loads(response.text)
        )


def load_llm_config(env: dict[str, str] | None = None) -> LLMConfig:
    env = os.environ if env is None else env

    provider = (env.get("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if provider != "gemini":
        raise LLMConfigError(
            f"Unknown LLM_PROVIDER '{provider}'; only 'gemini' is supported."
        )

    model = (env.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip()
    allow_paid_tier = env.get("ALLOW_PAID_TIER", "").strip().lower() == "true"
    if model not in FREE_TIER_MODELS and not allow_paid_tier:
        raise LLMConfigError(
            f"GEMINI_MODEL '{model}' is not on the free-tier allow-list. "
            "Set ALLOW_PAID_TIER=true to use it deliberately."
        )

    return LLMConfig(provider=provider, model=model, allow_paid_tier=allow_paid_tier)


def validate_llm_config() -> LLMConfig:
    return load_llm_config()


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    config = load_llm_config()
    return GeminiProvider(model=config.model, api_key=os.environ.get("GEMINI_API_KEY"))
