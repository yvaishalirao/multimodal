from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.llm_provider import (
    DEFAULT_GEMINI_MODEL,
    FREE_TIER_MODELS,
    GeminiProvider,
    LLMConfigError,
    get_llm_provider,
    load_llm_config,
)

PAID_MODEL = "gemini-3.1-pro-preview"  # no free tier on the pricing page


@pytest.fixture(autouse=True)
def clean_llm_env(monkeypatch):
    for var in ("LLM_PROVIDER", "GEMINI_MODEL", "ALLOW_PAID_TIER"):
        monkeypatch.delenv(var, raising=False)
    get_llm_provider.cache_clear()
    yield
    get_llm_provider.cache_clear()


def test_default_resolves_to_free_tier_gemini():
    provider = get_llm_provider()
    assert isinstance(provider, GeminiProvider)
    assert provider.model == DEFAULT_GEMINI_MODEL
    assert provider.model in FREE_TIER_MODELS


def test_blank_model_falls_back_to_free_default():
    assert load_llm_config({"GEMINI_MODEL": ""}).model == DEFAULT_GEMINI_MODEL


def test_paid_model_without_override_is_config_error():
    with pytest.raises(LLMConfigError, match="ALLOW_PAID_TIER"):
        load_llm_config({"GEMINI_MODEL": PAID_MODEL})


@pytest.mark.parametrize("value", ["", "false", "1", "yes", "TRUE-ish"])
def test_only_literal_true_unlocks_paid_tier(value):
    with pytest.raises(LLMConfigError):
        load_llm_config({"GEMINI_MODEL": PAID_MODEL, "ALLOW_PAID_TIER": value})


def test_paid_model_with_explicit_override_is_allowed():
    config = load_llm_config({"GEMINI_MODEL": PAID_MODEL, "ALLOW_PAID_TIER": "true"})
    assert config.model == PAID_MODEL
    assert config.allow_paid_tier


def test_unknown_provider_is_config_error_not_fallback():
    with pytest.raises(LLMConfigError, match="only 'gemini'"):
        load_llm_config({"LLM_PROVIDER": "openai"})


def test_paid_model_fails_app_startup(monkeypatch):
    from app.main import app

    monkeypatch.setenv("GEMINI_MODEL", PAID_MODEL)
    with pytest.raises(LLMConfigError):
        with TestClient(app):
            pass


class _Schema(BaseModel):
    vendor: str


async def test_gemini_provider_sends_images_and_text_with_schema():
    provider = GeminiProvider(model=DEFAULT_GEMINI_MODEL, api_key="test-key")
    generate = AsyncMock(return_value=SimpleNamespace(text='{"vendor": "Acme"}'))
    provider._client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    )

    result = await provider.extract_fields([b"\x89PNG-page-1"], "Invoice text", _Schema)

    kwargs = generate.await_args.kwargs
    assert kwargs["model"] == DEFAULT_GEMINI_MODEL
    image_part, text_part = kwargs["contents"]
    assert image_part.inline_data.data == b"\x89PNG-page-1"
    assert image_part.inline_data.mime_type == "image/png"
    assert text_part == "Invoice text"
    assert kwargs["config"].response_schema is _Schema
    assert result.raw_output == {"vendor": "Acme"}
    assert result.model == DEFAULT_GEMINI_MODEL


async def test_missing_api_key_errors_only_on_call():
    provider = GeminiProvider(model=DEFAULT_GEMINI_MODEL, api_key=None)
    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
        await provider.extract_fields([], "text", _Schema)
