"""Anthropic SDK wrapper with prompt caching support. Also supports OpenRouter via LLM_PROVIDER=openrouter."""
import os
import time
import logging

import anthropic

logger = logging.getLogger(__name__)

_anthropic_client: anthropic.Anthropic | None = None
_openrouter_client = None


def _provider() -> str:
    return os.environ.get("LLM_PROVIDER", "anthropic").lower()


def _get_anthropic_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in environment")
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


def _get_openrouter_client():
    global _openrouter_client
    if _openrouter_client is None:
        from openai import OpenAI
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set in environment")
        _openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return _openrouter_client


def call_with_cache(
    model: str,
    system_blocks: list[dict],
    user_message: str,
    max_tokens: int = 4000,
    retries: int = 3,
) -> str:
    if _provider() == "openrouter":
        return _call_openrouter(model, system_blocks, user_message, max_tokens, retries)
    return _call_anthropic(model, system_blocks, user_message, max_tokens, retries)


def _call_anthropic(model, system_blocks, user_message, max_tokens, retries):
    client = _get_anthropic_client()
    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_blocks,
                messages=[{"role": "user", "content": user_message}],
            )
            usage = resp.usage
            logger.debug(
                f"Tokens — input: {usage.input_tokens}, "
                f"cache_write: {getattr(usage, 'cache_creation_input_tokens', 0)}, "
                f"cache_read: {getattr(usage, 'cache_read_input_tokens', 0)}, "
                f"output: {usage.output_tokens}"
            )
            return resp.content[0].text
        except anthropic.RateLimitError:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(f"Rate limited, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
        except anthropic.APIError as e:
            if attempt < retries - 1:
                logger.warning(f"API error ({e}), retrying...")
                time.sleep(2)
            else:
                raise
    raise RuntimeError("All retries exhausted")


def get_model(cfg: dict, key: str) -> str:
    """Return the right model ID for the active LLM provider."""
    if _provider() == "openrouter":
        return cfg["llm"].get(f"openrouter_{key}", cfg["llm"][key])
    return cfg["llm"][key]


def _call_openrouter(model, system_blocks, user_message, max_tokens, retries):
    client = _get_openrouter_client()
    system_text = "\n\n".join(b["text"] for b in system_blocks if b.get("type") == "text")

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_message},
                ],
            )
            logger.debug(
                f"Tokens — input: {resp.usage.prompt_tokens}, output: {resp.usage.completion_tokens}"
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < retries - 1:
                logger.warning(f"OpenRouter error ({e}), retrying in {2 ** (attempt + 1)}s...")
                time.sleep(2 ** (attempt + 1))
            else:
                raise
    raise RuntimeError("All retries exhausted")


def call_simple(model: str, system: str, user: str, max_tokens: int = 1000) -> str:
    return call_with_cache(
        model=model,
        system_blocks=[{"type": "text", "text": system}],
        user_message=user,
        max_tokens=max_tokens,
    )
