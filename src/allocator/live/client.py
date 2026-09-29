"""Anthropic Messages API over httpx. Tests inject a stub instead of this."""

from __future__ import annotations

from typing import Any, Protocol

import httpx

from .config import API_URL, MODEL


class Completer(Protocol):
    def __call__(
        self, *, prompt: str, max_tokens: int
    ) -> tuple[str, dict[str, int]]:
        """Return (completion_text, usage dict)."""


def _response_detail(response: httpx.Response) -> str:
    try:
        data = response.json()
        err = data.get("error", data)
        return str(err)[:400]
    except ValueError:
        return response.text[:400]


def anthropic_completer(
    *,
    api_key: str,
    model: str = MODEL,
    timeout: float = 30.0,
    temperature: float = 0.0,
) -> Completer:
    """Build a completer that POSTs to Anthropic. Never log the key.

    v3 collect uses ``temperature=0``. v4 k-samples use ``temperature=0.5``.
    """

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }

    def complete(*, prompt: str, max_tokens: int) -> tuple[str, dict[str, int]]:
        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        last_exc: Exception | None = None
        for _attempt in range(3):
            try:
                response = httpx.post(API_URL, headers=headers, json=payload, timeout=timeout)
                if response.status_code == 404:
                    raise RuntimeError(
                        f"Anthropic 404 for model {model!r}: {_response_detail(response)}. "
                        "Use an id from GET /v1/models (this key lists claude-haiku-4-5-20251001)."
                    )
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    raise RuntimeError(
                        f"Anthropic HTTP {response.status_code}: {_response_detail(response)}"
                    )
                response.raise_for_status()
                body: dict[str, Any] = response.json()
                blocks = body.get("content") or []
                texts = [
                    str(block.get("text") or "")
                    for block in blocks
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                text = "".join(texts)
                raw_usage = body.get("usage") or {}
                prompt_tokens = int(raw_usage.get("input_tokens") or 0)
                completion_tokens = int(raw_usage.get("output_tokens") or 0)
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
                return text, usage
            except RuntimeError:
                raise
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                last_exc = exc
        raise RuntimeError(f"Anthropic completion failed: {last_exc}") from last_exc

    return complete
