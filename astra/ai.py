"""Disabled-by-default, advisory-only provider interface for external AI."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class AIProviderError(RuntimeError):
    """Sanitized provider error; never includes prompts, keys, or response bodies."""


@dataclass(frozen=True)
class AIResponse:
    text: str
    model: str
    request_id: str = ""
    usage: dict | None = None


class AIProvider(Protocol):
    def complete(self, prompt: str, *, system: str = "") -> AIResponse: ...


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


class OpenRouterProvider:
    """Bounded OpenRouter chat completion client with injected transport support."""

    max_request_bytes = 32_768
    max_response_bytes = 65_536

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "openrouter/free",
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 10.0,
        max_requests: int = 20,
        transport=None,
    ):
        parsed = urlsplit(base_url)
        if (
            not api_key
            or parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("OpenRouter requires a key and plain HTTPS API URL")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}", model):
            raise ValueError("invalid OpenRouter model identifier")
        if timeout <= 0 or max_requests < 1:
            raise ValueError("timeout and request limit must be positive")
        self._api_key = api_key
        self.model = model
        self._endpoint = base_url.rstrip("/") + "/chat/completions"
        self._timeout = timeout
        self._max_requests = max_requests
        self._requests = 0
        self._transport = transport or self._http

    @staticmethod
    def _http(request: Request, timeout: float):
        opener = build_opener(_NoRedirect())
        with opener.open(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            if not content_type.lower().startswith("application/json"):
                raise AIProviderError("OpenRouter returned an unsupported content type")
            return response.read(OpenRouterProvider.max_response_bytes + 1)

    def complete(self, prompt: str, *, system: str = "") -> AIResponse:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("AI prompt must be nonempty text")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = json.dumps(
            {"model": self.model, "messages": messages, "stream": False},
            allow_nan=False,
        ).encode("utf-8")
        if len(body) > self.max_request_bytes:
            raise ValueError("AI request exceeds the configured size limit")
        if self._requests >= self._max_requests:
            raise AIProviderError("OpenRouter request limit reached")
        self._requests += 1
        request = Request(
            self._endpoint,
            data=body,
            headers={
                "Authorization": "Bearer " + self._api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            raw = self._transport(request, self._timeout)
        except HTTPError as error:
            raise AIProviderError(
                f"OpenRouter request failed (HTTP {error.code})"
            ) from None
        except AIProviderError:
            raise
        except Exception:  # noqa: BLE001 - injected transports are an external boundary.
            raise AIProviderError("OpenRouter request failed") from None
        if not isinstance(raw, bytes) or len(raw) > self.max_response_bytes:
            raise AIProviderError("OpenRouter response exceeded the size limit")
        try:
            response = json.loads(raw)
            choice = response["choices"][0]["message"]["content"]
            model = response["model"]
            if not isinstance(choice, str) or not isinstance(model, str):
                raise TypeError
            usage = response.get("usage")
            if usage is not None and not isinstance(usage, dict):
                raise ValueError
            return AIResponse(choice, model, str(response.get("id", "")), usage)
        except (ValueError, TypeError, KeyError, IndexError):
            raise AIProviderError("OpenRouter returned an invalid completion") from None


def provider_from_settings(settings, *, transport=None) -> AIProvider | None:
    """Return no provider while AI is disabled; callers never enable it implicitly."""
    settings.validate()
    if not settings.ai_enabled:
        return None
    if settings.ai_provider != "openrouter":
        raise ValueError("unknown AI provider")
    return OpenRouterProvider(
        settings.ai_api_key or "",
        model=settings.ai_model,
        base_url=settings.ai_base_url,
        timeout=settings.ai_timeout_seconds,
        max_requests=settings.ai_max_requests,
        transport=transport,
    )
