import json

import pytest

from astra.ai import AIProviderError, OpenRouterProvider, provider_from_settings
from astra.config import Settings


class FakeTransport:
    def __init__(self, raw=None):
        self.raw = (
            raw
            or json.dumps(
                {
                    "id": "request-1",
                    "model": "vendor/free-model",
                    "choices": [{"message": {"content": "Advisory summary."}}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 3},
                }
            ).encode()
        )
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append((request, timeout))
        return self.raw


def test_ai_provider_is_absent_until_explicitly_enabled():
    assert provider_from_settings(Settings(ai_api_key="local-secret")) is None
    with pytest.raises(ValueError, match="API_KEY"):
        provider_from_settings(Settings(ai_enabled=True))


def test_openrouter_free_router_uses_bounded_advisory_chat_request():
    transport = FakeTransport()
    provider = OpenRouterProvider("local-secret", max_requests=1, transport=transport)

    result = provider.complete("Summarize this test result.", system="Advisory only.")

    request, timeout = transport.requests[0]
    body = json.loads(request.data)
    assert body["model"] == "openrouter/free"
    assert body["stream"] is False
    assert "tools" not in body
    assert request.get_header("Authorization") == "Bearer local-secret"
    assert timeout == 10.0
    assert result.text == "Advisory summary."
    assert result.model == "vendor/free-model"
    assert result.usage["completion_tokens"] == 3
    with pytest.raises(AIProviderError, match="request limit"):
        provider.complete("A second call is blocked.")


def test_openrouter_rejects_unsafe_endpoints_oversized_and_invalid_responses():
    with pytest.raises(ValueError, match="HTTPS"):
        OpenRouterProvider("key", base_url="http://example.invalid/v1")

    provider = OpenRouterProvider("key", transport=FakeTransport())
    with pytest.raises(ValueError, match="size limit"):
        provider.complete("x" * provider.max_request_bytes)

    malformed = OpenRouterProvider("key", transport=FakeTransport(b"{}"))
    with pytest.raises(AIProviderError, match="invalid completion") as error:
        malformed.complete("safe test")
    assert "local-secret" not in str(error.value)
