# AI integration status

The provider interface and OpenRouter chat-completion adapter are implemented, but **AI remains disabled by default and has not been activated or called**. `Settings.ai_enabled` must be explicitly enabled and a local `ASTRA_AI_API_KEY` supplied before a provider can be constructed. The key is excluded from configuration representations and public output. No key was requested or read for this work.

The default model is `openrouter/free`, OpenRouter's free-model router. It selects a compatible free model, so the actual response model is recorded in `AIResponse.model` and can change over time. The adapter does not enable tools, streaming, web search, or code execution. It enforces HTTPS, rejects redirects, caps request and response sizes, applies a timeout and per-instance request limit, and raises sanitized errors. Tests inject deterministic transports and make no network calls. The adapter returns advisory text only; it has no access to order, risk, persistence, or configuration APIs.

Configuration placeholders are in `.env.example`:

```dotenv
ASTRA_AI_ENABLED=false
ASTRA_AI_PROVIDER=openrouter
ASTRA_AI_API_KEY=
ASTRA_AI_MODEL=openrouter/free
ASTRA_AI_BASE_URL=https://openrouter.ai/api/v1
ASTRA_AI_TIMEOUT_SECONDS=10
ASTRA_AI_MAX_REQUESTS=20
```

Supply a key locally in `.env` only if a later, explicit phase enables AI. Check the provider's current [free model router](https://openrouter.ai/collections/free-models) and [chat completion API](https://openrouter.ai/docs/api-reference/chat-completion) before use. Free model availability and rate limits can change; no inference availability or response quality is guaranteed. Do not treat AI output as a trading instruction or calibrated probability.
