"""Anthropic Direct API provider using native Messages endpoint."""

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.defaults import ANTHROPIC_DEFAULT_BASE
from free_claude_code.providers.model_listing import model_infos_from_ids
from free_claude_code.providers.rate_limit import ProviderRateLimiter
from free_claude_code.providers.transports.anthropic_messages import (
    AnthropicMessagesTransport,
)

_ANTHROPIC_MODEL_IDS = frozenset(
    {
        "claude-sonnet-5",
        "claude-fable-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-opus-4-5-20251101",
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-1-20250805",
    }
)


class AnthropicDirectProvider(AnthropicMessagesTransport):
    """Anthropic Direct API provider using native Messages endpoint.

    Connects directly to the Anthropic Messages API at
    ``https://api.anthropic.com/v1/messages`` (the ``ANTHROPIC_DEFAULT_BASE``).
    Uses a static model list since Anthropic does not expose a public
    ``/models`` endpoint.
    """

    def __init__(
        self,
        config: ProviderConfig,
        *,
        rate_limiter: ProviderRateLimiter,
    ):
        super().__init__(
            config,
            provider_name="ANTHROPIC",
            default_base_url=ANTHROPIC_DEFAULT_BASE,
            rate_limiter=rate_limiter,
        )

    def _request_headers(self) -> dict[str, str]:
        """Return headers for the native messages request."""
        return {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def _model_list_headers(self) -> dict[str, str]:
        """Return headers for model-list requests."""
        return {"x-api-key": self._api_key}

    async def list_model_ids(self) -> frozenset[str]:
        """Return the static list of known Claude model ids.

        Anthropic does not expose a public ``/models`` endpoint, so the
        model list is maintained statically in ``_ANTHROPIC_MODEL_IDS``.
        """
        return _ANTHROPIC_MODEL_IDS

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """Return model metadata including thinking support for all Claude models."""
        return model_infos_from_ids(_ANTHROPIC_MODEL_IDS, supports_thinking=True)
