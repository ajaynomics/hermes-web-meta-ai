"""Meta web search plugin — user-installed, opt-in. Registers the ``meta-ai`` web search provider."""

from __future__ import annotations

from .provider import MetaAIWebSearchProvider


def register(ctx) -> None:
    ctx.register_web_search_provider(MetaAIWebSearchProvider())
