"""Meta Model API web search — a search-only Hermes web backend.

Routes Hermes' ``web_search`` tool through the server-side ``web_search`` built-in on Meta's
Responses API (``POST {base_url}/responses`` with ``tools=[{"type": "web_search"}]``). Muse Spark
searches, opens pages, and returns structured JSON rows, so results match every other Hermes web
provider. This mirrors the bundled xAI backend (``plugins/web/xai``) but imports only the public
``agent.web_search_provider`` surface.

Config::

    web:
      search_backend: meta-ai        # search only; web_extract keeps its own backend
      meta-ai:                       # all optional
        model: muse-spark-1.3        # default muse-spark-1.3
        base_url: https://api.meta.ai/v1
        timeout: 120                 # seconds

Auth: the same env vars as Hermes' bundled ``meta-ai`` model provider — ``MODEL_API_KEY`` (Meta's
documented name), ``META_API_KEY`` or ``META_MODEL_API_KEY``. ``META_BASE_URL`` overrides the base URL.

Trust model: like the xAI backend, Muse Spark *writes* the titles/descriptions and chooses the URLs,
steered by the query text. Treat returned URLs as model-generated links.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from agent.web_search_provider import WebSearchProvider, get_provider_env

logger = logging.getLogger(__name__)

PROVIDER_NAME = "meta-ai"
DEFAULT_MODEL = "muse-spark-1.3"
DEFAULT_BASE_URL = "https://api.meta.ai/v1"
DEFAULT_TIMEOUT = 120.0
API_KEY_ENV_VARS = ("MODEL_API_KEY", "META_API_KEY", "META_MODEL_API_KEY")

# Muse Spark narrates between tool calls ("I'll search for…"), so the JSON object usually sits in
# the last output_text block and may carry leading prose.
_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def _row(title: str, url: str, description: str, position: int) -> Dict[str, Any]:
    return {"title": title, "url": url, "description": description, "position": position}


def _ok(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"success": True, "data": {"web": rows}}


def _fail(error: str) -> Dict[str, Any]:
    return {"success": False, "error": error}


def _load_config() -> Dict[str, Any]:
    """``web.meta-ai`` from config.yaml; {} on miss."""
    try:
        from hermes_cli.config import load_config

        section = (load_config().get("web") or {}).get(PROVIDER_NAME)
        return section if isinstance(section, dict) else {}
    except Exception as exc:  # noqa: BLE001 — config layer is best-effort
        logger.debug("Could not load web.%s config: %s", PROVIDER_NAME, exc)
        return {}


def _api_key() -> str:
    return next((v for v in (get_provider_env(n) for n in API_KEY_ENV_VARS) if v), "")


def _str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _timeout(value: Any) -> float:
    try:
        return max(1.0, float(value))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT


class MetaAIWebSearchProvider(WebSearchProvider):
    """Search-only provider backed by Meta's server-side ``web_search`` built-in."""

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def display_name(self) -> str:
        return "Meta Web Search (Muse Spark)"

    def is_available(self) -> bool:
        """Env probe only — never touches the network (runs on every ``hermes tools`` paint)."""
        return bool(_api_key())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return _fail("Interrupted")
        except Exception:  # noqa: BLE001 — interrupt module is best-effort
            pass
        api_key = _api_key()
        if not api_key:
            return _fail(f"No Meta Model API key found. Set one of: {', '.join(API_KEY_ENV_VARS)}.")
        try:
            limit = max(1, min(int(limit), 100))
        except (TypeError, ValueError):
            limit = 5
        cfg = _load_config()
        model = _str(cfg.get("model")) or DEFAULT_MODEL
        base_url = (_str(cfg.get("base_url")) or get_provider_env("META_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        payload = {
            "model": model,
            "input": [{"role": "user", "content": build_prompt(query, limit)}],
            "tools": [{"type": "web_search"}],
        }
        logger.info("Meta web search via %s: %r (limit=%d, model=%s)", base_url, query, limit, model)
        data, error = _post_responses(base_url, payload, api_key, _timeout(cfg.get("timeout", DEFAULT_TIMEOUT)))
        if error:
            return error
        api_error = data.get("error") if isinstance(data, dict) else None
        if isinstance(api_error, dict):
            return _fail(f"Meta returned an error: {api_error.get('message') or api_error.get('code') or 'unknown error'}")
        status = data.get("status") if isinstance(data, dict) else None
        if status not in (None, "completed"):
            return _fail(f"Meta web search did not complete (status={status}).")
        # Zero rows is a success (matches the bundled providers).
        return _ok(extract_results(data, limit=limit))

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "paid",
            "tag": "Server-side web_search on Meta's Responses API (Muse Spark).",
            "env_vars": [{"key": "MODEL_API_KEY", "prompt": "Meta Model API key", "url": "https://dev.meta.ai/"}],
        }


def _post_responses(base_url: str, payload: Dict[str, Any], api_key: str, timeout: float) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """POST ``/responses`` → ``(json, None)`` or ``(None, failure_envelope)``."""
    try:
        import httpx
    except ImportError:
        return None, _fail("httpx is not installed (required for Meta web search)")
    try:
        resp = httpx.post(
            f"{base_url}/responses",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else 0
        body = exc.response.text[:300] if exc.response is not None else ""
        logger.warning("Meta web search HTTP %d: %s", status, body)
        return None, _fail(f"Meta web search returned HTTP {status}: {body}".rstrip())
    except httpx.RequestError as exc:
        logger.warning("Meta web search request error: %s", exc)
        return None, _fail(f"Could not reach Meta Model API: {exc}")
    try:
        return resp.json(), None
    except ValueError as exc:
        logger.warning("Meta web search bad JSON: %s", exc)
        return None, _fail("Could not parse Meta Responses API reply as JSON")


def build_prompt(query: str, limit: int) -> str:
    return (
        "Use the web_search tool to find current information for the query below, then respond with ONLY a single "
        "JSON object — no prose, no markdown fences, no inline citation links — matching this exact schema:\n\n"
        '{"results": [{"title": "string", "url": "string", "description": "1-2 sentence summary"}]}\n\n'
        f"Return at most {limit} results, ordered by relevance, with absolute https:// URLs. If no usable results "
        'exist, return {"results": []}.\n\n'
        f"Query: {query}"
    )


def extract_results(response: Dict[str, Any], *, limit: int) -> List[Dict[str, Any]]:
    """Rows from the last ``output_text`` block holding a JSON ``results`` array; otherwise
    ``url_citation`` annotations; otherwise pages the server-side tool opened."""
    texts, annotations = _output_text(response)
    for text in reversed(texts):
        rows = _parse_json_rows(text, limit=limit)
        if rows:
            return rows
    rows = _rows_from_urls((a.get("url"), a.get("title")) for a in annotations if a.get("type") == "url_citation")
    if rows:
        return rows[:limit]
    opened = (
        ((item.get("action") or {}).get("url"), "")
        for item in response.get("output") or []
        if isinstance(item, dict) and item.get("type") == "web_search_call"
        and isinstance(item.get("action"), dict) and item["action"].get("type") == "open_page"
    )
    return _rows_from_urls(opened)[:limit]


def _output_text(response: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
    chunks = [
        chunk
        for item in response.get("output") or []
        if isinstance(item, dict) and item.get("type") == "message" and isinstance(item.get("content"), list)
        for chunk in item["content"]
        if isinstance(chunk, dict) and chunk.get("type") == "output_text"
    ]
    texts = [c["text"] for c in chunks if isinstance(c.get("text"), str) and c["text"].strip()]
    annotations = [a for c in chunks if isinstance(c.get("annotations"), list) for a in c["annotations"] if isinstance(a, dict)]
    return texts, annotations


def _parse_json_rows(text: str, *, limit: int) -> List[Dict[str, Any]]:
    match = _JSON_BLOCK_RE.search(text)
    for candidate in dict.fromkeys([text, match.group(0) if match else text]):
        try:
            parsed = json.loads(candidate)
        except ValueError:
            continue
        results = parsed.get("results") if isinstance(parsed, dict) else None
        if not isinstance(results, list):
            continue
        rows: List[Dict[str, Any]] = []
        for item in results:
            url = _str(item.get("url")) if isinstance(item, dict) else ""
            if url.startswith(("https://", "http://")):
                rows.append(_row(_str(item.get("title")), url, _str(item.get("description")), len(rows) + 1))
            if len(rows) >= limit:
                break
        return rows
    return []


def _rows_from_urls(pairs) -> List[Dict[str, Any]]:
    seen: set = set()
    rows: List[Dict[str, Any]] = []
    for url, title in pairs:
        url = _str(url)
        if url.startswith(("https://", "http://")) and url not in seen:
            seen.add(url)
            rows.append(_row(_str(title), url, "", len(rows) + 1))
    return rows
