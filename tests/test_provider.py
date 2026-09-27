"""Run inside a Hermes install: ``python -m unittest discover -s tests`` with the Hermes source on PYTHONPATH."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "responses_web_search.json"


def _load_provider():
    spec = importlib.util.spec_from_file_location("web_meta_ai_provider", ROOT / "provider.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


provider = _load_provider()


def _message(text, annotations=None):
    return {"type": "message", "content": [{"type": "output_text", "text": text, "annotations": annotations or []}]}


class ExtractResultsTest(unittest.TestCase):
    def test_recorded_meta_response_yields_final_json_rows(self):
        data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        rows = provider.extract_results(data, limit=5)
        self.assertTrue(rows)
        self.assertLessEqual(len(rows), 5)
        self.assertEqual([r["position"] for r in rows], list(range(1, len(rows) + 1)))
        self.assertTrue(all(r["url"].startswith("https://") and r["title"] for r in rows))

    def test_json_after_narration_and_limit(self):
        body = {"results": [{"title": f"T{i}", "url": f"https://e.com/{i}", "description": "d"} for i in range(4)]}
        data = {"output": [_message("I'll search."), _message("Here you go: " + json.dumps(body))]}
        rows = provider.extract_results(data, limit=2)
        self.assertEqual([r["url"] for r in rows], ["https://e.com/0", "https://e.com/1"])

    def test_drops_rows_without_absolute_urls_and_renumbers(self):
        body = {"results": [{"title": "bad", "url": "/relative"}, {"title": "ok", "url": "https://ok.example"}]}
        rows = provider.extract_results({"output": [_message(json.dumps(body))]}, limit=5)
        self.assertEqual(rows, [{"title": "ok", "url": "https://ok.example", "description": "", "position": 1}])

    def test_empty_results_is_empty_list(self):
        self.assertEqual(provider.extract_results({"output": [_message('{"results": []}')]}, limit=5), [])

    def test_falls_back_to_url_citations_then_opened_pages(self):
        cited = {"output": [_message("prose", [{"type": "url_citation", "url": "https://a.example", "title": "A"}])]}
        self.assertEqual(provider.extract_results(cited, limit=5)[0]["url"], "https://a.example")
        opened = {"output": [
            {"type": "web_search_call", "action": {"type": "search", "query": "q"}},
            {"type": "web_search_call", "action": {"type": "open_page", "url": "https://b.example"}},
            {"type": "web_search_call", "action": {"type": "open_page", "url": "https://b.example"}},
            _message("no json here"),
        ]}
        self.assertEqual([r["url"] for r in provider.extract_results(opened, limit=5)], ["https://b.example"])


class SearchTest(unittest.TestCase):
    def setUp(self):
        self.p = provider.MetaAIWebSearchProvider()

    def test_identity(self):
        self.assertEqual(self.p.name, "meta-ai")
        self.assertTrue(self.p.supports_search())
        self.assertFalse(self.p.supports_extract())

    def test_unavailable_without_key(self):
        with mock.patch.object(provider, "get_provider_env", return_value=""):
            self.assertFalse(self.p.is_available())
            result = self.p.search("q")
        self.assertFalse(result["success"])
        self.assertIn("MODEL_API_KEY", result["error"])

    def test_request_shape_and_config(self):
        env = {"MODEL_API_KEY": "k", "META_BASE_URL": ""}
        captured = {}

        def fake_post(base_url, payload, api_key, timeout):
            captured.update(base_url=base_url, payload=payload, api_key=api_key, timeout=timeout)
            return {"status": "completed", "output": [_message('{"results": []}')]}, None

        cfg = {"model": "muse-spark-1.3-contributor", "timeout": 30}
        with mock.patch.object(provider, "get_provider_env", side_effect=lambda n: env.get(n, "")), \
             mock.patch.object(provider, "_load_config", return_value=cfg), \
             mock.patch.object(provider, "_post_responses", side_effect=fake_post):
            result = self.p.search("toronto weather", limit=3)
        self.assertEqual(result, {"success": True, "data": {"web": []}})
        self.assertEqual(captured["base_url"], "https://api.meta.ai/v1")
        self.assertEqual(captured["api_key"], "k")
        self.assertEqual(captured["timeout"], 30.0)
        self.assertEqual(captured["payload"]["model"], "muse-spark-1.3-contributor")
        self.assertEqual(captured["payload"]["tools"], [{"type": "web_search"}])
        self.assertIn("toronto weather", captured["payload"]["input"][0]["content"])

    def test_error_envelope_and_incomplete_status_fail(self):
        with mock.patch.object(provider, "get_provider_env", side_effect=lambda n: "k" if n == "MODEL_API_KEY" else ""), \
             mock.patch.object(provider, "_load_config", return_value={}):
            with mock.patch.object(provider, "_post_responses", return_value=({"error": {"message": "boom"}}, None)):
                self.assertEqual(self.p.search("q"), {"success": False, "error": "Meta returned an error: boom"})
            with mock.patch.object(provider, "_post_responses", return_value=({"status": "incomplete", "output": []}, None)):
                self.assertFalse(self.p.search("q")["success"])


if __name__ == "__main__":
    unittest.main()
