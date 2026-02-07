"""
test_helperbot.py – Unit tests for HelperBot.

Covers: config validation, trigger regex, image extraction, transcript
building, LLM message helpers, web_search tool, web_fetch tool,
web_render tool, caching, deduplication, HTML parsing, the ai_answer
tool-calling loop, and retry logic.
"""

import json
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


# ── Fake objects for testing ─────────────────────────────────────────────


class FakeSubreddit:
    display_name = "testsub"


class FakeSubmission:
    title = "Test Submission"
    selftext = "Test selftext"
    is_self = True
    permalink = "/r/testsub/comments/test_submission"
    url = ""
    post_hint = ""


class FakeAuthor:
    name = "testuser"


class FakeComment:
    def __init__(self, body: str):
        self.body = body
        self.id = "test_id"
        self.submission = FakeSubmission()
        self.subreddit = FakeSubreddit()
        self.author = FakeAuthor()

    def parent(self):
        return self

    @property
    def is_root(self):
        return True

    def reply(self, text: str) -> None:
        pass


# ── Config tests ─────────────────────────────────────────────────────────


class TestConfig(unittest.TestCase):
    def test_validate_env_exits_on_missing_vars(self):
        """validate_env should sys.exit when required vars are missing."""
        import config

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit):
                config.validate_env()

    def test_validate_env_passes_with_all_vars(self):
        """validate_env should not raise when all required vars are set."""
        import config

        env = {var: "test_value" for var in config.REQUIRED_ENV_VARS}
        with patch.dict(os.environ, env, clear=False):
            config.validate_env()

    def test_required_env_vars_list(self):
        """Ensure we check for all the critical env vars."""
        import config

        expected = {
            "OPENROUTER_API_KEY",
            "REDDIT_CLIENT_ID",
            "REDDIT_CLIENT_SECRET",
            "REDDIT_USERNAME",
            "REDDIT_PASSWORD",
            "USER_AGENT",
            "SEARXNG_BASE_URL",
        }
        self.assertEqual(set(config.REQUIRED_ENV_VARS), expected)


# ── Trigger regex tests ─────────────────────────────────────────────────


class TestTriggerRegex(unittest.TestCase):
    def test_matches_valid_triggers(self):
        import config

        valid = [
            "u/grok test",
            "@grok test",
            "   u/grok test",
            "[u/grok test",
            "@ai what is this?",
            "u/gemini hello",
            "@chatgpt explain",
            "u/gpt help me",
            "@Grok uppercase",
            "  @AI leading space",
        ]
        for s in valid:
            with self.subTest(s=s):
                self.assertIsNotNone(config.TRIGGER.match(s))

    def test_rejects_invalid_triggers(self):
        import config

        invalid = [
            "groktest",
            "something else",
            "hello u/grok",
            "not@grok",
            "",
            "random text",
        ]
        for s in invalid:
            with self.subTest(s=s):
                self.assertIsNone(config.TRIGGER.match(s))


# ── Image extraction tests ──────────────────────────────────────────────


class TestImageExtraction(unittest.TestCase):
    def test_extracts_direct_urls(self):
        from transcript import extract_image_urls_from_text

        text = "Look at this https://example.com/photo.jpg and https://cdn.site.io/img.png"
        urls = extract_image_urls_from_text(text)
        self.assertEqual(len(urls), 2)
        self.assertIn("https://example.com/photo.jpg", urls)
        self.assertIn("https://cdn.site.io/img.png", urls)

    def test_extracts_markdown_image_urls(self):
        from transcript import extract_image_urls_from_text

        text = "Here is ![alt](https://example.com/pic.jpeg) an image"
        urls = extract_image_urls_from_text(text)
        self.assertIn("https://example.com/pic.jpeg", urls)

    def test_deduplicates_urls(self):
        from transcript import extract_image_urls_from_text

        text = "https://example.com/a.png and https://example.com/a.png again"
        urls = extract_image_urls_from_text(text)
        self.assertEqual(len(urls), 1)

    def test_returns_empty_for_none(self):
        from transcript import extract_image_urls_from_text

        self.assertEqual(extract_image_urls_from_text(None), [])
        self.assertEqual(extract_image_urls_from_text(""), [])

    def test_extracts_various_extensions(self):
        from transcript import extract_image_urls_from_text

        text = (
            "https://a.com/x.gif https://b.com/y.webp "
            "https://c.com/z.bmp https://d.com/w.jpeg"
        )
        urls = extract_image_urls_from_text(text)
        self.assertEqual(len(urls), 4)


# ── Transcript building tests ───────────────────────────────────────────


class TestBuildThreadTranscript(unittest.TestCase):
    def test_basic_transcript_structure(self):
        from transcript import build_thread_transcript

        comment = FakeComment("u/grok What is Python?")
        transcript, images = build_thread_transcript(comment)

        self.assertIn("SUBREDDIT: r/testsub", transcript)
        self.assertIn("SUBMISSION TITLE: Test Submission", transcript)
        self.assertIn("testuser wrote:", transcript)
        self.assertIn("u/grok What is Python?", transcript)

    def test_transcript_includes_selftext(self):
        from transcript import build_thread_transcript

        comment = FakeComment("u/grok test")
        transcript, _ = build_thread_transcript(comment)
        self.assertIn("Test selftext", transcript)

    def test_transcript_extracts_images_from_body(self):
        from transcript import build_thread_transcript

        comment = FakeComment("u/grok check https://example.com/img.png")
        _, images = build_thread_transcript(comment)
        self.assertIn("https://example.com/img.png", images)

    def test_transcript_handles_image_submission(self):
        from transcript import build_thread_transcript

        comment = FakeComment("u/grok describe this")
        comment.submission = FakeSubmission()
        comment.submission.url = "https://i.imgur.com/abc.jpg"
        comment.submission.is_self = False
        _, images = build_thread_transcript(comment)
        self.assertIn("https://i.imgur.com/abc.jpg", images)

    def test_transcript_respects_max_chars(self):
        from transcript import build_thread_transcript

        with patch("config.MAX_CHARS", 100):
            comment = FakeComment("u/grok " + "x" * 200)
            transcript, _ = build_thread_transcript(comment)
            self.assertLessEqual(len(transcript), 100)

    def test_deleted_author_shows_placeholder(self):
        from transcript import build_thread_transcript

        comment = FakeComment("u/grok test")
        comment.author = None
        transcript, _ = build_thread_transcript(comment)
        self.assertIn("[deleted]", transcript)

    def test_gallery_image_extraction(self):
        from transcript import build_thread_transcript

        comment = FakeComment("u/grok test")
        comment.submission = FakeSubmission()
        comment.submission.is_self = False
        comment.submission.is_gallery = True
        comment.submission.media_metadata = {
            "item1": {
                "m": "image/jpeg",
                "e": "Image",
                "s": {"u": "https://preview.redd.it/img1.jpg?auto=webp&amp;s=abc"},
            },
            "item2": {
                "m": "image/png",
                "e": "Image",
                "s": {"u": "https://preview.redd.it/img2.png"},
            },
        }
        _, images = build_thread_transcript(comment)
        self.assertTrue(len(images) >= 2)
        self.assertTrue(any("&amp;" not in url for url in images))


# ── LLM message helper tests ────────────────────────────────────────────


class TestMessageHelpers(unittest.TestCase):
    def test_message_content_to_text_string(self):
        from llm import message_content_to_text

        self.assertEqual(message_content_to_text("hello"), "hello")

    def test_message_content_to_text_list(self):
        from llm import message_content_to_text

        content = [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}]
        result = message_content_to_text(content)
        self.assertIn("part1", result)
        self.assertIn("part2", result)

    def test_message_content_to_text_empty(self):
        from llm import message_content_to_text

        self.assertEqual(message_content_to_text(None), "")
        self.assertEqual(message_content_to_text(123), "")

    def test_truncate_for_log_short(self):
        from llm import truncate_for_log

        self.assertEqual(truncate_for_log("short text"), "short text")

    def test_truncate_for_log_long(self):
        from llm import truncate_for_log

        text = "x" * 2000
        result = truncate_for_log(text, max_chars=100)
        self.assertEqual(len(result), 100 + len("... [truncated]"))
        self.assertTrue(result.endswith("... [truncated]"))

    def test_message_to_dict_with_model_dump(self):
        from llm import message_to_dict

        obj = SimpleNamespace()
        obj.model_dump = lambda exclude_none=True: {"role": "assistant", "content": "hi"}
        self.assertEqual(message_to_dict(obj), {"role": "assistant", "content": "hi"})

    def test_message_to_dict_plain_dict(self):
        from llm import message_to_dict

        d = {"role": "user", "content": "test"}
        self.assertEqual(message_to_dict(d), d)

    def test_message_to_dict_unknown_type(self):
        from llm import message_to_dict

        self.assertEqual(message_to_dict(42), {})

    def test_extract_reasoning_string(self):
        from llm import extract_reasoning_for_log

        msg = SimpleNamespace(reasoning="thinking hard")
        msg.model_dump = lambda exclude_none=True: {"reasoning": "thinking hard"}
        result = extract_reasoning_for_log(msg)
        self.assertIn("thinking hard", result)

    def test_extract_reasoning_none(self):
        from llm import extract_reasoning_for_log

        msg = SimpleNamespace(reasoning=None)
        msg.model_dump = lambda exclude_none=True: {}
        self.assertEqual(extract_reasoning_for_log(msg), "")


# ── Web search tool tests ───────────────────────────────────────────────


class TestWebSearchTool(unittest.TestCase):
    def test_empty_query_returns_error(self):
        from tools import run_web_search_tool

        result = run_web_search_tool({"query": ""})
        self.assertIn("error", result)

    def test_missing_query_returns_error(self):
        from tools import run_web_search_tool

        result = run_web_search_tool({})
        self.assertIn("error", result)

    @patch("tools.requests.get")
    def test_returns_formatted_results(self, mock_get):
        from tools import run_web_search_tool

        fake_resp = MagicMock()
        fake_resp.raise_for_status.return_value = None
        fake_resp.json.return_value = {
            "results": [
                {
                    "title": "Example Result",
                    "url": "https://example.com",
                    "content": "A snippet",
                    "engines": ["google"],
                }
            ]
        }
        mock_get.return_value = fake_resp

        result = run_web_search_tool({"query": "test query"})
        self.assertEqual(result["result_count"], 1)
        self.assertEqual(result["results"][0]["title"], "Example Result")
        self.assertEqual(result["results"][0]["snippet"], "A snippet")

    @patch("tools.requests.get")
    def test_handles_searxng_failure(self, mock_get):
        from tools import run_web_search_tool

        mock_get.side_effect = ConnectionError("connection refused")

        result = run_web_search_tool({"query": "test"})
        self.assertEqual(result["result_count"], 0)
        self.assertEqual(result["results"], [])

    @patch("tools.requests.get")
    def test_respects_max_results(self, mock_get):
        from tools import run_web_search_tool

        fake_resp = MagicMock()
        fake_resp.raise_for_status.return_value = None
        fake_resp.json.return_value = {
            "results": [
                {"title": f"Result {i}", "url": f"https://example.com/{i}", "content": ""}
                for i in range(10)
            ]
        }
        mock_get.return_value = fake_resp

        result = run_web_search_tool({"query": "test", "max_results": 3})
        self.assertEqual(result["result_count"], 3)

    @patch("tools.requests.get")
    def test_passes_categories_and_time_range(self, mock_get):
        from tools import run_web_search_tool

        fake_resp = MagicMock()
        fake_resp.raise_for_status.return_value = None
        fake_resp.json.return_value = {"results": []}
        mock_get.return_value = fake_resp

        run_web_search_tool({
            "query": "news",
            "categories": ["news"],
            "time_range": "day",
        })

        call_kwargs = mock_get.call_args
        params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params")
        self.assertEqual(params["categories"], "news")
        self.assertEqual(params["time_range"], "day")

    def test_format_tool_search_results_handles_missing_fields(self):
        from tools import format_tool_search_results

        results = [{"title": None, "url": None, "content": None}]
        formatted = format_tool_search_results(results)
        self.assertEqual(formatted[0]["title"], "")
        self.assertEqual(formatted[0]["url"], "")
        self.assertEqual(formatted[0]["snippet"], "")
        self.assertEqual(formatted[0]["engines"], [])

    @patch("tools.requests.get")
    def test_includes_published_date(self, mock_get):
        from tools import run_web_search_tool

        fake_resp = MagicMock()
        fake_resp.raise_for_status.return_value = None
        fake_resp.json.return_value = {
            "results": [
                {
                    "title": "News Article",
                    "url": "https://example.com/news",
                    "content": "Breaking news",
                    "engines": ["google"],
                    "publishedDate": "2026-02-01T12:00:00Z",
                }
            ]
        }
        mock_get.return_value = fake_resp

        result = run_web_search_tool({"query": "news"})
        self.assertEqual(result["results"][0]["published_date"], "2026-02-01T12:00:00Z")

    @patch("tools.requests.get")
    def test_omits_published_date_when_absent(self, mock_get):
        from tools import run_web_search_tool

        fake_resp = MagicMock()
        fake_resp.raise_for_status.return_value = None
        fake_resp.json.return_value = {
            "results": [
                {"title": "No Date", "url": "https://example.com", "content": "", "engines": []}
            ]
        }
        mock_get.return_value = fake_resp

        result = run_web_search_tool({"query": "test"})
        self.assertNotIn("published_date", result["results"][0])


# ── Search deduplication tests ───────────────────────────────────────────


class TestSearchDeduplication(unittest.TestCase):
    def test_deduplicates_by_url(self):
        from tools import _deduplicate_results

        results = [
            {"url": "https://example.com", "title": "First", "engines": ["google"]},
            {"url": "https://example.com", "title": "Duplicate", "engines": ["bing"]},
            {"url": "https://other.com", "title": "Other", "engines": ["google"]},
        ]
        deduped = _deduplicate_results(results)
        self.assertEqual(len(deduped), 2)

    def test_merges_engines_on_dedup(self):
        from tools import _deduplicate_results

        results = [
            {"url": "https://example.com", "title": "Page", "engines": ["google"]},
            {"url": "https://example.com", "title": "Page", "engines": ["bing", "duckduckgo"]},
        ]
        deduped = _deduplicate_results(results)
        self.assertEqual(len(deduped), 1)
        engines = deduped[0]["engines"]
        self.assertIn("google", engines)
        self.assertIn("bing", engines)
        self.assertIn("duckduckgo", engines)

    def test_skips_results_without_url(self):
        from tools import _deduplicate_results

        results = [
            {"url": "", "title": "No URL"},
            {"title": "Missing URL key"},
            {"url": "https://valid.com", "title": "Valid"},
        ]
        deduped = _deduplicate_results(results)
        self.assertEqual(len(deduped), 1)
        self.assertEqual(deduped[0]["url"], "https://valid.com")

    @patch("tools.requests.get")
    def test_retries_on_failure(self, mock_get):
        """SearXNG should retry on transient failures."""
        from tools import _fetch_searxng

        fake_resp = MagicMock()
        fake_resp.raise_for_status.return_value = None
        fake_resp.json.return_value = {
            "results": [{"title": "OK", "url": "https://ok.com", "content": ""}]
        }
        mock_get.side_effect = [ConnectionError("fail"), fake_resp]

        results = _fetch_searxng("test query")
        self.assertEqual(len(results), 1)
        self.assertEqual(mock_get.call_count, 2)


# ── Web fetch tool tests ────────────────────────────────────────────────


class TestWebFetchTool(unittest.TestCase):
    def setUp(self):
        # Clear cache between tests
        import tools
        tools._url_cache.clear()

    def test_missing_url_returns_error(self):
        from tools import run_web_fetch_tool

        result = run_web_fetch_tool({})
        self.assertIn("error", result)

    def test_empty_url_returns_error(self):
        from tools import run_web_fetch_tool

        result = run_web_fetch_tool({"url": ""})
        self.assertIn("error", result)

    def test_rejects_non_http_scheme(self):
        from tools import run_web_fetch_tool

        result = run_web_fetch_tool({"url": "file:///etc/passwd"})
        self.assertIn("error", result)
        self.assertIn("http", result["error"])

    def test_rejects_ftp_scheme(self):
        from tools import run_web_fetch_tool

        result = run_web_fetch_tool({"url": "ftp://example.com/file"})
        self.assertIn("error", result)

    @patch("tools.requests.get")
    def test_fetches_html_page(self, mock_get):
        from tools import run_web_fetch_tool

        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.url = "https://example.com/page"
        fake_resp.headers = {"content-type": "text/html; charset=utf-8"}
        fake_resp.encoding = "utf-8"
        fake_resp.content = b"<html><head><title>Test</title></head><body><p>Hello world</p></body></html>"
        fake_resp.raise_for_status.return_value = None
        mock_get.return_value = fake_resp

        result = run_web_fetch_tool({"url": "https://example.com/page"})
        self.assertEqual(result["status_code"], 200)
        self.assertEqual(result["url"], "https://example.com/page")
        self.assertIn("Hello", result["text"])
        self.assertIsInstance(result["text_length"], int)
        self.assertFalse(result["bytes_truncated"])

    @patch("tools.requests.get")
    def test_pretty_prints_json(self, mock_get):
        from tools import run_web_fetch_tool

        json_body = json.dumps({"key": "value", "nested": {"a": 1}}).encode()
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.url = "https://api.example.com/data"
        fake_resp.headers = {"content-type": "application/json"}
        fake_resp.encoding = "utf-8"
        fake_resp.content = json_body
        fake_resp.raise_for_status.return_value = None
        mock_get.return_value = fake_resp

        result = run_web_fetch_tool({"url": "https://api.example.com/data"})
        self.assertIn('"key": "value"', result["text"])
        # Should be indented (pretty-printed)
        self.assertIn("\n", result["text"])

    @patch("tools.requests.get")
    def test_respects_max_chars(self, mock_get):
        from tools import run_web_fetch_tool

        long_text = "x" * 5000
        html = f"<html><body><p>{long_text}</p></body></html>".encode()
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.url = "https://example.com"
        fake_resp.headers = {"content-type": "text/html"}
        fake_resp.encoding = "utf-8"
        fake_resp.content = html
        fake_resp.raise_for_status.return_value = None
        mock_get.return_value = fake_resp

        result = run_web_fetch_tool({"url": "https://example.com", "max_chars": 500})
        self.assertLessEqual(len(result["text"]), 500)
        self.assertTrue(result["text_truncated"])

    @patch("tools.requests.get")
    def test_handles_http_error(self, mock_get):
        from tools import run_web_fetch_tool

        mock_get.side_effect = Exception("Connection refused")

        result = run_web_fetch_tool({"url": "https://down.example.com"})
        self.assertIn("error", result)
        self.assertEqual(result["url"], "https://down.example.com")

    @patch("tools.requests.get")
    def test_excludes_links_when_disabled(self, mock_get):
        from tools import run_web_fetch_tool

        html = b'<html><body><a href="https://other.com">link</a><p>text</p></body></html>'
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.url = "https://example.com"
        fake_resp.headers = {"content-type": "text/html"}
        fake_resp.encoding = "utf-8"
        fake_resp.content = html
        fake_resp.raise_for_status.return_value = None
        mock_get.return_value = fake_resp

        result = run_web_fetch_tool({"url": "https://example.com", "include_links": False})
        self.assertNotIn("links", result)

    @patch("tools.requests.get")
    def test_caches_results(self, mock_get):
        from tools import run_web_fetch_tool

        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.url = "https://example.com/cached"
        fake_resp.headers = {"content-type": "text/html"}
        fake_resp.encoding = "utf-8"
        fake_resp.content = b"<html><body>Cached content</body></html>"
        fake_resp.raise_for_status.return_value = None
        mock_get.return_value = fake_resp

        # First call – should hit the network
        result1 = run_web_fetch_tool({"url": "https://example.com/cached"})
        self.assertEqual(mock_get.call_count, 1)

        # Second call – should come from cache
        result2 = run_web_fetch_tool({"url": "https://example.com/cached"})
        self.assertEqual(mock_get.call_count, 1)  # Still 1 – no new HTTP call
        self.assertEqual(result1["text"], result2["text"])

    @patch("tools.requests.get")
    def test_handles_non_textual_content(self, mock_get):
        from tools import run_web_fetch_tool

        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.url = "https://example.com/image.png"
        fake_resp.headers = {"content-type": "image/png"}
        fake_resp.encoding = None
        fake_resp.content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        fake_resp.raise_for_status.return_value = None
        mock_get.return_value = fake_resp

        result = run_web_fetch_tool({"url": "https://example.com/image.png"})
        # Non-textual content should result in empty text
        self.assertEqual(result["text"], "")

    @patch("tools.requests.get")
    def test_clamps_max_chars_to_bounds(self, mock_get):
        """max_chars values outside bounds should be clamped."""
        from tools import run_web_fetch_tool

        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.url = "https://example.com"
        fake_resp.headers = {"content-type": "text/html"}
        fake_resp.encoding = "utf-8"
        fake_resp.content = b"<html><body>short</body></html>"
        fake_resp.raise_for_status.return_value = None
        mock_get.return_value = fake_resp

        # Excessively large value should be clamped to DEFAULT_MAX_CHARS
        result = run_web_fetch_tool({"url": "https://example.com", "max_chars": 999999})
        self.assertIsNotNone(result.get("text"))

        # Excessively small value should be clamped to 500
        import tools
        tools._url_cache.clear()
        result2 = run_web_fetch_tool({"url": "https://example.com", "max_chars": 1})
        self.assertIsNotNone(result2.get("text"))


# ── Web render tool tests ────────────────────────────────────────────────


class TestWebRenderTool(unittest.TestCase):
    def setUp(self):
        import tools
        tools._url_cache.clear()

    def test_missing_url_returns_error(self):
        from tools import run_web_render_tool

        result = run_web_render_tool({})
        self.assertIn("error", result)

    def test_rejects_non_http_scheme(self):
        from tools import run_web_render_tool

        result = run_web_render_tool({"url": "file:///etc/passwd"})
        self.assertIn("error", result)

    def test_returns_rendered_content(self):
        """Test render tool with mocked Playwright."""
        from tools import run_web_render_tool

        rendered_html = "<html><head><title>Rendered</title></head><body><p>JS content here</p></body></html>"

        mock_response = MagicMock()
        mock_response.status = 200

        mock_page = MagicMock()
        mock_page.content.return_value = rendered_html
        mock_page.url = "https://spa.example.com"
        mock_page.goto.return_value = mock_response

        mock_browser = MagicMock()
        mock_browser.new_page.return_value = mock_page

        mock_pw_instance = MagicMock()
        mock_pw_instance.chromium.launch.return_value = mock_browser

        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_pw_instance)
        mock_cm.__exit__ = MagicMock(return_value=False)

        with patch("playwright.sync_api.sync_playwright", return_value=mock_cm):
            result = run_web_render_tool({"url": "https://spa.example.com"})

        self.assertEqual(result["url"], "https://spa.example.com")
        self.assertEqual(result["status_code"], 200)
        self.assertIn("JS content", result["text"])

    def test_handles_browser_crash(self):
        from tools import run_web_render_tool

        mock_page = MagicMock()
        mock_page.goto.side_effect = Exception("Browser crashed")

        mock_browser = MagicMock()
        mock_browser.new_page.return_value = mock_page

        mock_pw_instance = MagicMock()
        mock_pw_instance.chromium.launch.return_value = mock_browser

        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_pw_instance)
        mock_cm.__exit__ = MagicMock(return_value=False)

        with patch("playwright.sync_api.sync_playwright", return_value=mock_cm):
            result = run_web_render_tool({"url": "https://crash.example.com"})

        self.assertIn("error", result)
        self.assertIn("Browser render failed", result["error"])

    def test_clamps_wait_seconds(self):
        """wait_seconds should be clamped to 0-10."""
        from tools import run_web_render_tool

        # Negative values → 0
        # Values > 10 → 10
        # This just verifies no crash; actual playwright is not called due to validation
        result = run_web_render_tool({"url": "ftp://bad"})
        self.assertIn("error", result)


# ── Cache tests ──────────────────────────────────────────────────────────


class TestUrlCache(unittest.TestCase):
    def setUp(self):
        import tools
        tools._url_cache.clear()

    def test_cache_set_and_get(self):
        from tools import _set_cached, _get_cached

        _set_cached("test_key", {"data": "value"})
        result = _get_cached("test_key")
        self.assertIsNotNone(result)
        self.assertEqual(result["data"], "value")

    def test_cache_miss(self):
        from tools import _get_cached

        self.assertIsNone(_get_cached("nonexistent"))

    def test_cache_expiry(self):
        from tools import _set_cached, _get_cached, _url_cache

        _set_cached("expiring", {"data": "old"})
        # Manually backdate the timestamp
        ts, result = _url_cache["expiring"]
        _url_cache["expiring"] = (ts - 600, result)  # 10 min ago

        self.assertIsNone(_get_cached("expiring"))


# ── URL validation tests ────────────────────────────────────────────────


class TestUrlValidation(unittest.TestCase):
    def test_valid_http(self):
        from tools import _validate_url

        self.assertIsNone(_validate_url("http://example.com"))

    def test_valid_https(self):
        from tools import _validate_url

        self.assertIsNone(_validate_url("https://example.com/path?q=1"))

    def test_empty_url(self):
        from tools import _validate_url

        self.assertIsNotNone(_validate_url(""))

    def test_file_scheme(self):
        from tools import _validate_url

        self.assertIsNotNone(_validate_url("file:///etc/passwd"))

    def test_ftp_scheme(self):
        from tools import _validate_url

        self.assertIsNotNone(_validate_url("ftp://example.com"))

    def test_no_scheme(self):
        from tools import _validate_url

        self.assertIsNotNone(_validate_url("example.com"))


# ── Content-type detection tests ─────────────────────────────────────────


class TestContentTypeDetection(unittest.TestCase):
    def test_html_content_type(self):
        from tools import _detect_content_type

        is_html, is_textual = _detect_content_type("text/html; charset=utf-8", "")
        self.assertTrue(is_html)
        self.assertTrue(is_textual)

    def test_json_content_type(self):
        from tools import _detect_content_type

        is_html, is_textual = _detect_content_type("application/json", "")
        self.assertFalse(is_html)
        self.assertTrue(is_textual)

    def test_image_content_type(self):
        from tools import _detect_content_type

        is_html, is_textual = _detect_content_type("image/png", "")
        self.assertFalse(is_html)
        self.assertFalse(is_textual)

    def test_html_sniffing(self):
        from tools import _detect_content_type

        is_html, is_textual = _detect_content_type(
            "application/octet-stream", "<html><body>hi</body></html>"
        )
        self.assertTrue(is_html)
        self.assertTrue(is_textual)

    def test_plain_text(self):
        from tools import _detect_content_type

        is_html, is_textual = _detect_content_type("text/plain", "just text")
        self.assertFalse(is_html)
        self.assertTrue(is_textual)


# ── JSON formatting tests ───────────────────────────────────────────────


class TestJsonFormatting(unittest.TestCase):
    def test_formats_valid_json(self):
        from tools import _format_json_if_applicable

        result = _format_json_if_applicable(
            "application/json", '{"key":"value"}'
        )
        self.assertIsNotNone(result)
        self.assertIn('"key": "value"', result)

    def test_returns_none_for_html(self):
        from tools import _format_json_if_applicable

        result = _format_json_if_applicable("text/html", "<html></html>")
        self.assertIsNone(result)

    def test_returns_none_for_invalid_json(self):
        from tools import _format_json_if_applicable

        result = _format_json_if_applicable("application/json", "not json")
        self.assertIsNone(result)


# ── HTML helper tests ────────────────────────────────────────────────────


class TestHtmlHelpers(unittest.TestCase):
    def test_simple_html_to_text(self):
        from tools import simple_html_to_text

        html = "<html><body><p>Hello <b>world</b></p></body></html>"
        result = simple_html_to_text(html)
        self.assertIn("Hello", result)
        self.assertIn("world", result)
        self.assertNotIn("<p>", result)

    def test_simple_html_strips_scripts(self):
        from tools import simple_html_to_text

        html = "<p>Before</p><script>alert('xss')</script><p>After</p>"
        result = simple_html_to_text(html)
        self.assertIn("Before", result)
        self.assertIn("After", result)
        self.assertNotIn("alert", result)

    def test_extract_title_from_html(self):
        from tools import extract_title_from_html

        html = "<html><head><title>My Page Title</title></head></html>"
        self.assertEqual(extract_title_from_html(html), "My Page Title")

    def test_extract_title_missing(self):
        from tools import extract_title_from_html

        html = "<html><head></head><body></body></html>"
        self.assertEqual(extract_title_from_html(html), "")

    def test_extract_links_from_html(self):
        from tools import extract_links_from_html

        html = '<a href="/page">link</a><a href="https://other.com">ext</a>'
        links = extract_links_from_html(html, "https://example.com")
        self.assertIn("https://example.com/page", links)
        self.assertIn("https://other.com", links)

    def test_extract_links_skips_non_http(self):
        from tools import extract_links_from_html

        html = '<a href="javascript:void(0)">js</a><a href="mailto:a@b.com">mail</a>'
        links = extract_links_from_html(html, "https://example.com")
        self.assertEqual(links, [])

    def test_truncate_text(self):
        from tools import truncate_text

        text, was_truncated = truncate_text("hello", 10)
        self.assertEqual(text, "hello")
        self.assertFalse(was_truncated)

        text, was_truncated = truncate_text("hello world", 5)
        self.assertEqual(text, "hello")
        self.assertTrue(was_truncated)


# ── ai_answer integration tests ─────────────────────────────────────────


class TestAiAnswer(unittest.TestCase):
    @patch("llm.client.chat.completions.create")
    def test_simple_response(self, mock_create):
        from llm import ai_answer

        fake_message = SimpleNamespace(
            content="Test reply", tool_calls=None, reasoning=None
        )
        fake_choice = SimpleNamespace(message=fake_message, finish_reason="stop")
        mock_create.return_value = SimpleNamespace(choices=[fake_choice])

        reply = ai_answer(FakeComment("u/grok What is the weather like?"))
        self.assertEqual(reply, "Test reply")

    @patch("llm.run_web_search_tool")
    @patch("llm.client.chat.completions.create")
    def test_tool_call_loop(self, mock_create, mock_search):
        from llm import ai_answer

        tool_call = SimpleNamespace(
            id="tc_1",
            function=SimpleNamespace(
                name="web_search",
                arguments=json.dumps({"query": "test query"}),
            ),
        )
        first_message = SimpleNamespace(
            content="", tool_calls=[tool_call], reasoning=None
        )
        second_message = SimpleNamespace(
            content="Final answer", tool_calls=None, reasoning=None
        )
        mock_create.side_effect = [
            SimpleNamespace(
                choices=[SimpleNamespace(message=first_message, finish_reason="tool_calls")]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(message=second_message, finish_reason="stop")]
            ),
        ]
        mock_search.return_value = {
            "query": "test query",
            "result_count": 1,
            "results": [{"title": "x", "url": "https://x", "snippet": "y", "engines": []}],
        }

        answer = ai_answer(FakeComment("u/grok What is happening?"))
        self.assertEqual(answer, "Final answer")
        self.assertEqual(mock_create.call_count, 2)
        mock_search.assert_called_once_with({"query": "test query"})

    @patch("llm.run_web_fetch_tool")
    @patch("llm.client.chat.completions.create")
    def test_web_fetch_tool_dispatch(self, mock_create, mock_fetch):
        """Verify the LLM can invoke web_fetch and get results back."""
        from llm import ai_answer

        tool_call = SimpleNamespace(
            id="tc_fetch",
            function=SimpleNamespace(
                name="web_fetch",
                arguments=json.dumps({"url": "https://example.com"}),
            ),
        )
        first_message = SimpleNamespace(content="", tool_calls=[tool_call], reasoning=None)
        second_message = SimpleNamespace(content="Read the page", tool_calls=None, reasoning=None)

        mock_create.side_effect = [
            SimpleNamespace(
                choices=[SimpleNamespace(message=first_message, finish_reason="tool_calls")]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(message=second_message, finish_reason="stop")]
            ),
        ]
        mock_fetch.return_value = {
            "url": "https://example.com",
            "status_code": 200,
            "title": "Example",
            "text": "Page content",
            "text_length": 12,
        }

        answer = ai_answer(FakeComment("u/grok read this page"))
        self.assertEqual(answer, "Read the page")
        mock_fetch.assert_called_once_with({"url": "https://example.com"})

    @patch("llm.run_web_render_tool")
    @patch("llm.client.chat.completions.create")
    def test_web_render_tool_dispatch(self, mock_create, mock_render):
        """Verify the LLM can invoke web_render and get results back."""
        from llm import ai_answer

        tool_call = SimpleNamespace(
            id="tc_render",
            function=SimpleNamespace(
                name="web_render",
                arguments=json.dumps({"url": "https://spa.example.com"}),
            ),
        )
        first_message = SimpleNamespace(content="", tool_calls=[tool_call], reasoning=None)
        second_message = SimpleNamespace(content="Rendered result", tool_calls=None, reasoning=None)

        mock_create.side_effect = [
            SimpleNamespace(
                choices=[SimpleNamespace(message=first_message, finish_reason="tool_calls")]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(message=second_message, finish_reason="stop")]
            ),
        ]
        mock_render.return_value = {
            "url": "https://spa.example.com",
            "status_code": 200,
            "title": "SPA",
            "text": "Rendered content",
            "text_length": 16,
        }

        answer = ai_answer(FakeComment("u/grok render this SPA"))
        self.assertEqual(answer, "Rendered result")
        mock_render.assert_called_once_with({"url": "https://spa.example.com"})

    @patch("llm.run_web_search_tool")
    @patch("llm.client.chat.completions.create")
    def test_fallback_after_max_tool_steps(self, mock_create, mock_search):
        from llm import ai_answer
        import config

        tool_call = SimpleNamespace(
            id="tc_loop",
            function=SimpleNamespace(
                name="web_search",
                arguments=json.dumps({"query": "looping"}),
            ),
        )
        tool_msg = SimpleNamespace(content="", tool_calls=[tool_call], reasoning=None)
        tool_resp = SimpleNamespace(
            choices=[SimpleNamespace(message=tool_msg, finish_reason="tool_calls")]
        )
        final_msg = SimpleNamespace(
            content="Best-effort answer", tool_calls=None, reasoning=None
        )
        final_resp = SimpleNamespace(
            choices=[SimpleNamespace(message=final_msg, finish_reason="stop")]
        )

        mock_create.side_effect = [tool_resp] * config.MAX_TOOL_STEPS + [final_resp]
        mock_search.return_value = {"query": "looping", "result_count": 0, "error": "timeout"}

        answer = ai_answer(FakeComment("u/grok news?"))
        self.assertEqual(answer, "Best-effort answer")
        self.assertEqual(mock_create.call_count, config.MAX_TOOL_STEPS + 1)

    @patch("llm.client.chat.completions.create")
    def test_empty_response_returns_fallback_text(self, mock_create):
        from llm import ai_answer

        fake_message = SimpleNamespace(content="", tool_calls=None, reasoning=None)
        mock_create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=fake_message, finish_reason="stop")]
        )

        answer = ai_answer(FakeComment("u/grok hi"))
        self.assertIn("sorry", answer.lower())

    @patch("llm.client.chat.completions.create")
    def test_timeout_is_passed(self, mock_create):
        from llm import ai_answer

        fake_message = SimpleNamespace(content="reply", tool_calls=None, reasoning=None)
        mock_create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=fake_message, finish_reason="stop")]
        )

        ai_answer(FakeComment("u/grok test"))
        call_kwargs = mock_create.call_args.kwargs
        self.assertIn("timeout", call_kwargs)
        self.assertGreater(call_kwargs["timeout"], 0)

    @patch("llm.client.chat.completions.create")
    def test_tool_definitions_include_all_three_tools(self, mock_create):
        from llm import ai_answer

        fake_message = SimpleNamespace(content="reply", tool_calls=None, reasoning=None)
        mock_create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=fake_message, finish_reason="stop")]
        )

        ai_answer(FakeComment("u/grok test"))
        call_kwargs = mock_create.call_args.kwargs
        tool_names = [t["function"]["name"] for t in call_kwargs["tools"]]
        self.assertIn("web_search", tool_names)
        self.assertIn("web_fetch", tool_names)
        self.assertIn("web_render", tool_names)
        self.assertEqual(len(tool_names), 3)

    @patch("llm.client.chat.completions.create")
    def test_unknown_tool_returns_error(self, mock_create):
        from llm import _execute_tool

        result = _execute_tool("nonexistent_tool", {})
        self.assertIn("error", result)
        self.assertIn("Unknown tool", result["error"])


# ── Main loop helper tests ──────────────────────────────────────────────


class TestMainHelpers(unittest.TestCase):
    def test_reply_with_retry_succeeds_first_try(self):
        from main import _reply_with_retry

        comment = FakeComment("test")
        comment.reply = MagicMock()
        _reply_with_retry(comment, "hello")
        comment.reply.assert_called_once_with("hello")

    def test_reply_with_retry_retries_on_failure(self):
        from main import _reply_with_retry

        comment = FakeComment("test")
        comment.reply = MagicMock(side_effect=[Exception("fail"), None])
        _reply_with_retry(comment, "hello", retries=2)
        self.assertEqual(comment.reply.call_count, 2)

    def test_reply_with_retry_raises_after_exhaustion(self):
        from main import _reply_with_retry

        comment = FakeComment("test")
        comment.reply = MagicMock(side_effect=Exception("permanent failure"))
        with self.assertRaises(Exception):
            _reply_with_retry(comment, "hello", retries=2)


# ── Tool result summary tests ────────────────────────────────────────────


class TestSummarizeToolResult(unittest.TestCase):
    def test_web_search_summary(self):
        from tools import summarize_tool_result

        result = {"result_count": 3, "query": "hello"}
        summary = summarize_tool_result("web_search", result)
        self.assertIn("result_count=3", summary)
        self.assertIn("hello", summary)

    def test_web_fetch_summary(self):
        from tools import summarize_tool_result

        result = {"status_code": 200, "text_length": 5000, "title": "Page", "error": None}
        summary = summarize_tool_result("web_fetch", result)
        self.assertIn("status=200", summary)
        self.assertIn("text_length=5000", summary)

    def test_web_render_summary(self):
        from tools import summarize_tool_result

        result = {"status_code": 200, "text_length": 3000, "title": "SPA", "error": None}
        summary = summarize_tool_result("web_render", result)
        self.assertIn("status=200", summary)

    def test_unknown_tool_summary(self):
        from tools import summarize_tool_result

        result = {"some": "data"}
        summary = summarize_tool_result("mystery_tool", result)
        self.assertIn("some", summary)


if __name__ == "__main__":
    unittest.main()
