"""
test_helperbot.py – Unit tests for HelperBot.

Covers: config validation, trigger regex, image extraction, transcript
building, LLM message helpers, web search tool, and the ai_answer
tool-calling loop.

Note: web_open_url / web fetch tests are intentionally excluded as that
feature is expected to change.
"""

import json
import os
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
            # Should not raise
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
            "hello u/grok",  # trigger must be at the start
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
        # Check &amp; is decoded
        self.assertTrue(
            any("&amp;" not in url for url in images)
        )


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

        text = "short text"
        self.assertEqual(truncate_for_log(text), text)

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
        result = extract_reasoning_for_log(msg)
        self.assertEqual(result, "")


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
        fake_response = SimpleNamespace(choices=[fake_choice])
        mock_create.return_value = fake_response

        comment = FakeComment("u/grok What is the weather like?")
        reply = ai_answer(comment)
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
        fake_choice = SimpleNamespace(message=fake_message, finish_reason="stop")
        mock_create.return_value = SimpleNamespace(choices=[fake_choice])

        answer = ai_answer(FakeComment("u/grok hi"))
        self.assertIn("sorry", answer.lower())

    @patch("llm.client.chat.completions.create")
    def test_timeout_is_passed(self, mock_create):
        from llm import ai_answer

        fake_message = SimpleNamespace(
            content="reply", tool_calls=None, reasoning=None
        )
        mock_create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=fake_message, finish_reason="stop")]
        )

        ai_answer(FakeComment("u/grok test"))
        call_kwargs = mock_create.call_args.kwargs
        self.assertIn("timeout", call_kwargs)
        self.assertGreater(call_kwargs["timeout"], 0)


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

    def test_web_open_url_summary(self):
        from tools import summarize_tool_result

        result = {
            "mode_used": "fetch",
            "status_code": 200,
            "text_length": 5000,
            "error": None,
        }
        summary = summarize_tool_result("web_open_url", result)
        self.assertIn("mode=fetch", summary)
        self.assertIn("status=200", summary)


if __name__ == "__main__":
    unittest.main()
