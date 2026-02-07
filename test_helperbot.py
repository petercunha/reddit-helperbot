import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
import json
import main

# --- Fake objects for testing ---

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
    def __init__(self, body):
        self.body = body
        self.id = "test_id"
        self.submission = FakeSubmission()
        self.subreddit = FakeSubreddit()
        self.author = FakeAuthor()

    def parent(self):
        # For testing, assume this comment is the root.
        self._is_root = True
        return self

    @property
    def is_root(self):
        # For testing, simple implementation: if parent returns self, treat as root.
        return True

# --- Unit tests ---

class TestHelperBot(unittest.TestCase):
    def test_trigger_regex_matches_valid(self):
        valid_strings = [
            "u/grok test",
            "@grok test",
            "   u/grok test",
            "[u/grok test"
        ]
        for s in valid_strings:
            self.assertIsNotNone(
                main.TRIGGER.match(s),
                f"Regex should match valid trigger string: {s}"
            )

    def test_trigger_regex_does_not_match_invalid(self):
        invalid_strings = [
            "groktest",
            "something else",
            "hello u/grok"  # trigger must be at beginning
        ]
        for s in invalid_strings:
            self.assertIsNone(
                main.TRIGGER.match(s),
                f"Regex should not match invalid trigger string: {s}"
            )

    @patch('main.client.chat.completions.create')
    def test_ai_answer(self, mock_create):
        # Set up a fake response from OpenRouter API
        fake_message = SimpleNamespace(content="Test reply", tool_calls=None, reasoning=None)
        fake_choice = SimpleNamespace(message=fake_message, finish_reason="stop")
        fake_response = SimpleNamespace(choices=[fake_choice])
        mock_create.return_value = fake_response

        fake_comment = FakeComment("u/grok What is the weather like?")
        reply = main.ai_answer(fake_comment)
        self.assertEqual(reply, "Test reply")

    def test_reddit_client_configuration(self):
        # Check that the Reddit client's configuration contains the expected user agent.
        agent = main.reddit.config.user_agent
        self.assertIn("helperbot", agent.lower())

    def test_web_open_url_rejects_non_http_scheme(self):
        result = main.run_web_open_url_tool({"url": "file:///etc/passwd"})
        self.assertIn("error", result)

    @patch("main.requests.get")
    def test_web_open_url_fetch_extracts_text(self, mock_get):
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.url = "https://example.com/page"
        fake_resp.headers = {"content-type": "text/html; charset=utf-8"}
        fake_resp.encoding = "utf-8"
        fake_resp.content = b"<html><head><title>Example</title></head><body><article><p>Hello world</p></article></body></html>"
        fake_resp.raise_for_status.return_value = None
        mock_get.return_value = fake_resp

        result = main.run_web_open_url_tool(
            {"url": "https://example.com/page", "mode": "fetch", "max_chars": 500}
        )
        self.assertEqual(result.get("mode_used"), "fetch")
        self.assertEqual(result.get("title"), "Example")
        self.assertIn("Hello", result.get("text", ""))

    @patch("main.requests.get")
    def test_web_search_tool_returns_results(self, mock_get):
        fake_resp = MagicMock()
        fake_resp.raise_for_status.return_value = None
        fake_resp.json.return_value = {
            "results": [
                {
                    "title": "OpenRouter Tool Calling",
                    "url": "https://openrouter.ai/docs/features/tool-calling",
                    "content": "How tool calling works",
                    "engines": ["duckduckgo"],
                }
            ]
        }
        mock_get.return_value = fake_resp

        result = main.run_web_search_tool({"query": "openrouter tool calling"})
        self.assertEqual(result.get("result_count"), 1)
        self.assertEqual(result["results"][0]["title"], "OpenRouter Tool Calling")

    @patch("main.run_web_search_tool")
    @patch("main.client.chat.completions.create")
    def test_ai_answer_runs_tool_calls(self, mock_create, mock_run_web_search):
        tool_call = SimpleNamespace(
            id="tc_1",
            function=SimpleNamespace(
                name="web_search",
                arguments=json.dumps({"query": "test query"})
            ),
        )
        first_message = SimpleNamespace(content="", tool_calls=[tool_call], reasoning=None)
        second_message = SimpleNamespace(content="Final answer", tool_calls=None, reasoning=None)

        first_response = SimpleNamespace(
            choices=[SimpleNamespace(message=first_message, finish_reason="tool_calls")]
        )
        second_response = SimpleNamespace(
            choices=[SimpleNamespace(message=second_message, finish_reason="stop")]
        )
        mock_create.side_effect = [first_response, second_response]
        mock_run_web_search.return_value = {
            "query": "test query",
            "result_count": 1,
            "results": [{"title": "x", "url": "https://x", "snippet": "y", "engines": []}],
        }

        fake_comment = FakeComment("u/grok What is happening?")
        answer = main.ai_answer(fake_comment)

        self.assertEqual(answer, "Final answer")
        self.assertEqual(mock_create.call_count, 2)
        mock_run_web_search.assert_called_once_with({"query": "test query"})
        first_call_kwargs = mock_create.call_args_list[0].kwargs
        tool_names = [tool["function"]["name"] for tool in first_call_kwargs["tools"]]
        self.assertIn("web_search", tool_names)
        self.assertIn("web_open_url", tool_names)

    @patch("main.run_web_open_url_tool")
    @patch("main.client.chat.completions.create")
    def test_ai_answer_runs_web_open_url_tool_call(self, mock_create, mock_run_web_open):
        tool_call = SimpleNamespace(
            id="tc_2",
            function=SimpleNamespace(
                name="web_open_url",
                arguments=json.dumps({"url": "https://example.com", "mode": "fetch"})
            ),
        )
        first_message = SimpleNamespace(content="", tool_calls=[tool_call], reasoning=None)
        second_message = SimpleNamespace(content="Final answer from url", tool_calls=None, reasoning=None)

        first_response = SimpleNamespace(
            choices=[SimpleNamespace(message=first_message, finish_reason="tool_calls")]
        )
        second_response = SimpleNamespace(
            choices=[SimpleNamespace(message=second_message, finish_reason="stop")]
        )
        mock_create.side_effect = [first_response, second_response]
        mock_run_web_open.return_value = {
            "url": "https://example.com",
            "mode_used": "fetch",
            "status_code": 200,
            "title": "Example",
            "text": "hello",
            "text_length": 5,
            "text_truncated": False,
            "bytes_truncated": False,
            "links": [],
        }

        fake_comment = FakeComment("u/grok summarize this url")
        answer = main.ai_answer(fake_comment)

        self.assertEqual(answer, "Final answer from url")
        mock_run_web_open.assert_called_once_with({"url": "https://example.com", "mode": "fetch"})

if __name__ == '__main__':
    unittest.main()
