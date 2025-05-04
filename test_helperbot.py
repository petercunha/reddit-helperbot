import unittest
from unittest.mock import patch, MagicMock
import main

# --- Fake objects for testing ---

class FakeSubreddit:
    display_name = "testsub"

class FakeSubmission:
    title = "Test Submission"
    selftext = "Test selftext"
    is_self = True

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
        fake_choice = MagicMock()
        fake_choice.message.content = "Test reply"
        fake_response = MagicMock()
        fake_response.choices = [fake_choice]
        mock_create.return_value = fake_response

        fake_comment = FakeComment("u/grok What is the weather like?")
        reply = main.ai_answer(fake_comment)
        self.assertEqual(reply, "Test reply")

    def test_reddit_client_configuration(self):
        # Check that the Reddit client's configuration contains the expected user agent.
        agent = main.reddit.config.user_agent
        self.assertIn("helperbot", agent.lower())

if __name__ == '__main__':
    unittest.main()
