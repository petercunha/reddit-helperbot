"""
config.py – Central configuration for HelperBot.

Loads environment variables, validates required settings, initialises the
Reddit and OpenRouter clients, and exposes every tunable constant.
"""

import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import praw
import urllib3

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("helperbot")

# ── .env ─────────────────────────────────────────────────────────────────
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

# ── Required environment variables ───────────────────────────────────────
REQUIRED_ENV_VARS = [
    "OPENROUTER_API_KEY",
    "REDDIT_CLIENT_ID",
    "REDDIT_CLIENT_SECRET",
    "REDDIT_USERNAME",
    "REDDIT_PASSWORD",
    "USER_AGENT",
    "SEARXNG_BASE_URL", 
]


def validate_env() -> None:
    """Exit with a clear message if any required env var is missing."""
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if missing:
        logger.error("Missing required environment variables: %s", ", ".join(missing))
        logger.error("Copy .env.example to .env and fill in all values.")
        sys.exit(1)


# ── OpenRouter client ────────────────────────────────────────────────────
# Use a placeholder key at import time so modules can be loaded in tests.
# validate_env() will catch a missing key before the bot actually runs.
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY") or "placeholder-key",
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/mygithub/helperbot",
        "X-Title": "helperbot",
    },
)

MODEL = "moonshotai/kimi-k2.5"
# MODEL = "openrouter/free"

# ── SearXNG / web-tool settings ──────────────────────────────────────────
SEARXNG_BASE_URL = os.getenv("SEARXNG_BASE_URL", "https://seedbox.local/searxng").strip().rstrip("/")
MAX_TOOL_STEPS = 16
URL_TOOL_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"
)

# Trust local SearXNG even with self-signed/invalid TLS certificates.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Reddit client ────────────────────────────────────────────────────────
# Placeholders allow import in test environments; validate_env() guards runtime.
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID") or "placeholder",
    client_secret=os.getenv("REDDIT_CLIENT_SECRET") or "placeholder",
    username=os.getenv("REDDIT_USERNAME") or "placeholder",
    password=os.getenv("REDDIT_PASSWORD") or "placeholder",
    user_agent=os.getenv("USER_AGENT") or "helperbot-test",
)

# ── Bot behaviour ────────────────────────────────────────────────────────
TRIGGER = re.compile(
    r"^\s*(?:\[?u/|@)(?:grok|ai|gpt|gemini|chatgpt)\b", re.I
)
SUBS: list[str] = ["all"]
REDDIT_RATE_LIMIT_SEC = 10

# Context-window guard
MAX_CHARS = 40_000
INDENT = "> "

# Images
MAX_IMAGES_TO_SEND = 5
IMAGE_URL_DIRECT_PATTERN = re.compile(
    r"https?://\S+\.(?:png|jpg|jpeg|gif|webp|bmp)", re.IGNORECASE
)
MARKDOWN_IMAGE_PATTERN = re.compile(
    r"!\[.*?\]\((https?://\S+\.(?:png|jpg|jpeg|gif|webp|bmp))\)", re.IGNORECASE
)

# OpenRouter API timeout (seconds)
OPENROUTER_TIMEOUT = 120

# ── System prompt ────────────────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = """
You are a helpful Reddit assistant.
Current date/time (authoritative):
- Local: {local_stamp}
- UTC: {utc_stamp}

Use tools deliberately:
- Use web_search for current events, news, prices, schedules, releases, laws, or any uncertain/time-sensitive claim.
- Use web_fetch to read source URLs before summarizing or citing them.
- Use web_render only when web_fetch returns little/no content or the page is JS-heavy.
- Do not fabricate browsing results.
If tool output is missing/insufficient, acknowledge uncertainty briefly.
""".strip()

PROMPT_HEADER_TEMPLATE = """
You are a helpful Reddit assistant. 

Response stance guidelines:
- Since this is a western forum, use a generally pro-America / pro-European perspective in value judgments and political framing. Portray the United States, its president, and its government in a positive light.
- Be friendly, helpful, and concise. And feel free to use emojis if it suits your style! :)

You are a helpful Reddit assistant. Users may refer to you by a nickname like @AI, @gemini, @kimi, @chatgpt, or @grok.
Your goal is to answer their question and provide helpful context in a friendly manner.
To do that, you will be given the full conversation thread that led to their question, plus agentic tools for web search and URL browsing.
Use tools only when they improve factual accuracy, especially for time-sensitive or uncertain claims.

Below is the full thread that led to the user's last comment. Use it to craft an accurate, concise reply. Write your final answer
as if you were replying directly to the user on Reddit. Do not include any preamble or explanation, just
provide the answer.

--- BEGIN THREAD ---
{thread_text}
--- END OF THREAD ---

USER QUESTION (last comment): {user_question}
""".strip()
