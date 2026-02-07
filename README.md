# HelperBot

HelperBot is a Python-based Reddit bot that listens for specific words in comments (like `@AI`). When activated, it reads the conversation thread, gives the model access to optional web-search and URL-reading tools backed by your own infrastructure, sends everything to an AI model via OpenRouter, and posts the AI's response back to Reddit.

## How to use
Summon bot on Reddit by starting your message with @grok

## Features

- Listens for comments starting with `@ai`, `@chatgpt`, `@gemini`, `@gpt`, or `@grok` (case-insensitive).
- Fetches the entire conversation context (submission + ancestor comments), including images.
- Uses OpenRouter to connect to various Large Language Models (LLMs).
- Exposes a model tool (`web_search`) backed by your own SearXNG instance (instead of OpenRouter's paid web plugin).
- Exposes a model tool (`web_open_url`) to fetch and extract readable content from URLs, with optional rendered fallback.
- Posts the LLM's generated reply back to the comment.
- Configurable settings for trigger words, target subreddits, and rate limiting.
- Structured logging via Python's `logging` module.
- Graceful shutdown on SIGINT/SIGTERM.
- Automatic retry with backoff for Reddit API failures.

## Requirements

- Python 3.9+
- `pip` (Python package installer)

## Project Structure

```
reddit-helperbot/
├── main.py              # Entry point: main loop, signal handling, stats
├── config.py            # Configuration, env validation, client init
├── llm.py               # LLM interaction and tool-calling loop
├── tools.py             # Web search (SearXNG) and URL fetching tools
├── transcript.py        # Reddit thread transcript and image extraction
├── test_helperbot.py    # Unit tests (51 tests)
├── requirements.txt     # Pinned dependencies
├── Dockerfile           # Container setup (Python 3.12)
├── .env.example         # Template for secrets
└── .gitignore
```

## Getting Started

Follow these steps to get HelperBot up and running:

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/petercunha/reddit-helperbot.git
    cd reddit-helperbot
    ```

2.  **Set up a Virtual Environment (Recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    For rendered web fetching (optional):
    ```bash
    playwright install chromium
    ```

4.  **Create `.env` File:**
    Create a file named `.env` in the project root. **Do not share this file.**

    Copy and paste the following template into your `.env` file and fill in your actual credentials:

    ```dotenv
    # .env file for HelperBot

    # 1. OpenRouter API Key (https://openrouter.ai/keys)
    OPENROUTER_API_KEY="sk-or-v1-..."

    # 2. SearXNG (self-hosted) for model tool calls
    SEARXNG_BASE_URL="https://seedbox.local/searxng"

    # 3. Reddit API Credentials (Create an app: https://www.reddit.com/prefs/apps)
    #    Select "script" type. Redirect URI can be http://localhost:8080
    REDDIT_CLIENT_ID="YOUR_REDDIT_CLIENT_ID"
    REDDIT_CLIENT_SECRET="YOUR_REDDIT_CLIENT_SECRET"

    # 4. Reddit Account Credentials (The bot's Reddit account)
    REDDIT_USERNAME="YOUR_BOTS_REDDIT_USERNAME"
    REDDIT_PASSWORD="YOUR_BOTS_REDDIT_PASSWORD"

    # 5. User Agent (A descriptive name for your bot instance)
    USER_AGENT="HelperBot v1.0 by u/YOUR_USERNAME"
    ```

5.  **Run the Bot:**
    ```bash
    python main.py
    ```
    The bot will validate that all required environment variables are set, then start listening for comments.

## Running with Docker

```bash
docker build -t helperbot .
docker run -d --restart=always --name helperbot --env-file .env helperbot
```

## Configuration

You can customize the bot's behavior by editing `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `MODEL` | `moonshotai/kimi-k2.5` | OpenRouter model for generating replies |
| `TRIGGER` | See regex | Regular expression for trigger words/patterns |
| `SUBS` | `["all"]` | Subreddits to listen to (e.g., `["askreddit", "python"]`) |
| `REDDIT_RATE_LIMIT_SEC` | `10` | Delay (seconds) after posting a reply |
| `MAX_CHARS` | `40000` | Maximum context length sent to the AI model |
| `MAX_TOOL_STEPS` | `16` | Maximum LLM tool-calling iterations |
| `MAX_IMAGES_TO_SEND` | `5` | Maximum images included in the prompt |
| `OPENROUTER_TIMEOUT` | `120` | API call timeout in seconds |

## Testing

Run the test suite:

```bash
python -m unittest test_helperbot -v
```

The test suite covers config validation, trigger regex, image extraction, transcript building, LLM message helpers, web search tools, the tool-calling loop, retry logic, and HTML parsing.

## SearXNG Categories (Examples)

When the model calls `web_search`, it can optionally pass a `categories` list.

Example values: `general`, `images`, `videos`, `news`, `map`, `music`, `it`, `science`, `files`, `social media`

```json
{
  "query": "latest spacex launch update",
  "categories": ["news", "science"],
  "time_range": "day"
}
```

Note: the exact available categories depend on your SearXNG instance configuration.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Missing required environment variables` at startup | Copy `.env.example` to `.env` and fill in all values |
| Reddit rate-limit errors (HTTP 429) | Increase `REDDIT_RATE_LIMIT_SEC` in `config.py` |
| SearXNG search failures | Check that `SEARXNG_BASE_URL` is reachable from the bot |
| OpenRouter timeout errors | Increase `OPENROUTER_TIMEOUT` or try a faster model |
| Bot account suspended | Reddit may suspend bot accounts for excessive posting; reduce `SUBS` scope |
| `playwright_not_installed` errors | Run `playwright install chromium` |
