# HelperBot

HelperBot is a Python-based Reddit bot that listens for specific words in comments (like `grok`, `gwok`, or `gork`). When activated, it reads the conversation thread, sends it to an AI model via OpenRouter, and posts the AI's response back to Reddit.

## ✨ Features

- Listens for comments starting with `u/grok`, `@grok`, `grok`, `gwok`, or `gork` (case-insensitive).
- Fetches the entire conversation context (submission + comments).
- Uses OpenRouter to connect to various Large Language Models (LLMs).
- Posts the LLM's generated reply back to the comment.
- Configurable settings for trigger words, target subreddits, and rate limiting.

## 📋 Requirements

- Python 3.x
- `pip` (Python package installer)

## 🚀 Getting Started

Follow these steps to get HelperBot up and running:

1.  **Clone the Repository (Optional):**
    If you haven't already, download or clone this repository to your local machine.

    ```bash
    git clone <repository_url>
    cd reddit-helperbot
    ```

2.  **Set up a Virtual Environment (Recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    Install the required Python libraries.

    ```bash
    pip install praw python-dotenv openai
    ```

4.  **Create `.env` File:**
    Create a file named `.env` in the same directory as `main.py`. This file will store your sensitive credentials. **Do not share this file.**

    Copy and paste the following template into your `.env` file and fill in your actual credentials:

    ```dotenv
    # .env file for HelperBot

    # 1. OpenRouter API Key (https://openrouter.ai/keys)
    OPENROUTER_API_KEY="sk-or-v1-..."

    # 2. Reddit API Credentials (Create an app: https://www.reddit.com/prefs/apps)
    #    Select "script" type. Redirect URI can be http://localhost:8080
    REDDIT_CLIENT_ID="YOUR_REDDIT_CLIENT_ID"        # Found under your app name
    REDDIT_CLIENT_SECRET="YOUR_REDDIT_CLIENT_SECRET"  # The "secret" field

    # 3. Reddit Account Credentials (The bot's Reddit account)
    REDDIT_USERNAME="YOUR_BOTS_REDDIT_USERNAME"
    REDDIT_PASSWORD="YOUR_BOTS_REDDIT_PASSWORD"

    # 4. User Agent (A descriptive name for your bot instance)
    #    Example: HelperBot v1.0 by u/YourUsername
    USER_AGENT="HelperBot v1.0 by u/YOUR_USERNAME"
    ```

5.  **Run the Bot:**
    Execute the main script from your terminal:
    ```bash
    python main.py
    ```
    You should see the message "🟢 helperbot is live…" indicating the bot is running and listening for comments.

## ⚙️ Configuration

You can customize the bot's behavior by editing the settings near the top of `main.py`:

- `MODEL`: Change the OpenRouter model used for generating replies.
- `TRIGGER`: Modify the regular expression to change the trigger words or patterns.
- `SUBS`: Specify which subreddits the bot should listen to (e.g., `["askreddit", "python"]`). `["all"]` listens everywhere.
- `REDDIT_RATE_LIMIT_SEC`: Adjust the delay (in seconds) after posting a reply.
- `MAX_CHARS`: Set a limit on the context length sent to the AI model.

---

Happy Botting! 🎉
