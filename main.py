#!/usr/bin/env python3
"""
grok_bot.py – Reddit reply‑bot that listens for comments starting with
“u/grok”, feeds the entire conversation thread to an LLM on OpenRouter,
and posts the model’s answer back to Reddit.

• Requires:  praw, python‑dotenv, openai
• Put a .env file (see template below) in the same directory.
"""

import os
import re
import time
import textwrap
from pathlib import Path

from dotenv import load_dotenv
import praw
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────
# 1. Load .env that sits next to this script
# ──────────────────────────────────────────────────────────────────────────
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

# ──────────────────────────────────────────────────────────────────────────
# 2. OpenRouter client (OpenAI‑compatible SDK)
# ──────────────────────────────────────────────────────────────────────────
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/mygithub/helperbot",
        "X-Title": "helperbot",
    },
)
MODEL = "deepseek/deepseek-r1:free"  # pick any model on OpenRouter

# ──────────────────────────────────────────────────────────────────────────
# 3. Reddit client
# ──────────────────────────────────────────────────────────────────────────
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    username=os.getenv("REDDIT_USERNAME"),
    password=os.getenv("REDDIT_PASSWORD"),
    user_agent=os.getenv("USER_AGENT"),
)

# ──────────────────────────────────────────────────────────────────────────
# 4. Bot settings
# ──────────────────────────────────────────────────────────────────────────
TRIGGER = re.compile(r"^\s*(?:u/|@)?grok\b", re.I)   # “u/grok”, “@grok”, or “grok” at start of comment
SUBS    = ["all"]                             # listen everywhere; tune as needed
REDDIT_RATE_LIMIT_SEC = 10                     # courtesy delay after replying

# context‑window guard
MAX_CHARS = 10_000    # rough safety cap for prompt length
INDENT    = "> "      # quote indent used in transcript


# ──────────────────────────────────────────────────────────────────────────
# 5. Build a transcript: submission + ancestor comments
# ──────────────────────────────────────────────────────────────────────────
def build_thread_transcript(trigger_comment: praw.models.Comment) -> str:
    """
    Return a single markdown‑flavoured string representing the entire
    conversation (submission + ancestor chain) that led to trigger_comment.
    """
    sub = trigger_comment.submission
    parts = [f"SUBMISSION TITLE: {sub.title.strip()}"]
    if sub.is_self and sub.selftext:
        parts.append(sub.selftext.strip())
    parts.append("\n---")   # divider

    # Collect ancestor comments (root → trigger)
    ancestors = []
    c = trigger_comment
    while isinstance(c, praw.models.Comment):
        ancestors.append(c)
        if c.is_root:
            break
        c = c.parent()
    ancestors.reverse()

    for cm in ancestors:
        author = cm.author.name if cm.author else "[deleted]"
        body   = cm.body.strip() or "[empty]"
        quoted = textwrap.indent(body, INDENT)
        parts.append(f"{author} wrote:\n{quoted}\n")

    transcript = "\n".join(parts)
    if len(transcript) > MAX_CHARS:           # trim if oversized
        transcript = transcript[-MAX_CHARS:]
    return transcript


# ──────────────────────────────────────────────────────────────────────────
# 6. Send prompt to LLM and get answer
# ──────────────────────────────────────────────────────────────────────────
def ai_answer(trigger_comment: praw.models.Comment) -> str:
    thread_text = build_thread_transcript(trigger_comment)
    user_question = TRIGGER.sub("", trigger_comment.body, 1).strip() or "(no explicit question)"

    prompt = f"""
You are a helpful Reddit assistant. Below is the full thread that led to the
user's trigger comment. Use it to craft an accurate, concise reply.

{thread_text}
--- END OF THREAD ---

USER QUESTION (last comment): {user_question}
""".strip()

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()


# ──────────────────────────────────────────────────────────────────────────
# 7. Main loop
# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("🟢 helperbot is live…")
    for comment in reddit.subreddit("+".join(SUBS)).stream.comments(skip_existing=True):
        try:
            # Ignore our own comments
            if comment.author == reddit.user.me():
                continue

            # Check trigger
            # if not TRIGGER.match(comment.body):
            #     continue

            print(f"↳ Trigger detected in r/{comment.subreddit.display_name} | {comment.id}")
            print("  ✔ Thread context:", build_thread_transcript(comment))
            reply_text = ai_answer(comment) + "\n\n---\n\n*This reply was generated by {MODEL} via OpenRouter*"
            # comment.reply(f"**OpenRouter bot says:**\n\n{reply_text}")
            print("  ✔ Generated reply:", reply_text)
            print("  ✔ Replied")

            time.sleep(REDDIT_RATE_LIMIT_SEC)   # be polite to Reddit
        except Exception as exc:
            print("  ⚠️  Error:", exc)
            time.sleep(10)                      # basic backoff


if __name__ == "__main__":
    main()
