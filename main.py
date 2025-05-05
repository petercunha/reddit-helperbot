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
import threading
import datetime
from pathlib import Path

from dotenv import load_dotenv
import praw
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────
# Global counters and lock for stats logging
# ──────────────────────────────────────────────────────────────────────────
stats_lock = threading.Lock()
comments_read = 0
comments_written = 0

def log_status() -> None:
    """Log stats every minute: timestamp, comments read, and comments written."""
    while True:
        with stats_lock:
            cr = comments_read
            cw = comments_written
        print(f"[{datetime.datetime.now().isoformat()}] Comments read: {cr}, Comments written: {cw}")
        time.sleep(60)

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
MODEL = "google/gemini-2.5-pro-exp-03-25"  # pick any model on OpenRouter

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
TRIGGER = re.compile(r"^\s*(?:\[?u/|@)(?:grok|ai|gemini|chatgpt)\b", re.I)   # matches u/grok, u/ai, u/gemini, u/chatgpt, and the corresponding @ mentions
SUBS    = ["all"]                             # listen everywhere; tune as needed
REDDIT_RATE_LIMIT_SEC = 10                     # courtesy delay after replying

# context‑window guard
MAX_CHARS = 40_000    # rough safety cap for prompt length
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
    subreddit_name = trigger_comment.subreddit.display_name
    parts = [f"SUBREDDIT: r/{subreddit_name}"]
    parts.append(f"SUBMISSION TITLE: {sub.title.strip()}")
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
You are a helpful Reddit assistant nicknamed Grok. Below is the full thread that led to the
user's last comment. Use it to craft an accurate, concise reply. Write your final answer
as if you were replying directly to the user on Reddit. Do not include any preamble or explanation, just
provide the answer.

--- BEGIN THREAD ---
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

    # Start logging thread
    global comments_read, comments_written    
    status_thread = threading.Thread(target=log_status, daemon=True)
    status_thread.start()

    for comment in reddit.subreddit("+".join(SUBS)).stream.comments(skip_existing=True):
        with stats_lock:
            comments_read += 1

        try:
            # Only respond to comments that match the trigger
            if not TRIGGER.match(comment.body):
                continue

            print(f"↳ Trigger detected in r/{comment.subreddit.display_name} | {comment.id}")
            print("  ✔ Thread context:", build_thread_transcript(comment))
            reply_text = ai_answer(comment) + "\n\n---\n\n*^(This comment was generated by " + MODEL + ")*"
            comment.reply(f"{reply_text}")
            with stats_lock:
                comments_written += 1
            print("  ✔ Generated reply:", reply_text)
            print("  ✔ Replied")

            time.sleep(REDDIT_RATE_LIMIT_SEC)   # be polite to Reddit
        except Exception as exc:
            print("  ⚠️  Error:", exc)
            time.sleep(10)                      # basic backoff

if __name__ == "__main__":
    main()
