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
MODEL = "x-ai/grok-4"  # pick any model on OpenRouter

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
TRIGGER = re.compile(r"^\s*(?:\[?u/|@)(?:grok|ai|gpt|gemini|chatgpt)\b", re.I)   # matches u/grok, u/ai, u/gpt, u/gemini, u/chatgpt, and the corresponding @ mentions
SUBS    = ["all"]                             # listen everywhere; tune as needed
REDDIT_RATE_LIMIT_SEC = 10                     # courtesy delay after replying

# context‑window guard
MAX_CHARS = 40_000    # rough safety cap for prompt length
INDENT    = "> "      # quote indent used in transcript

# Maximum number of images to send to the LLM
MAX_IMAGES_TO_SEND = 5

# Regex patterns for extracting image URLs
IMAGE_URL_DIRECT_PATTERN = re.compile(r"https?://\S+\.(?:png|jpg|jpeg|gif|webp|bmp)", re.IGNORECASE)
MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[.*?\]\((https?://\S+\.(?:png|jpg|jpeg|gif|webp|bmp))\)", re.IGNORECASE)

def extract_image_urls_from_text(text: str) -> list[str]:
    """Extracts direct image URLs and Markdown image links from text."""
    urls = []
    if not text: # Ensure text is not None or empty
        return urls
    # Find direct URLs
    for match in IMAGE_URL_DIRECT_PATTERN.finditer(text):
        urls.append(match.group(0))
    # Find Markdown image links
    for match in MARKDOWN_IMAGE_PATTERN.finditer(text):
        urls.append(match.group(1))
    # Remove duplicates while preserving order
    return list(dict.fromkeys(urls))

# ──────────────────────────────────────────────────────────────────────────
# 5. Build a transcript: submission + ancestor comments
# ──────────────────────────────────────────────────────────────────────────
def build_thread_transcript(trigger_comment: praw.models.Comment) -> tuple[str, list[str]]:
    """
    Return a single markdown‑flavoured string representing the entire
    conversation (submission + ancestor chain) that led to trigger_comment,
    and a list of image URLs found in the thread.
    """
    sub = trigger_comment.submission
    subreddit_name = trigger_comment.subreddit.display_name
    parts = [f"SUBREDDIT: r/{subreddit_name}"]
    parts.append(f"SUBMISSION URL: https://www.reddit.com{sub.permalink} ")
    if not sub.is_self and sub.url:
        # Check if the URL is not an image already captured or a reddit media link
        if not IMAGE_URL_DIRECT_PATTERN.fullmatch(sub.url) and "v.redd.it" not in sub.url and "i.redd.it" not in sub.url:
            parts.append(f"EXTERNAL LINK URL: {sub.url} ")
    parts.append(f"SUBMISSION TITLE: {sub.title.strip()}")

    all_image_urls = []

    # Extract images from submission post
    # Check sub.url if it's a direct image link (common for image posts)
    if hasattr(sub, 'url') and sub.url:
        if IMAGE_URL_DIRECT_PATTERN.fullmatch(sub.url):
            all_image_urls.append(sub.url)
        # If post_hint is 'image', sub.url is usually the direct image
        elif hasattr(sub, 'post_hint') and sub.post_hint == 'image':
            all_image_urls.append(sub.url)

    if sub.is_self and sub.selftext:
        stripped_selftext = sub.selftext.strip()
        parts.append(stripped_selftext)
        all_image_urls.extend(extract_image_urls_from_text(stripped_selftext))
    
    # Extract images from gallery posts
    if hasattr(sub, 'is_gallery') and sub.is_gallery and hasattr(sub, 'media_metadata') and sub.media_metadata:
        for item_id, media_item in sub.media_metadata.items():
            # Look for image type 'p' (presumably picture) or direct 'u' (URL)
            if media_item.get('m') and 'image' in media_item['m'] and media_item.get('s', {}).get('u'): # prefer 'u' for direct URL
                 all_image_urls.append(media_item['s']['u'].replace('&amp;', '&'))
            elif media_item.get('e') == 'Image' and media_item.get('s', {}).get('u'): # Fallback for other structures
                 all_image_urls.append(media_item['s']['u'].replace('&amp;', '&'))

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
        all_image_urls.extend(extract_image_urls_from_text(body))
        quoted = textwrap.indent(body, INDENT)
        parts.append(f"{author} wrote:\n{quoted}\n")

    transcript = "\n".join(parts)
    if len(transcript) > MAX_CHARS:           # trim if oversized
        transcript = transcript[-MAX_CHARS:]
    
    # Deduplicate and limit images
    unique_image_urls = list(dict.fromkeys(all_image_urls))
    return transcript, unique_image_urls[:MAX_IMAGES_TO_SEND]


# ──────────────────────────────────────────────────────────────────────────
# 6. Send prompt to LLM and get answer
# ──────────────────────────────────────────────────────────────────────────
def ai_answer(trigger_comment: praw.models.Comment) -> str:
    thread_text, image_urls = build_thread_transcript(trigger_comment)
    user_question = TRIGGER.sub("", trigger_comment.body, 1).strip() or "(no explicit question)"

    prompt_header = f"""
You are a helpful Reddit assistant. Users may refer to you by a nickname like @AI, @gemini, @chatgpt, or @grok. 
Your goal is to answer their question and provide helpful context in a friendly manner. 
To do that, you will be given the full conversation thread that led to their question, as well as tools such as web search and image analysis. 
Below is the full thread that led to the user's last comment. Use it to craft an accurate, concise reply. Write your final answer
as if you were replying directly to the user on Reddit. Do not include any preamble or explanation, just
provide the answer.

--- BEGIN THREAD ---
{thread_text}
--- END OF THREAD ---

USER QUESTION (last comment): {user_question}
""".strip()

    content_parts = [{"type": "text", "text": prompt_header}]

    if image_urls:
        print(f"  ℹ️ Including {len(image_urls)} image(s) in the prompt:")
        for url in image_urls:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": url}
            })
            print(f"    🖼️ {url}")
    else:
        print("  ℹ️ No images found or included for this thread.")

    # print("content_parts:", content_parts)

    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": content_parts}],
        extra_body={
            "plugins": [{
                "id": "web",
                "search_prompt": f"A web search was conducted on {current_date}. If you think it would be helpful to the user, you can incorporate the following web search results into your response. IMPORTANT: If you do decide to include link citations, cite them using markdown links named using the domain of the source. Example: [nytimes.com](https://nytimes.com/some-page)."
            }]
        }
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
            print(f"  Trigger comment: \"{comment.body.strip()}\"")
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
