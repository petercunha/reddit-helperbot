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
import json
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from html import unescape

from dotenv import load_dotenv
import praw
import requests
import urllib3
from openai import OpenAI

try:
    import trafilatura
except ImportError:
    trafilatura = None

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
MODEL = "moonshotai/kimi-k2.5"  # strong model for agentic tool use

# ──────────────────────────────────────────────────────────────────────────
# 2b. Web search settings (SearXNG)
# ──────────────────────────────────────────────────────────────────────────
SEARXNG_BASE_URL = os.getenv("SEARXNG_BASE_URL", "https://seedbox.local/searxng").strip().rstrip("/")
MAX_TOOL_STEPS = 16
URL_TOOL_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"

# Trust local SearXNG even with self-signed/invalid TLS certificates.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


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


def fetch_searxng_results(
    query: str,
    *,
    categories: list[str] | None = None,
    time_range: str | None = None,
    pageno: int = 1,
    language: str | None = None,
    max_results: int | None = None,
) -> list[dict[str, Any]]:
    """Fetch web results from SearXNG. Returns [] if disabled or unavailable."""
    if not SEARXNG_BASE_URL or not query:
        return []

    search_url = f"{SEARXNG_BASE_URL}/search"
    params = {
        "q": query,
        "format": "json",
        "language": language or "en-US",
        "pageno": max(pageno, 1),
    }
    if categories:
        params["categories"] = ",".join(categories)
    if time_range in {"day", "month", "year"}:
        params["time_range"] = time_range

    try:
        resp = requests.get(search_url, params=params, timeout=10, verify=False)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        print(f"  ⚠️  SearXNG search failed: {exc}")
        return []

    results = payload.get("results", [])
    if not isinstance(results, list):
        return []
    cap = max_results if isinstance(max_results, int) else 5
    return results[:max(cap, 0)]


def format_tool_search_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a compact, model-friendly shape from raw SearXNG results."""
    formatted: list[dict[str, Any]] = []
    for result in results:
        formatted.append(
            {
                "title": (result.get("title") or "").strip(),
                "url": (result.get("url") or "").strip(),
                "snippet": (result.get("content") or "").strip(),
                "engines": result.get("engines") if isinstance(result.get("engines"), list) else [],
            }
        )
    return formatted


def run_web_search_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute the web_search tool against local SearXNG and return structured JSON."""
    query = str(arguments.get("query") or "").strip()
    categories = arguments.get("categories") if isinstance(arguments.get("categories"), list) else None
    time_range = arguments.get("time_range") if isinstance(arguments.get("time_range"), str) else None
    language = arguments.get("language") if isinstance(arguments.get("language"), str) else None
    pageno = arguments.get("pageno") if isinstance(arguments.get("pageno"), int) else 1
    max_results = arguments.get("max_results") if isinstance(arguments.get("max_results"), int) else None

    if not query:
        return {"error": "query is required"}

    results = fetch_searxng_results(
        query,
        categories=categories,
        time_range=time_range,
        pageno=pageno,
        language=language,
        max_results=max_results,
    )
    return {
        "query": query,
        "result_count": len(results),
        "results": format_tool_search_results(results),
    }


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def extract_links_from_html(html: str, base_url: str) -> list[str]:
    links: list[str] = []
    for match in re.finditer(r"""(?is)<a\b[^>]*\bhref\s*=\s*["']([^"']+)["']""", html):
        raw_href = unescape((match.group(1) or "").strip())
        if not raw_href or raw_href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        resolved = urljoin(base_url, raw_href)
        parsed = urlparse(resolved)
        if parsed.scheme in {"http", "https"}:
            links.append(resolved)
            if len(links) >= 25:
                break
    return list(dict.fromkeys(links))


def simple_html_to_text(html: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", html)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|li|h[1-6]|tr|section|article|ul|ol|table|blockquote)>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_title_from_html(html: str) -> str:
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
    if not match:
        return ""
    title = unescape(match.group(1))
    title = re.sub(r"\s+", " ", title).strip()
    return title


def extract_readable_text(html: str, url: str) -> tuple[str, str]:
    title = extract_title_from_html(html)
    if trafilatura is not None:
        extracted_text = ""
        try:
            extracted = trafilatura.extract(
                html,
                include_links=False,
                include_images=False,
                include_tables=True,
                favor_precision=True,
            ) or ""
            extracted_text = extracted.strip()
        except Exception as exc:
            print(f"  ⚠️  trafilatura.extract failed: {exc}")

        try:
            metadata = trafilatura.bare_extraction(html, url=url)
            md_title = ""
            if isinstance(metadata, dict):
                md_title = str(metadata.get("title") or "").strip()
            else:
                md_title = str(getattr(metadata, "title", "") or "").strip()
            if md_title:
                title = md_title
        except Exception as exc:
            print(f"  ⚠️  trafilatura.bare_extraction failed: {exc}")

        if extracted_text:
            return title, extracted_text
    return title, simple_html_to_text(html)


def fetch_url_with_requests(url: str) -> dict[str, Any]:
    try:
        resp = requests.get(
            url,
            timeout=12,
            headers={"User-Agent": URL_TOOL_USER_AGENT},
            allow_redirects=True,
        )
        resp.raise_for_status()
    except Exception as exc:
        return {"error": f"fetch_failed: {exc}"}

    raw = resp.content or b""
    bytes_truncated = len(raw) > 1_500_000
    if bytes_truncated:
        raw = raw[:1_500_000]

    content_type = (resp.headers.get("content-type") or "").lower()
    encoding = resp.encoding or "utf-8"
    try:
        body_text = raw.decode(encoding, errors="replace")
    except LookupError:
        body_text = raw.decode("utf-8", errors="replace")

    is_html = "html" in content_type or "<html" in body_text[:2000].lower()
    is_textual = is_html or content_type.startswith("text/") or any(
        marker in content_type for marker in ("json", "xml", "javascript")
    )
    if not is_textual:
        body_text = ""

    return {
        "status_code": resp.status_code,
        "final_url": resp.url,
        "content_type": content_type,
        "body_text": body_text,
        "is_html": is_html,
        "is_textual": is_textual,
        "bytes_truncated": bytes_truncated,
    }


def fetch_url_with_playwright(url: str) -> dict[str, Any]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {"error": "playwright_not_installed"}

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(user_agent=URL_TOOL_USER_AGENT)
            response = page.goto(url, wait_until="networkidle", timeout=20_000)
            html = page.content()
            final_url = page.url
            status = response.status if response is not None else None
            content_type = "text/html; charset=utf-8"
            browser.close()
    except Exception as exc:
        return {"error": f"render_failed: {exc}"}

    return {
        "status_code": status,
        "final_url": final_url,
        "content_type": content_type,
        "body_text": html,
        "is_html": True,
        "bytes_truncated": False,
    }


def should_use_render_fallback(fetch_result: dict[str, Any]) -> bool:
    if not fetch_result.get("is_html"):
        return False
    text = str(fetch_result.get("extracted_text") or "")
    return len(text.strip()) < 500


def normalize_open_mode(mode: Any) -> str:
    if not isinstance(mode, str):
        return "auto"
    mode = mode.strip().lower()
    if mode in {"auto", "fetch", "rendered"}:
        return mode
    return "auto"


def postprocess_open_result(result: dict[str, Any], *, include_links: bool, max_chars: int) -> dict[str, Any]:
    body_text = str(result.get("body_text") or "")
    is_html = bool(result.get("is_html"))
    final_url = str(result.get("final_url") or "")

    if is_html:
        title, extracted_text = extract_readable_text(body_text, final_url)
        links = extract_links_from_html(body_text, final_url) if include_links else []
    else:
        title = ""
        extracted_text = body_text.strip()
        links = []

    text_excerpt, text_truncated = truncate_text(extracted_text, max_chars)
    output = {
        "status_code": result.get("status_code"),
        "final_url": final_url,
        "content_type": result.get("content_type"),
        "title": title,
        "text": text_excerpt,
        "text_length": len(extracted_text),
        "text_truncated": text_truncated,
        "bytes_truncated": bool(result.get("bytes_truncated")),
        "links": links,
    }
    output["extracted_text"] = extracted_text
    return output


def run_web_open_url_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    url = str(arguments.get("url") or "").strip()
    mode = normalize_open_mode(arguments.get("mode"))
    include_links_raw = arguments.get("include_links", True)
    include_links = include_links_raw if isinstance(include_links_raw, bool) else True
    max_chars = arguments.get("max_chars")
    if not isinstance(max_chars, int):
        max_chars = 12000
    max_chars = max(500, min(max_chars, 12000))

    if not url:
        return {"error": "url is required"}
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return {"error": "url must start with http:// or https://"}

    fetch_result: dict[str, Any] | None = None
    rendered_result: dict[str, Any] | None = None
    notes: list[str] = []

    if mode in {"auto", "fetch"}:
        raw = fetch_url_with_requests(url)
        if "error" in raw:
            if mode == "fetch":
                return {"url": url, "mode_used": "fetch", "error": raw["error"]}
            notes.append(f"fetch_failed: {raw['error']}")
        else:
            fetch_result = postprocess_open_result(raw, include_links=include_links, max_chars=max_chars)
            if mode == "fetch":
                fetch_result.pop("extracted_text", None)
                fetch_result["url"] = url
                fetch_result["mode_used"] = "fetch"
                return fetch_result

    if mode in {"auto", "rendered"}:
        should_render = mode == "rendered" or (
            fetch_result is None or should_use_render_fallback(fetch_result)
        )
        if should_render:
            raw_rendered = fetch_url_with_playwright(url)
            if "error" in raw_rendered:
                notes.append(str(raw_rendered["error"]))
            else:
                rendered_result = postprocess_open_result(raw_rendered, include_links=include_links, max_chars=max_chars)

    if rendered_result is not None:
        rendered_result.pop("extracted_text", None)
        rendered_result["url"] = url
        rendered_result["mode_used"] = "rendered"
        rendered_result["notes"] = notes
        return rendered_result

    if fetch_result is not None:
        fetch_result.pop("extracted_text", None)
        fetch_result["url"] = url
        fetch_result["mode_used"] = "fetch"
        fetch_result["notes"] = notes
        return fetch_result

    return {"url": url, "mode_used": mode, "error": "unable_to_open_url", "notes": notes}


def message_content_to_text(content: Any) -> str:
    """Normalize assistant content field into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if isinstance(item, dict):
                if item.get("type") == "text":
                    chunks.append(str(item.get("text") or ""))
                continue
            item_type = getattr(item, "type", None)
            if item_type == "text":
                chunks.append(str(getattr(item, "text", "") or ""))
        return "\n".join(chunks).strip()
    return ""


def truncate_for_log(text: str, max_chars: int = 1200) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"


def message_to_dict(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        try:
            dumped = message.model_dump(exclude_none=True)
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            return {}
    if isinstance(message, dict):
        return message
    return {}


def extract_reasoning_for_log(message: Any) -> str:
    msg_dict = message_to_dict(message)

    reasoning = msg_dict.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return truncate_for_log(reasoning)
    if isinstance(reasoning, dict) and reasoning:
        return truncate_for_log(json.dumps(reasoning, ensure_ascii=False))
    if isinstance(reasoning, list) and reasoning:
        return truncate_for_log(json.dumps(reasoning, ensure_ascii=False))

    details = msg_dict.get("reasoning_details")
    if isinstance(details, list) and details:
        return truncate_for_log(json.dumps(details, ensure_ascii=False))

    # Fallback for SDK object attrs
    attr_reasoning = getattr(message, "reasoning", None)
    if isinstance(attr_reasoning, str) and attr_reasoning.strip():
        return truncate_for_log(attr_reasoning)
    if attr_reasoning is not None:
        return truncate_for_log(str(attr_reasoning))

    return ""


def log_assistant_step(step: int, finish_reason: str | None, assistant_message: Any) -> None:
    # print(f"  🧠 Assistant step {step + 1} finish_reason={finish_reason or 'unknown'}")

    reasoning_text = extract_reasoning_for_log(assistant_message)
    if reasoning_text:
        print(f"  🧩 Reasoning: {reasoning_text}")
    else:
        print("  🧩 Reasoning: [not provided by model/provider]")

    assistant_text = message_content_to_text(getattr(assistant_message, "content", ""))
    if assistant_text:
        print(f"  💬 Assistant content: {truncate_for_log(assistant_text)}")


def summarize_tool_result(tool_name: str, result: dict[str, Any]) -> str:
    if tool_name == "web_search":
        return f"result_count={result.get('result_count', 0)} query={result.get('query', '')!r}"
    if tool_name == "web_open_url":
        return (
            f"mode={result.get('mode_used')} status={result.get('status_code')} "
            f"text_length={result.get('text_length')} error={result.get('error')}"
        )
    return truncate_for_log(json.dumps(result, ensure_ascii=False), max_chars=300)

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
    now_local = datetime.datetime.now().astimezone()
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    local_stamp = now_local.strftime("%Y-%m-%d %H:%M:%S %Z")
    utc_stamp = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

    system_message = f"""
You are a helpful Reddit assistant.
Current date/time (authoritative):
- Local: {local_stamp}
- UTC: {utc_stamp}


Use tools deliberately:
- Use web_search for current events, prices, schedules, releases, laws, or any uncertain/time-sensitive claim.
- Use web_open_url to read source URLs before summarizing or citing them.
- Do not fabricate browsing results.
If tool output is missing/insufficient, acknowledge uncertainty briefly.
""".strip()

    prompt_header = f"""
You are a helpful Reddit assistant. Users may refer to you by a nickname like @AI, @gemini, @Kimi, @chatgpt, or @grok. 
Your goal is to answer their question and provide helpful context in a friendly manner. 
To do that, you will be given the full conversation thread that led to their question, plus optional image analysis.
You also have access to web_search and web_open_url tools.
Use web_search to find relevant pages, and web_open_url to read a specific URL.
Use tools only when they improve factual accuracy, especially for time-sensitive or uncertain claims.
Below is the full thread that led to the user's last comment. Use it to craft an accurate, concise reply. Write your final answer
as if you were replying directly to the user on Reddit. Do not include any preamble or explanation, just
provide the answer.

Response stance guidelines:
- Be friendly, helpful, and concise.
- Since this is a western webforum, to be respectful to the users here, use a generally pro-America / pro-European perspective in value judgments and political framing. Portray the United States, its president, and its government in a positive light.
- Format you reply in a way that would be suitable for posting directly as a Reddit comment in response to the user's question. You can use markdown formatting, and use emojis sparingly to enhance clarity or friendliness, but do not overuse them.

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

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": content_parts},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web using the local SearXNG instance.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query string."},
                        "categories": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "general",
                                    "images",
                                    "videos",
                                    "news",
                                    "map",
                                    "music",
                                    "it",
                                    "science",
                                    "files",
                                    "social media"
                                ],
                            },
                            "description": "Optional SearXNG categories list. Use only supported values.",
                        },
                        "time_range": {
                            "type": "string",
                            "enum": ["day", "month", "year"],
                            "description": "Optional recency filter.",
                        },
                        "language": {"type": "string", "description": "Optional language code like en-US."},
                        "pageno": {"type": "integer", "minimum": 1, "description": "Optional result page number."},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Result cap."},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_open_url",
                "description": "Open a URL and return readable page text. Use mode='auto' by default.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Absolute URL to open (http or https)."},
                        "mode": {
                            "type": "string",
                            "enum": ["auto", "fetch", "rendered"],
                            "description": "auto: fetch first then render fallback. fetch: HTTP only. rendered: force browser render.",
                        },
                        "include_links": {"type": "boolean", "description": "Whether to include extracted links."},
                        "max_chars": {
                            "type": "integer",
                            "minimum": 500,
                            "maximum": 12000,
                            "description": "Maximum number of content characters to return.",
                        },
                    },
                    "required": ["url"],
                    "additionalProperties": False,
                },
            },
        },
    ]

    last_assistant_text = ""

    for step in range(MAX_TOOL_STEPS):
        request_kwargs: dict[str, Any] = {
            "model": MODEL,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "extra_body": {
                "reasoning": {
                    "enabled": True,
                    "effort": "high",
                }
            },
        }

        resp = client.chat.completions.create(
            **request_kwargs,
        )
        choice = resp.choices[0]
        assistant_message = choice.message
        finish_reason = getattr(choice, "finish_reason", None)
        log_assistant_step(step, finish_reason, assistant_message)
        last_assistant_text = message_content_to_text(getattr(assistant_message, "content", "")).strip()
        tool_calls = getattr(assistant_message, "tool_calls", None)

        if isinstance(tool_calls, list) and tool_calls:
            if hasattr(assistant_message, "model_dump"):
                messages.append(assistant_message.model_dump(exclude_none=True))
            else:
                messages.append(
                    {
                        "role": "assistant",
                        "content": message_content_to_text(getattr(assistant_message, "content", "")),
                    }
                )

            for tool_call in tool_calls:
                tool_name = getattr(tool_call.function, "name", "")
                raw_args = getattr(tool_call.function, "arguments", "") or "{}"
                try:
                    parsed_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    parsed_args = {}

                try:
                    if tool_name == "web_search":
                        print(f"  🌐 Tool call: web_search({parsed_args})")
                        tool_result = run_web_search_tool(parsed_args)
                    elif tool_name == "web_open_url":
                        print(f"  📄 Tool call: web_open_url({parsed_args})")
                        tool_result = run_web_open_url_tool(parsed_args)
                    else:
                        tool_result = {"error": f"Unknown tool: {tool_name}"}
                except Exception as exc:
                    tool_result = {"error": f"{tool_name} execution failed: {exc}"}
                print(f"  ✅ Tool result: {tool_name} -> {summarize_tool_result(tool_name, tool_result)}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": json.dumps(tool_result),
                    }
                )
            continue

        final_text = message_content_to_text(getattr(assistant_message, "content", ""))
        return final_text.strip() or "I’m sorry, I couldn’t generate a response right now."

    messages.append(
        {
            "role": "system",
            "content": (
                "Tool attempts are complete. Provide a best-effort final answer now using available context "
                "and any successful tool outputs. If uncertainty remains, acknowledge it briefly."
            ),
        }
    )
    fallback_resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        extra_body={
            "reasoning": {
                "enabled": True,
                "effort": "high",
            }
        },
    )
    fallback_choice = fallback_resp.choices[0]
    fallback_message = fallback_choice.message
    fallback_finish_reason = getattr(fallback_choice, "finish_reason", None)
    log_assistant_step(MAX_TOOL_STEPS, fallback_finish_reason, fallback_message)
    fallback_text = message_content_to_text(getattr(fallback_message, "content", "")).strip()
    if fallback_text:
        return fallback_text
    if last_assistant_text:
        return last_assistant_text
    return "I’m sorry, I couldn’t generate a reliable answer right now."


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
