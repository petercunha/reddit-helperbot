"""
tools.py – Web tools exposed to the LLM agent.

Three tools:
  1. web_search  – Search the web via a self-hosted SearXNG instance.
  2. web_fetch   – Lightweight HTTP GET that returns extracted readable text.
  3. web_render  – Full Playwright browser render for JS-heavy pages.

Each `run_*` function accepts a dict of arguments (as parsed from the LLM's
tool call JSON) and returns a dict suitable for serialising back as a tool
result message.
"""

import json
import logging
import re
import time
from html import unescape
from typing import Any
from urllib.parse import urljoin, urlparse

import requests

from config import SEARXNG_BASE_URL, URL_TOOL_USER_AGENT

try:
    import trafilatura
except ImportError:
    trafilatura = None

logger = logging.getLogger("helperbot.tools")

# ─────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────

# Simple in-memory URL cache: url -> (timestamp, result_dict)
_url_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_URL_CACHE_TTL = 300  # 5 minutes


def _get_cached(url: str) -> dict[str, Any] | None:
    entry = _url_cache.get(url)
    if entry is None:
        return None
    ts, result = entry
    if time.time() - ts > _URL_CACHE_TTL:
        del _url_cache[url]
        return None
    logger.info("Cache hit for %s", url)
    return result


def _set_cached(url: str, result: dict[str, Any]) -> None:
    _url_cache[url] = (time.time(), result)


def _validate_url(url: str) -> str | None:
    """Return an error string if the URL is invalid, or None if OK."""
    if not url:
        return "url is required"
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return "url must start with http:// or https://"
    return None


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    """Truncate text to max_chars. Returns (text, was_truncated)."""
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


# ─────────────────────────────────────────────────────────────────────────
# HTML parsing
# ─────────────────────────────────────────────────────────────────────────


def extract_title_from_html(html: str) -> str:
    """Extract the <title> text from an HTML document."""
    match = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
    if not match:
        return ""
    title = unescape(match.group(1))
    return re.sub(r"\s+", " ", title).strip()


def simple_html_to_text(html: str) -> str:
    """Regex-based fallback: strip tags and collapse whitespace."""
    text = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", html)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(
        r"(?i)</(p|div|li|h[1-6]|tr|section|article|ul|ol|table|blockquote)>",
        "\n",
        text,
    )
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_readable_text(html: str, url: str) -> tuple[str, str]:
    """
    Extract (title, readable_text) from HTML.
    Uses trafilatura when available, falls back to regex stripping.
    """
    title = extract_title_from_html(html)

    if trafilatura is not None:
        extracted_text = ""
        try:
            extracted = (
                trafilatura.extract(
                    html,
                    include_links=False,
                    include_images=False,
                    include_tables=True,
                    favor_precision=True,
                )
                or ""
            )
            extracted_text = extracted.strip()
        except Exception as exc:
            logger.warning("trafilatura.extract failed: %s", exc)

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
            logger.warning("trafilatura.bare_extraction failed: %s", exc)

        if extracted_text:
            return title, extracted_text

    return title, simple_html_to_text(html)


def extract_links_from_html(html: str, base_url: str) -> list[str]:
    """Extract up to 25 unique absolute http(s) links from HTML."""
    links: list[str] = []
    for match in re.finditer(
        r"""(?is)<a\b[^>]*\bhref\s*=\s*["']([^"']+)["']""", html
    ):
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


def _detect_content_type(
    content_type_header: str, body_text: str
) -> tuple[bool, bool]:
    """Return (is_html, is_textual) based on Content-Type and body sniffing."""
    ct = content_type_header.lower()
    is_html = "html" in ct or "<html" in body_text[:2000].lower()
    is_textual = is_html or ct.startswith("text/") or any(
        marker in ct for marker in ("json", "xml", "javascript")
    )
    return is_html, is_textual


def _format_json_if_applicable(content_type: str, body: str) -> str | None:
    """If the response is JSON, return a pretty-printed version."""
    if "json" not in content_type.lower():
        return None
    try:
        parsed = json.loads(body)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────
# 1. web_search – SearXNG
# ─────────────────────────────────────────────────────────────────────────

SEARXNG_MAX_RETRIES = 2
SEARXNG_RETRY_DELAY = 1  # seconds


def _fetch_searxng(
    query: str,
    *,
    categories: list[str] | None = None,
    time_range: str | None = None,
    pageno: int = 1,
    language: str | None = None,
    max_results: int | None = None,
) -> list[dict[str, Any]]:
    """
    Query SearXNG with automatic retry on transient failures.
    Returns raw result dicts from the SearXNG JSON API, or [].
    """
    if not SEARXNG_BASE_URL or not query:
        return []

    search_url = f"{SEARXNG_BASE_URL}/search"
    params: dict[str, Any] = {
        "q": query,
        "format": "json",
        "language": language or "en-US",
        "pageno": max(pageno, 1),
    }
    if categories:
        params["categories"] = ",".join(categories)
    if time_range in {"day", "month", "year"}:
        params["time_range"] = time_range

    last_exc: Exception | None = None
    for attempt in range(1, SEARXNG_MAX_RETRIES + 2):
        try:
            resp = requests.get(
                search_url, params=params, timeout=10, verify=False
            )
            resp.raise_for_status()
            payload = resp.json()
            results = payload.get("results", [])
            if not isinstance(results, list):
                return []
            cap = max_results if isinstance(max_results, int) else 5
            return results[: max(cap, 0)]
        except Exception as exc:
            last_exc = exc
            if attempt <= SEARXNG_MAX_RETRIES:
                logger.warning(
                    "SearXNG search attempt %d/%d failed: %s – retrying in %ds",
                    attempt,
                    SEARXNG_MAX_RETRIES + 1,
                    exc,
                    SEARXNG_RETRY_DELAY,
                )
                time.sleep(SEARXNG_RETRY_DELAY)

    logger.warning("SearXNG search failed after %d attempts: %s", SEARXNG_MAX_RETRIES + 1, last_exc)
    return []


def _deduplicate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate results by URL, merging engine lists."""
    seen: dict[str, dict[str, Any]] = {}
    for r in results:
        url = (r.get("url") or "").strip()
        if not url:
            continue
        if url in seen:
            # Merge engines from the duplicate into the first occurrence
            existing_engines = seen[url].get("engines", [])
            new_engines = r.get("engines", [])
            if isinstance(new_engines, list):
                merged = list(dict.fromkeys(existing_engines + new_engines))
                seen[url]["engines"] = merged
        else:
            seen[url] = r
    return list(seen.values())


def _format_search_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shape raw SearXNG results into a compact, model-friendly format."""
    formatted: list[dict[str, Any]] = []
    for r in results:
        entry: dict[str, Any] = {
            "title": (r.get("title") or "").strip(),
            "url": (r.get("url") or "").strip(),
            "snippet": (r.get("content") or "").strip(),
            "engines": r.get("engines")
            if isinstance(r.get("engines"), list)
            else [],
        }
        # Include published_date when available (helps LLM judge recency)
        pub_date = r.get("publishedDate") or r.get("published_date") or ""
        if isinstance(pub_date, str) and pub_date.strip():
            entry["published_date"] = pub_date.strip()
        formatted.append(entry)
    return formatted

# Keep old name as alias so existing tests still work
format_tool_search_results = _format_search_results


def run_web_search_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Execute the web_search tool: query SearXNG, deduplicate, and return
    structured results.
    """
    query = str(arguments.get("query") or "").strip()
    if not query:
        return {"error": "query is required"}

    categories = (
        arguments.get("categories")
        if isinstance(arguments.get("categories"), list)
        else None
    )
    time_range = (
        arguments.get("time_range")
        if isinstance(arguments.get("time_range"), str)
        else None
    )
    language = (
        arguments.get("language")
        if isinstance(arguments.get("language"), str)
        else None
    )
    pageno = (
        arguments.get("pageno") if isinstance(arguments.get("pageno"), int) else 1
    )
    max_results = (
        arguments.get("max_results")
        if isinstance(arguments.get("max_results"), int)
        else None
    )

    raw_results = _fetch_searxng(
        query,
        categories=categories,
        time_range=time_range,
        pageno=pageno,
        language=language,
        max_results=max_results,
    )
    deduped = _deduplicate_results(raw_results)
    formatted = _format_search_results(deduped)

    return {
        "query": query,
        "result_count": len(formatted),
        "results": formatted,
    }


# ─────────────────────────────────────────────────────────────────────────
# 2. web_fetch – lightweight HTTP GET
# ─────────────────────────────────────────────────────────────────────────

MAX_FETCH_BYTES = 1_500_000  # ~1.5 MB cap on response body
DEFAULT_MAX_CHARS = 20_000


def _http_get(url: str) -> dict[str, Any]:
    """
    Perform a plain HTTP GET and return raw response metadata.
    Returns a dict with either an "error" key or response fields.
    """
    try:
        resp = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": URL_TOOL_USER_AGENT},
            allow_redirects=True,
        )
        resp.raise_for_status()
    except Exception as exc:
        return {"error": f"HTTP request failed: {exc}"}

    raw = resp.content or b""
    bytes_truncated = len(raw) > MAX_FETCH_BYTES
    if bytes_truncated:
        raw = raw[:MAX_FETCH_BYTES]

    content_type = (resp.headers.get("content-type") or "").lower()
    encoding = resp.encoding or "utf-8"
    try:
        body_text = raw.decode(encoding, errors="replace")
    except LookupError:
        body_text = raw.decode("utf-8", errors="replace")

    is_html, is_textual = _detect_content_type(content_type, body_text)

    return {
        "status_code": resp.status_code,
        "final_url": str(resp.url),
        "content_type": content_type,
        "body_text": body_text if is_textual else "",
        "is_html": is_html,
        "is_textual": is_textual,
        "bytes_truncated": bytes_truncated,
    }


def run_web_fetch_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Lightweight URL fetch via HTTP GET.

    Fetches the page, extracts readable text (using trafilatura when
    available), and returns structured metadata. No JavaScript execution.
    Good for articles, docs, APIs, and any page that doesn't require
    client-side rendering.
    """
    url = str(arguments.get("url") or "").strip()
    url_error = _validate_url(url)
    if url_error:
        return {"error": url_error}

    include_links = arguments.get("include_links", True)
    if not isinstance(include_links, bool):
        include_links = True

    max_chars = arguments.get("max_chars")
    if not isinstance(max_chars, int):
        max_chars = DEFAULT_MAX_CHARS
    max_chars = max(500, min(max_chars, DEFAULT_MAX_CHARS))

    # Check cache
    cache_key = f"fetch:{url}"
    cached = _get_cached(cache_key)
    if cached is not None:
        # Re-truncate to the requested max_chars (may differ from cached call)
        text = cached.get("_full_text", cached.get("text", ""))
        excerpt, was_truncated = truncate_text(text, max_chars)
        result = {**cached, "text": excerpt, "text_truncated": was_truncated}
        result.pop("_full_text", None)
        return result

    raw = _http_get(url)
    if "error" in raw:
        return {"url": url, "error": raw["error"]}

    body_text = raw["body_text"]
    final_url = raw["final_url"]
    content_type = raw["content_type"]
    is_html = raw["is_html"]

    # JSON responses: pretty-print instead of extracting "readable text"
    pretty_json = _format_json_if_applicable(content_type, body_text)
    if pretty_json is not None:
        text = pretty_json
        title = ""
        links: list[str] = []
    elif is_html:
        title, text = extract_readable_text(body_text, final_url)
        links = extract_links_from_html(body_text, final_url) if include_links else []
    else:
        title = ""
        text = body_text.strip()
        links = []

    excerpt, text_truncated = truncate_text(text, max_chars)

    result: dict[str, Any] = {
        "url": url,
        "final_url": final_url,
        "status_code": raw["status_code"],
        "content_type": content_type,
        "title": title,
        "text": excerpt,
        "text_length": len(text),
        "text_truncated": text_truncated,
        "bytes_truncated": raw["bytes_truncated"],
    }
    if include_links and links:
        result["links"] = links

    # Cache with full text so future calls with different max_chars still work
    _set_cached(cache_key, {**result, "_full_text": text})

    return result


# ─────────────────────────────────────────────────────────────────────────
# 3. web_render – Playwright browser render
# ─────────────────────────────────────────────────────────────────────────

DEFAULT_RENDER_MAX_CHARS = 20_000


def run_web_render_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Render a URL in a headless Chromium browser via Playwright and return
    the fully-rendered page text.

    Use this for JavaScript-heavy pages (SPAs, Twitter/X, Reddit, etc.),
    pages that returned little or no content from web_fetch, or sites that
    block simple HTTP requests. This is slower and more resource-intensive
    than web_fetch — prefer web_fetch for most URLs.
    """
    url = str(arguments.get("url") or "").strip()
    url_error = _validate_url(url)
    if url_error:
        return {"error": url_error}

    include_links = arguments.get("include_links", True)
    if not isinstance(include_links, bool):
        include_links = True

    max_chars = arguments.get("max_chars")
    if not isinstance(max_chars, int):
        max_chars = DEFAULT_RENDER_MAX_CHARS
    max_chars = max(500, min(max_chars, DEFAULT_RENDER_MAX_CHARS))

    wait_seconds = arguments.get("wait_seconds")
    if not isinstance(wait_seconds, (int, float)) or wait_seconds < 0:
        wait_seconds = 0
    wait_seconds = min(wait_seconds, 10)

    # Check cache
    cache_key = f"render:{url}"
    cached = _get_cached(cache_key)
    if cached is not None:
        text = cached.get("_full_text", cached.get("text", ""))
        excerpt, was_truncated = truncate_text(text, max_chars)
        result = {**cached, "text": excerpt, "text_truncated": was_truncated}
        result.pop("_full_text", None)
        return result

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {
            "url": url,
            "error": "playwright is not installed. Run: pip install playwright && playwright install chromium",
        }

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(user_agent=URL_TOOL_USER_AGENT)
            response = page.goto(url, wait_until="networkidle", timeout=30_000)

            if wait_seconds > 0:
                page.wait_for_timeout(int(wait_seconds * 1000))

            html = page.content()
            final_url = page.url
            status_code = response.status if response is not None else None
            browser.close()
    except Exception as exc:
        return {"url": url, "error": f"Browser render failed: {exc}"}

    title, text = extract_readable_text(html, final_url)
    links = extract_links_from_html(html, final_url) if include_links else []

    excerpt, text_truncated = truncate_text(text, max_chars)

    result: dict[str, Any] = {
        "url": url,
        "final_url": final_url,
        "status_code": status_code,
        "title": title,
        "text": excerpt,
        "text_length": len(text),
        "text_truncated": text_truncated,
    }
    if include_links and links:
        result["links"] = links

    _set_cached(cache_key, {**result, "_full_text": text})

    return result


# ─────────────────────────────────────────────────────────────────────────
# Tool-result summaries (for logging)
# ─────────────────────────────────────────────────────────────────────────


def summarize_tool_result(tool_name: str, result: dict[str, Any]) -> str:
    """Return a concise one-line summary of a tool result for logging."""
    if tool_name == "web_search":
        return (
            f"result_count={result.get('result_count', 0)} "
            f"query={result.get('query', '')!r}"
        )
    if tool_name in {"web_fetch", "web_render"}:
        return (
            f"status={result.get('status_code')} "
            f"text_length={result.get('text_length')} "
            f"title={result.get('title', '')!r} "
            f"error={result.get('error')}"
        )
    return json.dumps(result, ensure_ascii=False)[:300]
