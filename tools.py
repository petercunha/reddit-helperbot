"""
tools.py – Web-search and URL-fetching tools exposed to the LLM.

Contains SearXNG integration (web_search) and URL fetching/rendering
(web_open_url), plus all HTML parsing helpers.
"""

import json
import logging
import re
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

# ── SearXNG search ───────────────────────────────────────────────────────


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

    try:
        resp = requests.get(search_url, params=params, timeout=10, verify=False)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        logger.warning("SearXNG search failed: %s", exc)
        return []

    results = payload.get("results", [])
    if not isinstance(results, list):
        return []
    cap = max_results if isinstance(max_results, int) else 5
    return results[: max(cap, 0)]


def format_tool_search_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a compact, model-friendly shape from raw SearXNG results."""
    formatted: list[dict[str, Any]] = []
    for result in results:
        formatted.append(
            {
                "title": (result.get("title") or "").strip(),
                "url": (result.get("url") or "").strip(),
                "snippet": (result.get("content") or "").strip(),
                "engines": result.get("engines")
                if isinstance(result.get("engines"), list)
                else [],
            }
        )
    return formatted


def run_web_search_tool(arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute the web_search tool against local SearXNG and return structured JSON."""
    query = str(arguments.get("query") or "").strip()
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


# ── HTML helpers ─────────────────────────────────────────────────────────


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def extract_links_from_html(html: str, base_url: str) -> list[str]:
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


def simple_html_to_text(html: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", html)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(
        r"(?i)</(p|div|li|h[1-6]|tr|section|article|ul|ol|table|blockquote)>", "\n", text
    )
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


# ── URL fetching ─────────────────────────────────────────────────────────


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


def postprocess_open_result(
    result: dict[str, Any], *, include_links: bool, max_chars: int
) -> dict[str, Any]:
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
            fetch_result = postprocess_open_result(
                raw, include_links=include_links, max_chars=max_chars
            )
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
                rendered_result = postprocess_open_result(
                    raw_rendered, include_links=include_links, max_chars=max_chars
                )

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


# ── Tool-result summary (for logging) ───────────────────────────────────


def summarize_tool_result(tool_name: str, result: dict[str, Any]) -> str:
    if tool_name == "web_search":
        return f"result_count={result.get('result_count', 0)} query={result.get('query', '')!r}"
    if tool_name == "web_open_url":
        return (
            f"mode={result.get('mode_used')} status={result.get('status_code')} "
            f"text_length={result.get('text_length')} error={result.get('error')}"
        )
    from llm import truncate_for_log

    return truncate_for_log(json.dumps(result, ensure_ascii=False), max_chars=300)
