"""
transcript.py – Build conversation transcripts from Reddit threads.

Handles image-URL extraction and thread traversal for providing
context to the LLM.
"""

import logging
import textwrap

import praw.models

import config
from config import (
    IMAGE_URL_DIRECT_PATTERN,
    INDENT,
    MARKDOWN_IMAGE_PATTERN,
    MAX_IMAGES_TO_SEND,
)

logger = logging.getLogger("helperbot.transcript")


def extract_image_urls_from_text(text: str) -> list[str]:
    """Extract direct image URLs and Markdown image links from text."""
    urls: list[str] = []
    if not text:
        return urls
    for match in IMAGE_URL_DIRECT_PATTERN.finditer(text):
        urls.append(match.group(0))
    for match in MARKDOWN_IMAGE_PATTERN.finditer(text):
        urls.append(match.group(1))
    return list(dict.fromkeys(urls))


def build_thread_transcript(
    trigger_comment: praw.models.Comment,
) -> tuple[str, list[str]]:
    """
    Return a markdown-flavoured string representing the entire conversation
    (submission + ancestor chain) that led to *trigger_comment*, and a list
    of image URLs found in the thread.
    """
    sub = trigger_comment.submission
    subreddit_name = trigger_comment.subreddit.display_name
    parts = [f"SUBREDDIT: r/{subreddit_name}"]
    parts.append(f"SUBMISSION URL: https://www.reddit.com{sub.permalink} ")
    if not sub.is_self and sub.url:
        if (
            not IMAGE_URL_DIRECT_PATTERN.fullmatch(sub.url)
            and "v.redd.it" not in sub.url
            and "i.redd.it" not in sub.url
        ):
            parts.append(f"EXTERNAL LINK URL: {sub.url} ")
    parts.append(f"SUBMISSION TITLE: {sub.title.strip()}")

    all_image_urls: list[str] = []

    # Extract images from submission
    if hasattr(sub, "url") and sub.url:
        if IMAGE_URL_DIRECT_PATTERN.fullmatch(sub.url):
            all_image_urls.append(sub.url)
        elif hasattr(sub, "post_hint") and sub.post_hint == "image":
            all_image_urls.append(sub.url)

    if sub.is_self and sub.selftext:
        stripped_selftext = sub.selftext.strip()
        parts.append(stripped_selftext)
        all_image_urls.extend(extract_image_urls_from_text(stripped_selftext))

    # Gallery posts
    if (
        hasattr(sub, "is_gallery")
        and sub.is_gallery
        and hasattr(sub, "media_metadata")
        and sub.media_metadata
    ):
        for _item_id, media_item in sub.media_metadata.items():
            if (
                media_item.get("m")
                and "image" in media_item["m"]
                and media_item.get("s", {}).get("u")
            ):
                all_image_urls.append(
                    media_item["s"]["u"].replace("&amp;", "&")
                )
            elif (
                media_item.get("e") == "Image"
                and media_item.get("s", {}).get("u")
            ):
                all_image_urls.append(
                    media_item["s"]["u"].replace("&amp;", "&")
                )

    parts.append("\n---")

    # Ancestor comments (root -> trigger)
    ancestors = []
    c = trigger_comment
    while c is not None and hasattr(c, "body"):
        ancestors.append(c)
        if c.is_root:
            break
        c = c.parent()
    ancestors.reverse()

    for cm in ancestors:
        author = cm.author.name if cm.author else "[deleted]"
        body = cm.body.strip() or "[empty]"
        all_image_urls.extend(extract_image_urls_from_text(body))
        quoted = textwrap.indent(body, INDENT)
        parts.append(f"{author} wrote:\n{quoted}\n")

    transcript = "\n".join(parts)
    if len(transcript) > config.MAX_CHARS:
        transcript = transcript[-config.MAX_CHARS:]

    unique_image_urls = list(dict.fromkeys(all_image_urls))
    return transcript, unique_image_urls[:MAX_IMAGES_TO_SEND]
