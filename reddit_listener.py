"""
reddit_listener.py - Reddit comment stream orchestration.

Owns stream retries, trigger matching, reply posting retries, and stats logging.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Pattern


MAX_STREAM_RETRIES = 5
STREAM_RETRY_BACKOFF = [10, 30, 60, 120, 300]  # seconds


@dataclass
class ListenerStats:
    comments_read: int = 0
    comments_written: int = 0


def _log_status(
    *,
    stats: ListenerStats,
    stats_lock: threading.Lock,
    shutdown_event: threading.Event,
    bot_logger: logging.Logger,
) -> None:
    """Log listener stats every 60 seconds until shutdown is requested."""
    while not shutdown_event.is_set():
        with stats_lock:
            comments_read = stats.comments_read
            comments_written = stats.comments_written
        bot_logger.info(
            "Comments read: %d, Comments written: %d",
            comments_read,
            comments_written,
        )
        shutdown_event.wait(60)


def _reply_with_retry(
    comment: object,
    text: str,
    retries: int = 3,
    *,
    bot_logger: logging.Logger | None = None,
) -> None:
    """Post a reply, retrying on transient Reddit API failures."""
    active_logger = bot_logger or logging.getLogger("helperbot")
    for attempt in range(retries):
        try:
            comment.reply(text)  # type: ignore[union-attr]
            return
        except Exception as exc:
            if attempt < retries - 1:
                wait = 2 ** (attempt + 1)
                active_logger.warning(
                    "comment.reply() failed (attempt %d/%d): %s - retrying in %ds",
                    attempt + 1,
                    retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                raise


def run_comment_listener(
    *,
    reddit_client: Any,
    subs: list[str],
    trigger: Pattern[str],
    responder: Callable[[Any], str],
    reddit_rate_limit_sec: int,
    shutdown_event: threading.Event,
    bot_logger: logging.Logger,
) -> int:
    """
    Run the Reddit comment listener loop.

    Returns 0 on graceful shutdown and 1 if stream retries are exhausted.
    """
    stats = ListenerStats()
    stats_lock = threading.Lock()
    status_thread = threading.Thread(
        target=_log_status,
        kwargs={
            "stats": stats,
            "stats_lock": stats_lock,
            "shutdown_event": shutdown_event,
            "bot_logger": bot_logger,
        },
        daemon=True,
    )
    status_thread.start()

    stream_failures = 0

    while not shutdown_event.is_set():
        try:
            stream = reddit_client.subreddit("+".join(subs)).stream.comments(
                skip_existing=True
            )
            for comment in stream:
                if shutdown_event.is_set():
                    break

                with stats_lock:
                    stats.comments_read += 1

                if not trigger.match(comment.body):
                    continue

                bot_logger.info(
                    "Trigger detected in r/%s | %s",
                    comment.subreddit.display_name,
                    comment.id,
                )
                bot_logger.info("Trigger comment: %r", comment.body.strip())

                try:
                    reply_text = responder(comment)
                    _reply_with_retry(comment, reply_text, bot_logger=bot_logger)
                    with stats_lock:
                        stats.comments_written += 1
                    bot_logger.info("Replied successfully")
                except Exception as exc:
                    bot_logger.error("Failed to generate/post reply: %s", exc)

                time.sleep(reddit_rate_limit_sec)

            # Stream ended normally (shouldn't happen)
            stream_failures = 0

        except Exception as exc:
            if shutdown_event.is_set():
                break
            backoff = STREAM_RETRY_BACKOFF[
                min(stream_failures, len(STREAM_RETRY_BACKOFF) - 1)
            ]
            bot_logger.error(
                "Comment stream error (attempt %d): %s - retrying in %ds",
                stream_failures + 1,
                exc,
                backoff,
            )
            stream_failures += 1
            if stream_failures > MAX_STREAM_RETRIES:
                bot_logger.critical(
                    "Exceeded max stream retries (%d). Exiting.", MAX_STREAM_RETRIES
                )
                return 1
            time.sleep(backoff)

    return 0
