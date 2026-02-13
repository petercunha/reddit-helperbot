#!/usr/bin/env python3
"""
main.py – Entry point for HelperBot.

Validates configuration, installs signal handlers, and starts the Reddit
comment listener loop.
"""

import signal
import sys
import threading
from typing import Callable

from ai_responder import build_reply_text
from config import (
    REDDIT_RATE_LIMIT_SEC,
    SUBS,
    TRIGGER,
    logger,
    reddit,
    validate_env,
)
from reddit_listener import run_comment_listener


def _build_signal_handler(
    shutdown_event: threading.Event,
) -> Callable[[int, object], None]:
    def _handle_signal(signum: int, _frame: object) -> None:
        sig_name = signal.Signals(signum).name
        logger.info("Received %s - shutting down gracefully...", sig_name)
        shutdown_event.set()

    return _handle_signal


def register_signal_handlers(shutdown_event: threading.Event) -> None:
    """Install SIGINT/SIGTERM handlers for graceful shutdown."""
    handler = _build_signal_handler(shutdown_event)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def main() -> None:
    validate_env()
    logger.info("helperbot is live")

    shutdown_event = threading.Event()
    register_signal_handlers(shutdown_event)

    exit_code = run_comment_listener(
        reddit_client=reddit,
        subs=SUBS,
        trigger=TRIGGER,
        responder=build_reply_text,
        reddit_rate_limit_sec=REDDIT_RATE_LIMIT_SEC,
        shutdown_event=shutdown_event,
        bot_logger=logger,
    )

    logger.info("Shutdown complete.")
    if exit_code:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
