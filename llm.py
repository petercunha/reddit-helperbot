"""
llm.py – LLM interaction: build messages, run the tool-calling loop,
and return a final answer string.
"""

import datetime
import json
import logging
from typing import Any

from config import (
    MODEL,
    MAX_TOOL_STEPS,
    OPENROUTER_TIMEOUT,
    PROMPT_HEADER_TEMPLATE,
    SYSTEM_PROMPT_TEMPLATE,
    TRIGGER,
    client,
)
from tools import run_web_open_url_tool, run_web_search_tool, summarize_tool_result
from transcript import build_thread_transcript

import praw.models

logger = logging.getLogger("helperbot.llm")


# ── Message helpers ──────────────────────────────────────────────────────


def message_content_to_text(content: Any) -> str:
    """Normalize assistant content field into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
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
    if isinstance(reasoning, (dict, list)) and reasoning:
        return truncate_for_log(json.dumps(reasoning, ensure_ascii=False))

    details = msg_dict.get("reasoning_details")
    if isinstance(details, list) and details:
        return truncate_for_log(json.dumps(details, ensure_ascii=False))

    attr_reasoning = getattr(message, "reasoning", None)
    if isinstance(attr_reasoning, str) and attr_reasoning.strip():
        return truncate_for_log(attr_reasoning)
    if attr_reasoning is not None:
        return truncate_for_log(str(attr_reasoning))

    return ""


def log_assistant_step(
    step: int, finish_reason: str | None, assistant_message: Any
) -> None:
    reasoning_text = extract_reasoning_for_log(assistant_message)
    if reasoning_text:
        logger.info("Reasoning: %s", reasoning_text)
    else:
        logger.info("Reasoning: [not provided by model/provider]")

    assistant_text = message_content_to_text(
        getattr(assistant_message, "content", "")
    )
    if assistant_text:
        logger.info("Assistant content: %s", truncate_for_log(assistant_text))


# ── Tool definitions (sent to the LLM) ──────────────────────────────────

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using the local SearXNG instance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string.",
                    },
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
                                "social media",
                            ],
                        },
                        "description": "Optional SearXNG categories list. Use only supported values.",
                    },
                    "time_range": {
                        "type": "string",
                        "enum": ["day", "month", "year"],
                        "description": "Optional recency filter.",
                    },
                    "language": {
                        "type": "string",
                        "description": "Optional language code like en-US.",
                    },
                    "pageno": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional result page number.",
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Result cap.",
                    },
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
                    "url": {
                        "type": "string",
                        "description": "Absolute URL to open (http or https).",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["auto", "fetch", "rendered"],
                        "description": "auto: fetch first then render fallback. fetch: HTTP only. rendered: force browser render.",
                    },
                    "include_links": {
                        "type": "boolean",
                        "description": "Whether to include extracted links.",
                    },
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


# ── Core LLM call ────────────────────────────────────────────────────────


def _execute_tool(tool_name: str, parsed_args: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a tool call by name and return the result dict."""
    if tool_name == "web_search":
        logger.info("Tool call: web_search(%s)", parsed_args)
        return run_web_search_tool(parsed_args)
    if tool_name == "web_open_url":
        logger.info("Tool call: web_open_url(%s)", parsed_args)
        return run_web_open_url_tool(parsed_args)
    return {"error": f"Unknown tool: {tool_name}"}


def ai_answer(trigger_comment: praw.models.Comment) -> str:
    """Build context from the Reddit thread and run the LLM tool-calling loop."""
    thread_text, image_urls = build_thread_transcript(trigger_comment)
    user_question = (
        TRIGGER.sub("", trigger_comment.body, 1).strip() or "(no explicit question)"
    )

    now_local = datetime.datetime.now().astimezone()
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    local_stamp = now_local.strftime("%Y-%m-%d %H:%M:%S %Z")
    utc_stamp = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

    system_message = SYSTEM_PROMPT_TEMPLATE.format(
        local_stamp=local_stamp, utc_stamp=utc_stamp
    )
    prompt_header = PROMPT_HEADER_TEMPLATE.format(
        thread_text=thread_text, user_question=user_question
    )

    content_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt_header}]

    if image_urls:
        logger.info("Including %d image(s) in the prompt:", len(image_urls))
        for url in image_urls:
            content_parts.append({"type": "image_url", "image_url": {"url": url}})
            logger.info("  Image: %s", url)
    else:
        logger.info("No images found or included for this thread.")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": content_parts},
    ]

    last_assistant_text = ""

    for step in range(MAX_TOOL_STEPS):
        request_kwargs: dict[str, Any] = {
            "model": MODEL,
            "messages": messages,
            "tools": TOOL_DEFINITIONS,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "timeout": OPENROUTER_TIMEOUT,
            "extra_body": {
                "reasoning": {
                    "enabled": True,
                    "effort": "high",
                }
            },
        }

        resp = client.chat.completions.create(**request_kwargs)
        choice = resp.choices[0]
        assistant_message = choice.message
        finish_reason = getattr(choice, "finish_reason", None)
        log_assistant_step(step, finish_reason, assistant_message)
        last_assistant_text = message_content_to_text(
            getattr(assistant_message, "content", "")
        ).strip()
        tool_calls = getattr(assistant_message, "tool_calls", None)

        if isinstance(tool_calls, list) and tool_calls:
            if hasattr(assistant_message, "model_dump"):
                messages.append(assistant_message.model_dump(exclude_none=True))
            else:
                messages.append(
                    {
                        "role": "assistant",
                        "content": message_content_to_text(
                            getattr(assistant_message, "content", "")
                        ),
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
                    tool_result = _execute_tool(tool_name, parsed_args)
                except Exception as exc:
                    tool_result = {
                        "error": f"{tool_name} execution failed: {exc}"
                    }
                logger.info(
                    "Tool result: %s -> %s",
                    tool_name,
                    summarize_tool_result(tool_name, tool_result),
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": json.dumps(tool_result),
                    }
                )
            continue

        final_text = message_content_to_text(
            getattr(assistant_message, "content", "")
        )
        return final_text.strip() or "I'm sorry, I couldn't generate a response right now."

    # Exhausted tool steps – ask for a best-effort wrap-up
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
        timeout=OPENROUTER_TIMEOUT,
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
    fallback_text = message_content_to_text(
        getattr(fallback_message, "content", "")
    ).strip()
    if fallback_text:
        return fallback_text
    if last_assistant_text:
        return last_assistant_text
    return "I'm sorry, I couldn't generate a reliable answer right now."
