"""
prompt_templates.py - Prompt template loading for HelperBot.

System prompts are stored as editable text files so non-code changes are easy.
"""

from pathlib import Path


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_prompt_template(filename: str) -> str:
    prompt_path = PROMPTS_DIR / filename
    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(f"Prompt template not found: {prompt_path}") from exc


SYSTEM_PROMPT_TEMPLATE = _load_prompt_template("system_prompt.txt")

PROMPT_HEADER_TEMPLATE = """You are HelperBot, an AI assistant that helps Reddit users by replying to their comments.

Below is the full thread that led to the user's last comment. Use it to craft an accurate, concise reply. Write your final answer
as if you were replying directly to the user on Reddit. Do not include any preamble or explanation, just provide the answer.

--- BEGIN THREAD ---
{thread_text}
--- END OF THREAD ---

USER QUESTION (last comment): {user_question}
""".strip()
