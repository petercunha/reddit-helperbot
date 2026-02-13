# Repository Guidelines

## Project Structure & Module Organization
Core runtime code lives in the repository root:
- `main.py`: entry point; validates env, registers signals, starts listener.
- `reddit_listener.py`: Reddit stream loop, trigger filtering, retry/backoff, stats.
- `ai_responder.py` + `llm.py`: AI reply composition and tool-calling loop.
- `tools.py`: web tools (`web_search`, `web_fetch`, `web_render`).
- `transcript.py`: Reddit thread/context extraction.
- `config.py`: env loading, client initialization, runtime constants.
- `prompt_templates.py` + `prompts/system_prompt.txt`: prompt loading and editable system prompt text.
- `test_helperbot.py`: unit/integration-style tests.

## Build, Test, and Development Commands
- `python main.py`: run the bot locally (requires `.env` values).
- `python -m unittest test_helperbot -v`: run full test suite.
- `make dev`: build and run with Docker Compose in foreground.
- `make watch`: run with auto-rebuild on file changes.
- `make deploy` / `make deploy-fresh`: detached deployment (fresh recreates containers).
- `make logs`, `make ps`, `make down`: inspect logs/status and stop stack.

## Coding Style & Naming Conventions
- Language: Python 3.9+; use 4-space indentation and PEP 8 style.
- Prefer clear, small modules with single responsibilities.
- Naming: `snake_case` for functions/variables, `UPPER_SNAKE_CASE` for constants, `PascalCase` for classes/dataclasses.
- Keep prompt text in `prompts/*.txt`; avoid embedding large prompt blocks in code.
- Favor explicit typing where practical (`Callable`, `Pattern`, `dict[str, Any]`).

## Testing Guidelines
- Framework: built-in `unittest` with `unittest.mock` for API/network boundaries.
- Test names should start with `test_` and describe behavior (`test_reply_with_retry_retries_on_failure`).
- Add or update tests with every behavior change; prioritize listener retry/backoff flow, LLM tool-call loop behavior, and prompt/template loading.

## Commit & Pull Request Guidelines
- Current history uses short, imperative summaries (e.g., `Refactor codebase`, `Add Docker Compose support...`), though style is mixed. Prefer concise imperative messages.
- Keep commits focused and logically grouped.
- PRs should include what changed and why, risk/rollback notes for runtime behavior, test evidence (command + result), and linked issue/context when applicable.

## Security & Configuration Tips
- Never commit secrets; keep credentials in `.env` only.
- Start from `.env.example` and verify required vars before running.
- Treat Reddit and OpenRouter credentials as production secrets.
