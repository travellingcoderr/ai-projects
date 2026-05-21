# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup & Commands

```bash
make setup          # create .env from .env.example + pip install -r requirements.txt
make run-with       # run the LangChain agent
make run-without    # run the raw OpenAI agent
make lint           # ruff check .
make format         # ruff format .
make clean          # remove __pycache__, .pytest_cache, .ruff_cache, .mypy_cache
```

The Makefile uses `./venv/bin/python`, so activate the venv or use `make` targets rather than calling `python` directly.

Tests are expected in a `tests/` directory (`pytest` is configured in `pyproject.toml`) but none exist yet.

## Architecture

The repo is a side-by-side comparison of the same multi-turn tool-calling agent built two ways — raw OpenAI API vs. LangChain.

**`without_langchain/agent.py`** — single self-contained file. Manually implements everything: JSON tool schemas, a `FUNCTION_MAP` dict for routing, a `while True` agent loop, chunk accumulation for streaming, a `chat_history` list for memory, and inline `@with_timeout` / `@with_error_recovery` decorators.

**`with_langchain/`** — a Python package. The shared decorator logic has been extracted to `utils.py`; each tool lives in its own module under `tools/` and exports a `StructuredTool` instance; `tools/__init__.py` collects them into a single `tools` list that `agent.py` imports. LangChain's `AgentExecutor` replaces the manual loop, `ConversationBufferMemory` replaces the list, and Pydantic `BaseModel` schemas replace hand-written JSON.

| Concern | `without_langchain/agent.py` | `with_langchain/` package |
|---|---|---|
| Tool schemas | Hand-written JSON | Pydantic `BaseModel` + `StructuredTool` |
| Tool routing / agent loop | Manual `FUNCTION_MAP` + `while True` | `AgentExecutor` |
| Streaming | Manual chunk accumulation | `StreamingStdOutCallbackHandler` |
| Memory | Manual `chat_history` list | `ConversationBufferMemory` |
| Shared utilities | Inline in `agent.py` | `with_langchain/utils.py` |

**Adding a new tool to `with_langchain`:** create `with_langchain/tools/<name>.py` (Pydantic input schema + `@with_error_recovery` / `@with_timeout` + `StructuredTool`), then add it to the `tools` list in `with_langchain/tools/__init__.py`.

## Key constraints

- `@with_timeout` uses `signal.SIGALRM` — Unix/macOS only, will not work on Windows.
- `with_langchain/agent.py` imports from `langchain_classic` (not `langchain`); `requirements.txt` currently lists `langchain>=0.2.0`. Verify the installed package name if you hit import errors.
- Linter: `ruff` (line-length 88, Python 3.10 target, rules E/F/I/N/W/B/C4/ARG/PTH).
