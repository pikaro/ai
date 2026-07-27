# Suggested commands

Run from repository root on Darwin/zsh.

## Environment
- `uv sync --frozen`
- Service-specific: `uv sync --frozen --package ai-assistant`, `ai-stt`, or `ai-tts`

## Verification
- `uvx --from ruff==0.15.18 ruff check .`
- `uvx --from ruff==0.15.18 ruff format --check .`
- `uvx --from pyright==1.1.411 pyright`
- `uv run --frozen python -m unittest discover`
- Existing local alternatives documented in `docs/overview.md`: `ruff check .`, `.venv/bin/pyright`, `.venv/bin/python -m unittest discover`

## Formatting/fixes
- `uv run ruff check --fix`
- `uv run ruff format`
- `pre-commit run --all-files` if pre-commit is installed

## Run entrypoints
- Assistant: `uv run uvicorn assistant.src.main:app --host 0.0.0.0 --port 8080 --workers 1`
- STT/TTS images set their service source directory on PYTHONPATH and run `uvicorn main:app ... --workers 1`; Docker is the documented execution path.

## Images
- `docker build -f assistant/Dockerfile -t ai-assistant .`
- `docker build -f stt/Dockerfile -t ai-stt .`
- `docker build -f tts/Dockerfile -t ai-tts .`

## Navigation
Use `rg`/`rg --files` first; `fd` for file discovery; `git status --short` before and after edits. Standard Darwin commands such as `pwd`, `cd`, `sed`, `git diff`, and `git log` apply.