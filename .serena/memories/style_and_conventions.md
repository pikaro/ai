# Style and conventions

- Follow root `AGENTS.md`: correctness first, minimal scoped diffs, no speculative abstractions/refactors/dependencies, and ask before changes to shared code, schemas, architecture, or dependencies.
- Python 3.12 with explicit type hints; async FastAPI/httpx patterns; dataclasses commonly use `frozen=True, slots=True`.
- Ruff targets Python 3.12, line length 100, all lint rules with project exclusions; formatter uses single quotes and four-space indentation.
- Pyright is enabled with strict checks including private usage and constant redefinition.
- Google docstring convention (pydoclint/ruff); public APIs and non-obvious complex logic need docstrings. Comments should explain intent/constraints, not restate code.
- Stable structured logs use descriptive `ID_snake_case` event IDs and structured fields. Avoid payloads at INFO; sensitive request data is DEBUG-only.
- Preserve single-worker/busy-rejection behavior, deterministic prompts/data flow, bounded queues, and low-overhead execution in latency-sensitive audio/LLM paths.
- Update `docs/overview.md` when behavior or structure changes.