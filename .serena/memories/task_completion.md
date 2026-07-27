# Task completion checklist

1. Confirm the requested behavior and protected invariants; keep scope minimal.
2. Inspect relevant symbols and neighboring patterns; read `docs/overview.md` before coding.
3. Add/update focused unit tests.
4. Run Ruff lint and formatting check, Pyright, and the relevant unittest suite (full suite when proportionate).
5. Update `docs/overview.md` for user-visible behavior or architecture changes.
6. Review `git diff` and `git status --short`; preserve unrelated user changes.
7. Report changes, rationale, verification, and any high-impact findings.