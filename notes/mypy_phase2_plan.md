# Mypy readiness for Phase 2 tagging

Current blockers (2025-12-09):

- `src/nemora/dataprep/hps.py`: type errors around int/Datetime conversions and sorting key signatures.
- `src/nemora/ingest/fia.py`: pandas read_csv overload mismatches; sort_values/assignment errors.
- `src/nemora/ingest/faib.py`: Series.rename overload mismatch.

Plan to ship Phase 2 with clean `mypy src`:

1. Keep synthesis/fit/sampling targets clean (already passing).
2. Add a temporary `[[tool.mypy.overrides]]` stanza to `pyproject.toml` that ignores errors in the
   three ingest/dataprep modules above (strict=False or ignore_errors=True), to be removed once
   those modules are fixed.
3. When fixing ingest/dataprep, remove the overrides and re-run `mypy src` to confirm a clean run,
   then tag the Phase 2 milestone.
