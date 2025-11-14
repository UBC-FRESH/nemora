# `nemora.ingest`

Nemora’s ingest package wraps reusable dataset abstractions and pipelines that
convert raw inventory releases (FAIB PSP, FIA Datamart, etc.) into the tidy stand
tables consumed by `nemora.distfit`, `nemora.sampling`, and downstream tooling.
The key entry points mirror the concepts introduced in the
[`Ingest Module` how-to](../howto/ingest.md):

- `DatasetSource` / `DatasetFetcher` describe how to locate and cache raw files.
- `TransformPipeline` orchestrates composable DataFrame transformations.
- Submodules (`faib`, `fia`, `hps`) provide dataset-specific helpers and CLI-ready
  pipelines for stand-table and HPS tally generation.

```{seealso}
- [`docs/howto/ingest.md`](../howto/ingest.md) for step-by-step ingest workflows.
- [`nemora.cli`](../api/nemora.md) for Typer commands such as `ingest-faib`,
  `faib-manifest`, and `ingest-faib-hps` that wrap these helpers.
```

## Package API

```{automodule} nemora.ingest
:members:
:undoc-members:
:show-inheritance:
```

## Dataset helpers

### FAIB (`nemora.ingest.faib`)

```{automodule} nemora.ingest.faib
:members:
:undoc-members:
:show-inheritance:
```

### FIA (`nemora.ingest.fia`)

```{automodule} nemora.ingest.fia
:members:
:undoc-members:
:show-inheritance:
```

### HPS (`nemora.ingest.hps`)

```{automodule} nemora.ingest.hps
:members:
:undoc-members:
:show-inheritance:
```
