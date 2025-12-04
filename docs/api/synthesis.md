# `nemora.synthesis`

Nemora’s synthesis package is beginning with helper utilities that translate
sampling bootstrap results into the DataFrame/metadata payloads that future stem
and stand generators will consume. These helpers standardise how downstream
modules access provenance (distribution, parameters, bins, tallies) alongside
the sampled stems.

```{seealso}
- [`docs/howto/synthesis.md`](../howto/synthesis.md) for the integration
  guide that outlines how bootstrap payloads power upcoming synthesis flows.
- [`docs/howto/sampling.md`](../howto/sampling.md) for background on
  `BootstrapResult` and the bootstrap sampling APIs.
```

## Helper API

```{automodule} nemora.synthesis.helpers
:members:
:undoc-members:
:show-inheritance:
```
