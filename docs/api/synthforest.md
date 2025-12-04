# `nemora.synthforest`

Nemora’s synthforest package is beginning with helper utilities that translate
sampling bootstrap results into the DataFrame/metadata payloads that future stem
and stand generators will consume. These helpers standardise how downstream
modules access provenance (distribution, parameters, bins, tallies) alongside
the sampled stems.

```{seealso}
- [`docs/howto/synthforest.md`](../howto/synthforest.md) for the integration
  guide that outlines how bootstrap payloads power upcoming synthforest flows.
- [`docs/howto/sampling.md`](../howto/sampling.md) for background on
  `BootstrapResult` and the bootstrap sampling APIs.
```

## Helper API

```{automodule} nemora.synthforest.helpers
:members:
:undoc-members:
:show-inheritance:
```
