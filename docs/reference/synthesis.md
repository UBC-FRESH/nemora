<!--
Reference stub for the synthesis module. Content will expand as Phase 1+ code lands.
-->

# `nemora.synthesis`

The synthesis package bundles helper modules for:

- `tessellation`: configuration objects describing the rlandscape-style point processes,
  seed counts, and editing knobs that inform Voronoi tiling.
- `stands`: routines that translate vegetation/age-class distributions into reusable
  templates (inspired by the FLG Weibull-based workflow).
- `exporters`: JSON/GeoJSON convenience functions for persisting synthesis metadata
  and placeholder geometries while the full CLI matures.

```{eval-rst}
.. automodule:: nemora.synthesis.tessellation
   :members:

.. automodule:: nemora.synthesis.stands
   :members:

.. automodule:: nemora.synthesis.exporters
   :members:
```
