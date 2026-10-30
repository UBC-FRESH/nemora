# Phase 9 FHOPS tree-inventory handoff

Date: 2026-10-30

Phase: P9

Task: P9.2

Status: complete.

## Contract

The export targets FHOPS `fhops.tree.inventory.v1`:

- required fields: `tree_id`, `block_id`, `x_m`, `y_m`, `species_code`,
  `dbh_cm`, `height_m`, `status`, `confidence`;
- optional fields: elevation, crown radius, source IDs, merchantable height,
  defect, grade hint;
- units: x/y/elevation/height/crown radius in metres, DBH in centimetres;
- metadata: schema version, inventory id, generated timestamp, source, producer,
  method, CRS, units, review status, block membership, caveats, row count, and
  CSV SHA-256.

## Export

- Source: P8 empirical Gavin Lake stand.
- Trees: **1,978**.
- Species codes: 8.
- Block: `B01`.
- Synthetic one-hectare placement: 100 m × 100 m, deterministic seed 20261030.
- Confidence: 1.0 because this is a synthetic inventory, not an ITD/ITC product.
- Review status: `not_reviewed`.
- Inventory SHA-256: `0c431c492d1875a104bc05fdc7174facc8c60291a974b446986ce0ac6bf0ca58`.

The standalone FHOPS loader accepts the export with `block_ids={'B01'}`:
`is_valid=True`, 1,978 trees, 1 block, 8 species, no errors.

## Reproduce

```bash
python scripts/export_phase9_fhops.py \
  --faib-root data/external/faib/raw \
  --report notes/reports/phase9_fhops_export.json
```

The exported CSV and metadata are written to `build/phase9/fhops/` and are not
tracked. The summary JSON is tracked.

## Limits

- This is a synthetic tree list, not a surveyed stand.
- Coordinates are deterministic synthetic placement, not a tree map.
- The inventory is `not_reviewed`.
- No FHOPS core change is required or included.
