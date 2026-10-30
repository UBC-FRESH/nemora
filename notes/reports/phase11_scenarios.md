# Phase 11 scenario set

Date: 2026-10-30

Phase: P11

Task: P11.1

Status: complete.

## Catalog

| Scenario | Block | Landing | Inventory | work_required |
| --- | --- | --- | --- | ---: |
| gavin-empirical | B01 | L1 | 1,978 trees, P8 empirical | 530.543340 m³ |
| mkrf-local-empirical | B02 | L2 | Malcolm Knapp local six-plot empirical | pending P11.2 |

- Both scenario pins validate.
- Both FHOPS scenarios pass `fhops validate`.
- FHOPS remains one block/one landing per run; the catalog describes the set.
- The Malcolm Knapp inventory is a synthetic transfer case, not an independent
  hold-out-validated stand.
- Scenario and inventory bytes live under `build/phase11/` and are not tracked;
  the catalog records checksums.

## Reproduce

```bash
python scripts/build_phase11_scenarios.py \
  --faib-root data/external/faib/raw \
  --scenario-template path/to/fhops/examples/tree_system/scenario \
  --gavin-inventory-dir build/phase9/fhops \
  --output-root build/phase11 \
  --catalog notes/reports/phase11_scenario_catalog.json
```

## Limits

- One block/one landing per run.
- No simultaneous multi-block optimization.
- Synthetic inventories, not tree maps.
- Malcolm Knapp `work_required` remains open for P11.2.
