# Phase 9 validation report: cross-forest and FHOPS handoff

Date: 2026-10-30

Phase: P9 Cross-forest validation and FHOPS handoff

Status: complete; cross-forest transfer is limited and the FHOPS handoff runs.

## Cross-forest results

### Gavin Lake (P8 empirical)

- Training 15 plots / 2,086 trees; hold-out 5 plots / 784 trees.
- BA error +16.8%, QMD +9.2%, mean DBH -2.4%, mean height +0.6%.
- DBH KS 0.094; species TV 0.143.

### Malcolm Knapp (P9 regional transfer)

- Local hold-out: 6 mature CWHdm plots.
- Regional training: 43 CWHdm plots in the 50–100 km ring.
- Model: empirical; 100 bootstrap replicates.

| Metric | Regional training | Local hold-out | Synthetic | Relative error |
| --- | ---: | ---: | ---: | ---: |
| Stems/ha | 1,476.05 | 1,855.92 | 1,476.00 | -20.5% |
| Basal area (m²/ha) | 48.64 | 52.04 | 48.18 | -7.4% |
| QMD (cm) | 24.79 | 20.87 | 20.39 | -2.3% |
| Mean DBH (cm) | 22.15 | 18.71 | 15.35 | -18.0% |
| Mean height (m) | 20.62 | 20.83 | 15.36 | -26.3% |

BA and QMD transfer reasonably; DBH and height do not. The regional CWHdm pool
is not a substitute for local calibration.

## FHOPS handoff

- Exported 1,978 synthetic trees from the Gavin empirical stand.
- Inventory: `fhops.tree.inventory.v1`, block `B01`, 8 species,
  `not_reviewed`, SHA-256 `0c431c49…ca58`.
- FHOPS loader validation: valid, 1,978 trees, 1 block, 8 species, no errors.
- FHOPS 1.2.0 tree-system run:
  - 1,978 trees, 0 unreachable;
  - 530.54 m³ processed;
  - makespan 415,640.8 s (115.46 h);
  - 106 truck loads / 530.0 m³ dispatched;
  - landing inventory maximum 14.89 m³;
  - warning: processed merchantable volume differs from the fixture
    `work_required` value.

The warning is expected: the copied FHOPS fixture block still has its original
small work requirement. It is a scenario-integration note, not a production
result.

## Cross-forest interpretation

- The empirical model is useful within the Gavin training support.
- A coarse CWHdm regional pool is too broad for Malcolm Knapp DBH/height
  transfer.
- The FHOPS export contract is workable without FHOPS core changes.
- The synthetic tree inventory and system run are reference evidence only.

## Limits

- Malcolm Knapp has six local hold-out plots; the regional ring is a different
  population.
- The exported coordinates are deterministic synthetic placement, not a tree
  map.
- The FHOPS scenario uses fixture machine rates and a mismatched work
  requirement.
- No operational, financial, or regulatory claim is supported.

## Reproduce

```bash
python scripts/validate_phase9_mkrf.py \
  --faib-root data/external/faib/raw \
  --report notes/reports/phase9_mkrf_validation.json

python scripts/export_phase9_fhops.py \
  --faib-root data/external/faib/raw \
  --report notes/reports/phase9_fhops_export.json

fhops tree system build/phase9/fhops/scenario/scenario.yaml \
  --block B01 --feller H1 --skidder H2 --skidder H3 \
  --processor H5 --loader H4 \
  --truck-capacity 5 --truck-count 1 --landing-capacity 15 \
  --out build/phase9/fhops/system
```
