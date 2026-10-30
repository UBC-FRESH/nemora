# Phase 10 FHOPS scenario registration

Date: 2026-10-30

Phase: P10

Task: P10.2

Status: complete.

## Registration

- Template: FHOPS tree-system fixture scenario.
- Inventory: Phase 9 export, 1,978 trees, block `B01`.
- Pin: inventory SHA-256 and metadata SHA-256 written by
  `nemora.fhops_handoff.register_fhops_scenario`.
- Pin check: `validate_fhops_scenario_pin` passes.
- Block reconciliation: `work_required` and `volume_per_ha_m3` set to
  **530.543340 m³**, the Phase 9 processed merchantable volume.
- FHOPS scenario validation passes.
- FHOPS system run now has **no work_required warning**.

## Reproduce

```bash
python scripts/register_phase10_fhops.py \
  --scenario-template path/to/fhops/examples/tree_system/scenario \
  --inventory-dir build/phase9/fhops \
  --output-dir build/phase10/fhops/scenario \
  --work-required-m3 530.5433400362283 \
  --report notes/reports/phase10_fhops_scenario_registration.json
```

## Limits

- The scenario is still a fixture; machine rates are not operationally calibrated.
- The work_required reconciliation uses the processed volume from the same
  inventory, so it removes the integration warning but is not an independent
  yield estimate.
- The inventory is `not_reviewed` and synthetic.
