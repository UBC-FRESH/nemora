# FHOPS tree-inventory handoff

Nemora can export a synthetic stand to the FHOPS `fhops.tree.inventory.v1`
contract and register a FHOPS scenario against it. FHOPS remains the consumer;
Nemora does not import FHOPS.

## Export

```python
from nemora.export import export_fhops_tree_inventory

export = export_fhops_tree_inventory(
    synthetic_trees,
    "build/handoff/inventory",
    block_id="B01",
    inventory_id="nemora-example-stand",
)
```

The export writes `tree_inventory.csv` and `tree_inventory.meta.json`, including
the CSV SHA-256, tree count, units, CRS, review status, and caveats. Coordinates
are deterministic synthetic placement unless the source tree list already has a
spatial contract.

## Register a scenario

```python
from nemora.fhops_handoff import register_fhops_scenario, validate_fhops_scenario_pin

registration = register_fhops_scenario(
    "path/to/fhops/scenario-template",
    "build/handoff/scenario",
    "build/handoff/inventory",
    block_id="B01",
    work_required_m3=530.543340,
)
assert validate_fhops_scenario_pin(registration.scenario_path)
```

Registration copies the scenario template, writes the tree-inventory pin and
checksums, and optionally reconciles the block `work_required` and
`volume_per_ha_m3`. It fails if the inventory block is not declared or the
metadata checksum does not match.

## Independent yield check

```python
from nemora.yield_check import estimate_inventory_volume

estimate = estimate_inventory_volume(tree_table)
print(estimate.total_volume_m3)
```

The estimator is a paraboloid-style independent check, not a taper model. It
uses DBH, total height, a 70% merchantable-height ratio, and a 0.5 form factor.
Report the difference from FHOPS processed volume rather than forcing a match.

## Run FHOPS

```bash
fhops tree system build/handoff/scenario/scenario.yaml \
  --block B01 --feller H1 --skidder H2 --skidder H3 \
  --processor H5 --loader H4 --truck-capacity 5 --truck-count 1 \
  --landing-capacity 15 --out build/handoff/system
```

The ground system is one block/one landing per run. A multi-block set is a
catalog of separate scenarios, not a simultaneous multi-block optimizer.

## Limits

- Synthetic inventories are not surveyed stands or tree maps.
- The independent volume estimate is a geometric check, not a taper model.
- Machine rates are reference/fixture values, not field-calibrated.
- No operational, financial, or regulatory claim is made.
