# P5: Uncertainty, sampling, and synthetic validation

Parent issue: `#7`
Branch: `feature/p5-uncertainty-sampling`
Status: planned (2026-10-25)

## Purpose

Make Nemora the place where FRESH tests uncertainty, sampling design, synthetic
landscapes, and inventory ingest. The phase turns the existing fitting,
sampling, synthesis, and ingest modules into cross-repo validation surfaces.

## Children

- `P5.1` (`#6`): curve uncertainty and a 90% band for one family.
- `P5.2` (`#8`): synthetic landscape fixture with known properties.
- `P5.3` (`#9`): shared inventory ingest schema and one migrated path.
- `P5.4` (`#10`): sgsR structurally guided sampling wrapper.

## Out of scope

- A full measurement-campaign simulator (`DATA-03`, wave 2).
- A general SDM pipeline (`BIO-03`, wave 3).
- A new distribution-fitting theory or a Bayesian platform.
- Changing a downstream model objective.

## Acceptance

- The curve band is emitted and its coverage check is recorded.
- The synthetic fixture is deterministic and runs through ws4 build/simulate.
- The ingest schema rejects a missing key and reproduces the existing output.
- The sampling wrapper produces the same allocation for the same inputs.
- Each child has a test that can fail and a docs note.

## Risks

- Synthetic realism is not evidence about a real forest.
- The sgsR wrapper may add an R/Python bridge dependency.
- Upstream FAIB/FIA/HPS schemas drift; the nightly ingest is the early warning.
- Scope creep into the simulation module.
