# Phase 14 release verification

Date: 2026-10-30

Phase: P14

Task: P14.1

Status: complete.

## Release under review

`0.1.0a1`.

## Installs

Base PyPI install (from P13 verification):
- `nemora==0.1.0a1`
- import/metadata version `0.1.0a1`
- `nemora --version` = `nemora 0.1.0a1`
- `pip check` clean

Lidar-extra install:
- `nemora[lidar]==0.1.0a1`
- import/metadata version `0.1.0a1`
- rasterio `1.5.2`
- `nemora --version` = `nemora 0.1.0a1`
- `pip check` clean

## Public surfaces

| Surface | Result |
| --- | --- |
| PyPI JSON | 200 |
| TestPyPI JSON | 200 |
| GitHub Pages docs | 200 |
| Release notes page | 200 |
| GitHub release | present, target `main` |
| Release assets | wheel + sdist |

Downloaded release assets match PyPI hashes:

- Wheel: `67034381f2cbfe3d10faeaaef55046c075332c785cf1819e6cc2399d83db2def`
- Sdist: `81d3277104c1f7cdbe7c52b0f93935ff0257d18c087fbb4f9ebbe16820576543`

## Defects

None found in installation, extras, docs, or assets.

## Limits

- The `typer` extra warning is from the dependency, not the package install.
- No operational or validation claim is implied by a clean install.
