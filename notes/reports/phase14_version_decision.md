# Phase 14 next-version decision

Date: 2026-10-30

Phase: P14

Task: P14.3

Status: complete.

## Evidence

- `0.1.0a1` installs cleanly with and without the lidar extra.
- Public docs, indexes, release, and assets are reachable.
- No packaging defect was found.
- Validation gaps are deferred with explicit triggers.
- No critical post-release bug has been reported.

## Decision

**Remain at `0.1.0a1` in maintenance mode.**

Do not prepare `0.1.0b1` or `0.1.0` now.

## Triggers for a new version

- A packaging or installation regression that requires a patch.
- A user-visible bug in the published release.
- Completion of a scoped validation/calibration phase that changes the
  recommendation materially.
- An explicit maintainer request for a stable release.

## Approval boundary

No tag or publication follows from this decision. A new version needs its own
phase, version bump, artifacts, gates, and explicit publication approval.
