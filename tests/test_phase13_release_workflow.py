"""Checks for the release workflow contract."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_release_workflow_has_separate_trusted_publish_jobs() -> None:
    payload = yaml.safe_load(Path(".github/workflows/release.yaml").read_text(encoding="utf-8"))
    trigger = payload.get("on", payload.get(True))

    assert trigger["push"]["tags"] == ["v*"]
    jobs = payload["jobs"]
    assert "build" in jobs
    assert jobs["publish-testpypi"]["environment"] == "testpypi"
    assert jobs["publish-pypi"]["environment"] == "pypi"
    assert jobs["publish-testpypi"]["needs"] == "build"
    assert jobs["publish-pypi"]["needs"] == "publish-testpypi"
    assert "secrets" not in str(payload)
