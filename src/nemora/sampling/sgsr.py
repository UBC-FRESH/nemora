"""Python bridge to the GPL-3 ``sgsR`` structurally guided sampling package.

sgsR is not copied or reimplemented here. The bridge calls the installed R
package through a small bundled script and returns the allocation, samples, and
structural strata as DataFrames.

The R bridge is optional: install R plus ``sgsR`` and its dependencies before
using :func:`run_sgsr`.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

__all__ = ["SgsRError", "SgsRResult", "run_sgsr"]

_BRIDGE = Path(__file__).with_name("sgsr_bridge.R")
_ALLOCATIONS = frozenset({"prop", "equal", "optim"})
_METHODS = frozenset({"Queinnec", "random"})


class SgsRError(RuntimeError):
    """Raised when the R/sgsR bridge cannot produce a sampling result."""


@dataclass(slots=True)
class SgsRResult:
    """Portable outputs from one sgsR bridge run."""

    output_dir: Path
    allocation: pd.DataFrame
    samples: pd.DataFrame
    strata: pd.DataFrame
    metadata: dict[str, str]
    strata_path: Path
    samples_path: Path
    allocation_path: Path

    @property
    def sample_count(self) -> int:
        """Number of returned samples."""
        return int(len(self.samples))


def _read_metadata(path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, _, value = line.partition("\t")
        metadata[key.strip()] = value.strip()
    return metadata


def _check_table(path: Path, label: str, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise SgsRError(f"sgsR bridge did not write {label}: {path}")
    frame = pd.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SgsRError(f"sgsR {label} is missing columns: {', '.join(missing)}")
    return frame


def run_sgsr(
    metric_path: str | Path,
    output_dir: str | Path,
    *,
    n_samples: int,
    seed: int,
    n_strata: int = 4,
    allocation: str = "prop",
    method: str = "Queinnec",
    mindist: float | None = None,
    rscript: str | Path = "Rscript",
    timeout: float = 600.0,
    env: Mapping[str, str] | None = None,
) -> SgsRResult:
    """Run sgsR quantile stratification and stratified sampling.

    Parameters
    ----------
    metric_path:
        Single-layer structural metric raster readable by ``terra`` (for
        example a GeoTIFF or ESRI ASCII grid).
    output_dir:
        Directory for the R bridge artifacts: ``strata.tif``,
        ``allocation.csv``, ``samples.csv``, ``strata.csv``, and
        ``metadata.tsv``.
    n_samples:
        Target plot count.
    seed:
        Required R RNG seed. The same metric and parameters must reproduce the
        same allocation and sample coordinates.
    n_strata:
        Number of quantile strata.
    allocation:
        ``prop``, ``equal``, or ``optim``.
    method:
        ``Queinnec`` or ``random``.
    mindist:
        Optional minimum distance between samples, in the raster CRS units.
    rscript:
        Path or command name for ``Rscript``.
    timeout:
        Subprocess timeout in seconds.
    env:
        Optional environment overrides. Use this to set ``R_LIBS_USER`` when
        sgsR is installed in a non-default R library.

    Returns
    -------
    SgsRResult
        Allocation table, sample coordinates, structural strata, and bridge
        metadata.

    Raises
    ------
    SgsRError
        If R/sgsR is unavailable, the input is invalid, the metric is constant,
        or the bridge fails.
    """
    metric = Path(metric_path)
    target = Path(output_dir)
    if not metric.is_file():
        raise SgsRError(f"metric raster does not exist: {metric}")
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if n_strata < 2:
        raise ValueError("n_strata must be >= 2")
    if allocation not in _ALLOCATIONS:
        raise ValueError(f"allocation must be one of: {', '.join(sorted(_ALLOCATIONS))}")
    if method not in _METHODS:
        raise ValueError(f"method must be one of: {', '.join(sorted(_METHODS))}")
    if mindist is not None and mindist < 0:
        raise ValueError("mindist must be non-negative")
    if not _BRIDGE.is_file():
        raise SgsRError(f"bundled sgsR bridge is missing: {_BRIDGE}")

    target.mkdir(parents=True, exist_ok=True)
    command = [
        str(rscript),
        str(_BRIDGE),
        "--metric",
        str(metric),
        "--out",
        str(target),
        "--n-samples",
        str(n_samples),
        "--n-strata",
        str(n_strata),
        "--allocation",
        allocation,
        "--method",
        method,
        "--seed",
        str(seed),
    ]
    if mindist is not None:
        command.extend(["--mindist", str(mindist)])

    run_env = os.environ.copy()
    if env is not None:
        run_env.update({str(key): str(value) for key, value in env.items()})

    # Only the configured Rscript is executed.
    try:
        completed = subprocess.run(  # noqa: S603
            command,  # noqa: S603
            capture_output=True,
            text=True,
            timeout=timeout,
            env=run_env,
            check=False,
        )
    except FileNotFoundError as exc:
        raise SgsRError(
            f"Rscript not found ({rscript!r}). Install R and sgsR before using this bridge."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise SgsRError(f"sgsR bridge timed out after {timeout} seconds") from exc

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        if len(detail) > 4000:
            detail = detail[-4000:]
        raise SgsRError(
            "sgsR bridge failed. Ensure Rscript, sgsR, terra, and sf are installed.\n"
            f"exit code: {completed.returncode}\n{detail}"
        )

    allocation_frame = _check_table(target / "allocation.csv", "allocation", {"strata", "total"})
    samples_frame = _check_table(
        target / "samples.csv", "samples", {"x", "y", "strata", "type", "rule"}
    )
    strata_frame = _check_table(
        target / "strata.csv", "strata", {"strata", "cells", "min", "max", "mean"}
    )
    metadata_path = target / "metadata.tsv"
    if not metadata_path.is_file():
        raise SgsRError(f"sgsR bridge did not write metadata: {metadata_path}")
    metadata = _read_metadata(metadata_path)

    if allocation_frame.empty:
        raise SgsRError("sgsR returned an empty allocation table")
    if strata_frame.empty:
        raise SgsRError("sgsR returned no structural strata")
    if samples_frame.empty:
        raise SgsRError("sgsR returned no samples")
    strata_path = target / "strata.tif"
    if not strata_path.is_file():
        raise SgsRError(f"sgsR did not write the structural strata raster: {strata_path}")
    return SgsRResult(
        output_dir=target,
        allocation=allocation_frame,
        samples=samples_frame,
        strata=strata_frame,
        metadata=metadata,
        strata_path=strata_path,
        samples_path=target / "samples.csv",
        allocation_path=target / "allocation.csv",
    )
