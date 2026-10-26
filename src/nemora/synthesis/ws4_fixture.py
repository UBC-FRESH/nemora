"""Deterministic synthetic landscape exported for the WS4 build/simulate path.

The fixture is intentionally small and homogeneous: Voronoi fragments, three
synthetic age classes, one species, and placeholder tree structure. It is an
integration fixture, not a calibrated forest or a yield-model claim.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from . import exporters, stands, stems, tessellation
from .helpers import StandDBHSampler

FIXTURE_SCHEMA_VERSION = "nemora.ws4_synthetic.v1"
FIXTURE_MANIFEST = "fixture.json"
TRUTH_TABLE = "truth_table.csv"

AGE_CLASSES: tuple[int, ...] = (40, 60, 80)
VOLUME_CURVE_POINTS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (40.0, 120.0),
    (80.0, 220.0),
)
LOGNORMAL_PARAMETERS: dict[int, dict[str, float]] = {
    40: {"mu": 2.5, "sigma2": 0.16},
    60: {"mu": 2.9, "sigma2": 0.18},
    80: {"mu": 3.2, "sigma2": 0.20},
}
DEFAULT_COUNT = 12
DEFAULT_ASPECT_RATIO = 1.5
DEFAULT_AREA_HA = 36.0
DEFAULT_TREES_PER_HA = 200.0

WS4_RECIPE_XML = textwrap.dedent(
    """\
    <?xml version="1.0" encoding="UTF-8"?>
    <!-- Synthetic WS4 recipe: one natural curve, one managed curve, and a
         CC treatment. The fixture's expected volume is the natural curve at
         each fragment's age. -->
    <ForestModel horizon="120" year="0" match="single">
      <curve id="natural_volume">
        <point x="0" y="0"/>
        <point x="40" y="120"/>
        <point x="80" y="220"/>
      </curve>
      <curve id="managed_volume">
        <point x="0" y="0"/>
        <point x="10" y="50"/>
        <point x="40" y="180"/>
      </curve>
      <define field="AU" column="AU"/>
      <define field="IFM" column="IFM"/>
      <define field="treatment"/>
      <input block="BLOCK" area="AREA_HA" age="F_AGE"/>
      <output messages="messages.csv" blocks="blocks.csv" features="features.csv"
              products="products.csv" treatments="treatments.csv"
              curves="curves.csv" tracknames="tracknames.csv"/>
      <select statement="IFM eq 'unmanaged'">
        <features>
          <attribute label="feature.Volume">
            <curve idref="natural_volume"/>
          </attribute>
        </features>
        <track>
          <treatment label="CC" minage="40" maxage="120">
            <produce>
              <assign field="treatment" value="'CC'"/>
            </produce>
            <transition>
              <assign field="IFM" value="'managed'"/>
            </transition>
          </treatment>
        </track>
      </select>
      <select statement="IFM eq 'managed'">
        <features>
          <attribute label="feature.Volume">
            <curve idref="managed_volume"/>
          </attribute>
        </features>
        <track/>
      </select>
      <select statement="IFM eq 'unmanaged' and treatment eq 'CC'">
        <products>
          <attribute label="product.Volume">
            <curve idref="natural_volume"/>
          </attribute>
        </products>
      </select>
    </ForestModel>
    """
)

TRUTH_COLUMNS: tuple[str, ...] = (
    "fragment_id",
    "block_id",
    "area_ha",
    "age",
    "volume_per_ha",
    "expected_volume_m3",
    "trees",
    "trees_per_ha",
    "mean_dbh_cm",
    "mean_height_m",
    "basal_area_m2_per_ha",
    "polygon_area_fraction",
)


@dataclass(slots=True)
class Ws4SyntheticFixture:
    """Loaded fixture metadata and truth table."""

    directory: Path
    seed: int
    truth_table: pd.DataFrame
    metadata: dict[str, Any]
    recipe_path: Path
    fragments_path: Path
    schedule_path: Path


def curve_value(age: float) -> float:
    """Interpolate the synthetic natural-volume curve at ``age``."""
    points = VOLUME_CURVE_POINTS
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=True):
        if x0 <= age <= x1:
            if x1 == x0:
                return y1
            return y0 + (y1 - y0) * (age - x0) / (x1 - x0)
    raise ValueError(f"age {age!r} is outside the fixture curve {points[0][0]}-{points[-1][0]}")


def _polygon_area(polygon: np.ndarray) -> float:
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _tree_summary(records: list[dict[str, object]], area_ha: float) -> dict[str, float]:
    dbh_values = [float(cast(float, record["dbh"])) for record in records]
    attributes = [cast(stems.TreeAttributes, record["attributes"]) for record in records]
    heights = [attribute.height_m for attribute in attributes]
    basal_areas = [attribute.basal_area_m2 for attribute in attributes]
    tree_count = len(records)
    return {
        "trees": float(tree_count),
        "trees_per_ha": float(tree_count / area_ha) if area_ha > 0 else 0.0,
        "mean_dbh_cm": float(sum(dbh_values) / tree_count) if tree_count else 0.0,
        "mean_height_m": float(sum(heights) / tree_count) if tree_count else 0.0,
        "basal_area_m2_per_ha": (float(sum(basal_areas) / area_ha) if area_ha > 0 else 0.0),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _round_json(value: Any, digits: int) -> Any:
    """Round floats recursively so exported GeoJSON is cross-platform stable."""
    if isinstance(value, float | np.floating):
        return round(float(value), digits)
    if isinstance(value, dict):
        return {str(key): _round_json(item, digits) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_round_json(item, digits) for item in value]
    return value


def generate_ws4_fixture(
    output_dir: str | Path,
    *,
    seed: int,
    count: int = DEFAULT_COUNT,
    aspect_ratio: float = DEFAULT_ASPECT_RATIO,
    area_ha: float = DEFAULT_AREA_HA,
    trees_per_ha: float = DEFAULT_TREES_PER_HA,
) -> Ws4SyntheticFixture:
    """Generate a deterministic synthetic landscape and WS4 build inputs.

    Parameters
    ----------
    output_dir:
        Directory to write the fixture files into.
    seed:
        Required random seed. The same seed and parameters reproduce the same
        fragment geometry, tree summaries, truth table, and fixture metadata.
    count:
        Number of Voronoi fragments.
    aspect_ratio:
        Landscape width divided by height in local coordinates.
    area_ha:
        Total landscape area in hectares. Polygon coordinate areas are scaled
        to this total.
    trees_per_ha:
        Target stem density used to size the placeholder tree list.
    """
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if count < len(AGE_CLASSES):
        raise ValueError(f"count must be >= {len(AGE_CLASSES)}")
    if aspect_ratio <= 0 or area_ha <= 0 or trees_per_ha <= 0:
        raise ValueError("aspect_ratio, area_ha, and trees_per_ha must be positive")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    seed_result = tessellation.generate_seed_points(
        tessellation.VoronoiSeedConfig(
            count=count,
            aspect_ratio=aspect_ratio,
            mix=tessellation.PointProcessMix(uniform=1.0),
            rng=np.random.default_rng(seed),
        )
    )
    polygons = [polygon for polygon in seed_result.polygons if polygon.size > 0]
    if len(polygons) != count:
        raise ValueError(f"expected {count} polygons, got {len(polygons)}")

    raw_areas = np.asarray([_polygon_area(polygon) for polygon in polygons], dtype=float)
    if float(raw_areas.sum()) <= 0:
        raise ValueError("generated polygons have zero total area")
    area_fractions = raw_areas / float(raw_areas.sum())
    fragment_areas = np.round(area_fractions * float(area_ha), 6)
    fragment_areas[-1] = round(float(area_ha) - float(fragment_areas[:-1].sum()), 6)

    ages = [AGE_CLASSES[index % len(AGE_CLASSES)] for index in range(count)]
    samples = [
        stands.StandAttributeSample("fir", str(age), float(fragment_areas[index]))
        for index, age in enumerate(ages)
    ]
    features = stands.build_stand_features(polygons, samples)

    truth_rows: list[dict[str, object]] = []
    tree_config = stems.TreePlacementConfig(mode="poisson", min_spacing=0.0)
    attribute_config = stems.TreeAttributeConfig(
        height_a=1.0,
        height_b=0.8,
        provenance="nemora.ws4_synthetic.v1",
    )

    for index, (polygon, age, fragment_area) in enumerate(
        zip(polygons, ages, fragment_areas, strict=True)
    ):
        tree_count = max(1, int(round(trees_per_ha * float(fragment_area))))
        entry = stands.StandBootstrapLibraryEntry(
            identifier=f"age-{age}",
            source="analytic",
            metadata={
                "distribution": "lognormal",
                "parameters": dict(LOGNORMAL_PARAMETERS[age]),
                "sample_size": tree_count,
                "mode": "analytic",
                "age": age,
            },
            dbh_vectors={},
        )
        assignment = stands.StandBootstrapAssignment(
            stand_id=f"stand-{index + 1:04d}",
            vegetation_type="fir",
            age_class=str(age),
            area=float(fragment_area),
            bootstrap_id=f"age-{age}",
        )
        sampler = StandDBHSampler(assignment=assignment, entry=entry)
        records = stems.place_trees_with_dbh(
            polygon,
            sampler,
            count=tree_count,
            rng=np.random.default_rng(seed + index + 1),
            config=tree_config,
        )
        records = stems.attach_tree_attributes(records, config=attribute_config)
        summary = _tree_summary(records, float(fragment_area))
        volume_per_ha = curve_value(age)
        expected_volume = volume_per_ha * float(fragment_area)
        truth_rows.append(
            {
                "fragment_id": index + 1,
                "block_id": index + 1,
                "area_ha": float(fragment_area),
                "age": age,
                "volume_per_ha": volume_per_ha,
                "expected_volume_m3": expected_volume,
                "trees": int(summary["trees"]),
                "trees_per_ha": summary["trees_per_ha"],
                "mean_dbh_cm": summary["mean_dbh_cm"],
                "mean_height_m": summary["mean_height_m"],
                "basal_area_m2_per_ha": summary["basal_area_m2_per_ha"],
                "polygon_area_fraction": float(area_fractions[index]),
            }
        )

        properties = cast(dict[str, object], features[index]["properties"])
        properties.update(
            {
                "fragment_id": index + 1,
                "block_id": index + 1,
                "stand_id": assignment.stand_id,
                "area_ha": float(fragment_area),
                "age": age,
                "volume_per_ha": volume_per_ha,
                "expected_volume_m3": expected_volume,
                "trees": int(summary["trees"]),
                "mean_dbh_cm": summary["mean_dbh_cm"],
                "mean_height_m": summary["mean_height_m"],
                "basal_area_m2_per_ha": summary["basal_area_m2_per_ha"],
            }
        )

    truth = pd.DataFrame.from_records(truth_rows, columns=list(TRUTH_COLUMNS))
    total_area = float(truth["area_ha"].sum())
    total_volume = float(truth["expected_volume_m3"].sum())
    total_trees = int(truth["trees"].sum())

    fragments = pd.DataFrame(
        {
            "BLOCK": truth["block_id"].astype(int),
            "AREA_HA": truth["area_ha"].astype(float),
            "F_AGE": truth["age"].astype(int),
            "AU": 1,
            "IFM": "unmanaged",
        }
    )
    fragments.to_csv(
        target / "fragments.csv",
        index=False,
        float_format="%.6f",
        lineterminator="\n",
    )
    schedule = pd.DataFrame(columns=["block_id", "treatment", "offset"])
    schedule.to_csv(target / "schedule.csv", index=False, lineterminator="\n")
    truth.to_csv(
        target / TRUTH_TABLE,
        index=False,
        float_format="%.6f",
        lineterminator="\n",
    )
    (target / "recipe.xml").write_text(WS4_RECIPE_XML, encoding="utf-8")
    rounded_features = [
        {
            "type": feature["type"],
            "properties": _round_json(feature["properties"], 6),
            "geometry": _round_json(feature["geometry"], 10),
        }
        for feature in features
    ]
    exporters.export_geojson(rounded_features, target / "landscape.geojson", crs="local")

    metadata: dict[str, Any] = {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "seed": seed,
        "generator": {
            "module": "nemora.synthesis.ws4_fixture",
            "fixture": "synthetic WS4 build/simulate input",
        },
        "landscape": {
            "count": count,
            "aspect_ratio": aspect_ratio,
            "area_ha": round(total_area, 6),
            "trees_per_ha": trees_per_ha,
        },
        "volume_curve": {
            "name": "natural_volume",
            "points": [[x, y] for x, y in VOLUME_CURVE_POINTS],
        },
        "inputs": {
            "recipe": "recipe.xml",
            "fragments": "fragments.csv",
            "schedule": "schedule.csv",
        },
        "truth_table": TRUTH_TABLE,
        "landscape_features": "landscape.geojson",
        "totals": {
            "area_ha": round(total_area, 6),
            "expected_volume_m3": round(total_volume, 6),
            "trees": total_trees,
        },
        "structure": {
            "provenance": attribute_config.provenance,
            "note": (
                "Placeholder allometry from nemora.synthesis.stems; "
                "the fixture tests integration, not taper or yield realism."
            ),
        },
        "limits": [
            "One species (fir), three age classes, and no mortality or treatment schedule.",
            "Placeholder tree allometry is not calibrated to a real stand.",
            "Volume is the synthetic natural-volume curve, not a local yield estimate.",
            "The fixture is not a calibrated synthetic forest or a real landscape.",
        ],
    }
    _write_json(target / FIXTURE_MANIFEST, metadata)
    return load_ws4_fixture(target)


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"fixture.json field {key!r} must be an object")
    return value


def load_ws4_fixture(directory: str | Path) -> Ws4SyntheticFixture:
    """Load and validate a fixture directory, including its truth table."""
    target = Path(directory)
    manifest_path = target / FIXTURE_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing fixture manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("fixture.json must contain an object")
    if payload.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise ValueError(
            f"fixture schema_version must be {FIXTURE_SCHEMA_VERSION!r}; "
            f"got {payload.get('schema_version')!r}"
        )
    if "seed" not in payload or not isinstance(payload["seed"], int):
        raise ValueError("fixture.json must contain an integer seed")
    truth_name = payload.get("truth_table")
    if not isinstance(truth_name, str) or not truth_name:
        raise ValueError("fixture.json must name its truth_table")
    truth_path = target / truth_name
    if not truth_path.is_file():
        raise FileNotFoundError(f"missing fixture truth table: {truth_path}")
    truth = pd.read_csv(truth_path)
    missing = sorted(set(TRUTH_COLUMNS) - set(truth.columns))
    if missing:
        raise ValueError(f"truth table is missing columns: {', '.join(missing)}")
    inputs = _require_mapping(payload, "inputs")
    recipe_path = target / str(inputs.get("recipe", "recipe.xml"))
    fragments_path = target / str(inputs.get("fragments", "fragments.csv"))
    schedule_path = target / str(inputs.get("schedule", "schedule.csv"))
    for path, label in (
        (recipe_path, "recipe"),
        (fragments_path, "fragments"),
        (schedule_path, "schedule"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"missing fixture {label}: {path}")
    totals = _require_mapping(payload, "totals")
    expected_total = float(totals.get("expected_volume_m3", float("nan")))
    actual_total = float(truth["expected_volume_m3"].sum())
    if not np.isclose(expected_total, actual_total, rtol=0.0, atol=1e-6):
        raise ValueError(
            "fixture total expected volume does not match the truth table: "
            f"{expected_total} != {actual_total}"
        )
    return Ws4SyntheticFixture(
        directory=target,
        seed=int(payload["seed"]),
        truth_table=truth,
        metadata=payload,
        recipe_path=recipe_path,
        fragments_path=fragments_path,
        schedule_path=schedule_path,
    )


__all__ = [
    "AGE_CLASSES",
    "DEFAULT_AREA_HA",
    "DEFAULT_ASPECT_RATIO",
    "DEFAULT_COUNT",
    "DEFAULT_TREES_PER_HA",
    "FIXTURE_MANIFEST",
    "FIXTURE_SCHEMA_VERSION",
    "LOGNORMAL_PARAMETERS",
    "TRUTH_COLUMNS",
    "TRUTH_TABLE",
    "VOLUME_CURVE_POINTS",
    "WS4_RECIPE_XML",
    "Ws4SyntheticFixture",
    "curve_value",
    "generate_ws4_fixture",
    "load_ws4_fixture",
]
