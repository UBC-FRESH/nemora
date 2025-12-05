"""Stand-attribute templates inspired by the FLG workflow."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

__all__ = [
    "StandAttributeTemplate",
    "StandAttributeSample",
    "StandBootstrapAssignment",
    "StandBootstrapLibraryEntry",
    "StandBootstrapManifest",
    "StandBootstrapPlan",
    "StandBootstrapRule",
    "build_bootstrap_assignments",
    "build_stand_features",
    "build_templates",
    "load_bootstrap_manifest",
    "load_bootstrap_plan",
    "load_samples_from_json",
    "load_templates_from_json",
    "sample_stand_attributes",
]


@dataclass(slots=True)
class StandAttributeTemplate:
    """Reusable patch recipe derived from vegetation/age-class tables."""

    vegetation_type: str
    age_class_cdf: Sequence[tuple[str, float]]
    patch_weibull: tuple[float, float, float]

    def sample_age_class(self, rng: np.random.Generator | None = None) -> str:
        """Sample an age-class label using the stored cumulative probabilities."""

        rng = rng or np.random.default_rng()
        draw = float(rng.random())
        for label, cumulative in self.age_class_cdf:
            if draw <= cumulative:
                return label
        return self.age_class_cdf[-1][0]

    def sample_patch_size(
        self,
        rng: np.random.Generator | None = None,
        size: int = 1,
    ) -> np.ndarray:
        """Sample patch areas via a shifted/scaled Weibull distribution."""

        rng = rng or np.random.default_rng()
        shape, scale, shift = self.patch_weibull
        samples = rng.weibull(shape, size=size) * scale + shift
        return samples


@dataclass(slots=True)
class StandAttributeSample:
    """Concrete stand attribute sampled from a template."""

    vegetation_type: str
    age_class: str
    area: float


def sample_stand_attributes(
    templates: Sequence[StandAttributeTemplate],
    *,
    total_area: float,
    rng: np.random.Generator | None = None,
    weights: Sequence[float] | None = None,
) -> list[StandAttributeSample]:
    """Fill ``total_area`` with sampled stand attributes."""

    if total_area <= 0:
        raise ValueError("total_area must be positive.")
    if not templates:
        raise ValueError("At least one template is required.")
    if weights is not None and len(weights) != len(templates):
        raise ValueError("Length of weights must match the length of templates.")
    rng = rng or np.random.default_rng()
    if weights is not None:
        probs = np.asarray(weights, dtype=float)
        if np.any(probs < 0):
            raise ValueError("Weights must be non-negative.")
        if probs.sum() == 0:
            raise ValueError("At least one weight must be positive.")
        probs = probs / probs.sum()
    else:
        probs = None
    remaining = float(total_area)
    samples: list[StandAttributeSample] = []
    iterations = 0
    max_iterations = 10000
    while remaining > 1e-6 and iterations < max_iterations:
        if probs is None:
            index = int(rng.integers(0, len(templates)))
        else:
            index = int(rng.choice(len(templates), p=probs))
        template = templates[index]
        patch_area = float(template.sample_patch_size(rng, size=1)[0])
        if patch_area <= 0:
            iterations += 1
            continue
        age_class = template.sample_age_class(rng)
        area = min(patch_area, remaining)
        samples.append(
            StandAttributeSample(
                vegetation_type=template.vegetation_type,
                age_class=age_class,
                area=area,
            )
        )
        remaining -= area
        iterations += 1
    return samples


def build_templates(
    records: Iterable[Mapping[str, object]],
) -> list[StandAttributeTemplate]:
    """Convert raw vegetation-type tables into reusable templates.

    Each *record* is expected to expose:

    - ``vegetation_type``: identifier string
    - ``age_classes``: sequence of ``(label, cumulative_probability)``
    - ``patch_weibull``: ``(shape, scale, shift)`` tuple describing patch areas
    """

    templates: list[StandAttributeTemplate] = []
    for record in records:
        veg = str(record["vegetation_type"])
        raw_age_classes = cast(Sequence[tuple[Any, Any]], record["age_classes"])
        age_classes = tuple(
            (str(label), float(cumulative)) for label, cumulative in raw_age_classes
        )
        if not age_classes or age_classes[-1][1] < 1.0:
            raise ValueError(
                f"Age-class CDF for vegetation type '{veg}' must end at probability 1.0."
            )
        weibull_seq = tuple(float(value) for value in cast(Sequence[Any], record["patch_weibull"]))
        if len(weibull_seq) != 3:
            raise ValueError("patch_weibull must contain (shape, scale, shift).")
        weibull_params = cast(tuple[float, float, float], tuple(weibull_seq))
        templates.append(
            StandAttributeTemplate(
                vegetation_type=veg,
                age_class_cdf=age_classes,
                patch_weibull=weibull_params,  # type: ignore[arg-type]
            )
        )
    return templates


def load_templates_from_json(path: Path) -> list[StandAttributeTemplate]:
    """Load stand templates from a JSON file (list of record dictionaries)."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Template JSON must be a list of records.")
    return build_templates(cast(Iterable[Mapping[str, object]], payload))


def load_samples_from_json(path: Path) -> list[StandAttributeSample]:
    """Load stand attribute samples (vegetation/age/area) from JSON."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Stand attributes JSON must be a list of records.")
    samples: list[StandAttributeSample] = []
    for record in payload:
        if not isinstance(record, Mapping):
            raise ValueError("Each stand attribute entry must be a mapping.")
        vegetation_type = str(record.get("vegetation_type", ""))
        age_class = str(record.get("age_class", ""))
        area = float(record.get("area", 0.0))
        samples.append(
            StandAttributeSample(
                vegetation_type=vegetation_type,
                age_class=age_class,
                area=area,
            )
        )
    return samples


def build_stand_features(
    polygons: Sequence[np.ndarray],
    samples: Sequence[StandAttributeSample],
    *,
    assignments: Sequence[StandBootstrapAssignment] | None = None,
    bootstrap_library: Mapping[str, StandBootstrapLibraryEntry] | None = None,
) -> list[dict[str, object]]:
    """Pair Voronoi polygons with sampled attributes and return GeoJSON features."""

    valid_polygons = [poly for poly in polygons if poly.size > 0]
    if not valid_polygons or not samples:
        return []
    count = min(len(valid_polygons), len(samples))
    if assignments is not None and len(assignments) < count:
        raise ValueError("Not enough bootstrap assignments to match the available stand samples.")
    features: list[dict[str, object]] = []
    for idx in range(count):
        polygon = valid_polygons[idx]
        sample = samples[idx]
        area = _polygon_area(polygon)
        assignment = assignments[idx] if assignments is not None else None
        entry = None
        if assignment is not None and bootstrap_library is not None:
            entry = bootstrap_library.get(assignment.bootstrap_id)
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "veg_type": sample.vegetation_type,
                    "age_class": sample.age_class,
                    "area_template": sample.area,
                    "polygon_area": area,
                    **(
                        {
                            "stand_id": assignment.stand_id,
                            "bootstrap_id": assignment.bootstrap_id,
                            "bootstrap_metadata": _bootstrap_feature_metadata(entry),
                        }
                        if assignment is not None
                        else {}
                    ),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [polygon.tolist()],
                },
            }
        )
    return features


def _polygon_area(polygon: np.ndarray) -> float:
    if polygon.shape[0] < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _bootstrap_feature_metadata(
    entry: StandBootstrapLibraryEntry | None,
) -> dict[str, object] | None:
    if entry is None:
        return None
    metadata = entry.metadata
    subset: dict[str, object] = {}
    for key in ("distribution", "resamples", "sample_size"):
        value = metadata.get(key)
        if value is not None:
            subset[key] = value
    params = metadata.get("parameters")
    if isinstance(params, Mapping):
        subset["parameters"] = dict(params)
    subset["source"] = entry.source
    return subset


@dataclass(slots=True)
class StandBootstrapRule:
    """Rule describing how to attach a bootstrap payload to a stand sample."""

    identifier: str
    bootstrap_path: Path
    source: str
    vegetation_type: str | None = None
    age_class: str | None = None

    def matches(self, sample: StandAttributeSample) -> bool:
        if self.vegetation_type is not None and sample.vegetation_type != self.vegetation_type:
            return False
        if self.age_class is not None and sample.age_class != self.age_class:
            return False
        return True


@dataclass(slots=True)
class StandBootstrapPlan:
    """Parsed bootstrap plan describing stand-to-payload rules."""

    rules: Sequence[StandBootstrapRule]
    default_rule: StandBootstrapRule | None = None

    def select_rule(self, sample: StandAttributeSample) -> StandBootstrapRule:
        for rule in self.rules:
            if rule.matches(sample):
                return rule
        if self.default_rule is not None:
            return self.default_rule
        raise ValueError(
            "No bootstrap rule matched stand "
            f"(vegetation_type={sample.vegetation_type}, age_class={sample.age_class})."
        )


@dataclass(slots=True)
class StandBootstrapAssignment:
    """Resolved bootstrap reference for a stand attribute sample."""

    stand_id: str
    vegetation_type: str
    age_class: str
    area: float
    bootstrap_id: str


@dataclass(slots=True)
class StandBootstrapLibraryEntry:
    """Loaded bootstrap payload (metadata + DBH vectors)."""

    identifier: str
    source: str
    metadata: dict[str, object]
    dbh_vectors: dict[str, list[float]]


@dataclass(slots=True)
class StandBootstrapManifest:
    """Resolved bootstrap manifest linking stands to payloads."""

    attributes_source: str | None
    plan_source: str | None
    assignments: Sequence[StandBootstrapAssignment]
    bootstraps: Mapping[str, StandBootstrapLibraryEntry]


def load_bootstrap_plan(path: Path) -> StandBootstrapPlan:
    """Load a bootstrap assignment plan describing stand → payload rules."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Bootstrap plan must be a mapping.")
    raw_rules = payload.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError("Bootstrap plan must include a non-empty 'rules' list.")
    base_dir = path.parent
    used_names: set[str] = set()
    rules: list[StandBootstrapRule] = []
    for index, record in enumerate(raw_rules):
        if not isinstance(record, Mapping):
            raise ValueError("Each bootstrap rule must be a mapping.")
        bootstrap_ref = record.get("bootstrap")
        if not isinstance(bootstrap_ref, str) or not bootstrap_ref:
            raise ValueError("Each bootstrap rule must supply a 'bootstrap' path.")
        bootstrap_path = _resolve_plan_path(base_dir, bootstrap_ref)
        name_value = record.get("name")
        derived_name = (
            str(name_value)
            if name_value
            else (Path(bootstrap_ref).stem or f"bootstrap-{index + 1}")
        )
        identifier = _ensure_unique_name(derived_name, used_names)
        raw_veg = record.get("vegetation_type")
        vegetation_type = str(raw_veg) if raw_veg is not None else None
        raw_age = record.get("age_class")
        age_class = str(raw_age) if raw_age is not None else None
        rules.append(
            StandBootstrapRule(
                identifier=identifier,
                bootstrap_path=bootstrap_path,
                source=bootstrap_ref,
                vegetation_type=vegetation_type,
                age_class=age_class,
            )
        )
    default_rule = None
    if "default_bootstrap" in payload:
        default_ref = payload["default_bootstrap"]
        if not isinstance(default_ref, str) or not default_ref:
            raise ValueError("default_bootstrap must be a string path when provided.")
        default_name = str(payload.get("default_name") or "default")
        identifier = _ensure_unique_name(default_name, used_names)
        default_rule = StandBootstrapRule(
            identifier=identifier,
            bootstrap_path=_resolve_plan_path(base_dir, default_ref),
            source=default_ref,
        )
    return StandBootstrapPlan(rules=tuple(rules), default_rule=default_rule)


def load_bootstrap_manifest(path: Path) -> StandBootstrapManifest:
    """Load a stand→bootstrap manifest produced by the linker CLI."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Bootstrap manifest must be a mapping.")
    assignments_payload = payload.get("assignments")
    if not isinstance(assignments_payload, list) or not assignments_payload:
        raise ValueError("Bootstrap manifest must include a non-empty 'assignments' list.")
    assignments: list[StandBootstrapAssignment] = []
    for record in assignments_payload:
        if not isinstance(record, Mapping):
            raise ValueError("Each assignment entry must be a mapping.")
        stand_id_raw = record.get("stand_id")
        bootstrap_id_raw = record.get("bootstrap_id")
        if not stand_id_raw or not bootstrap_id_raw:
            raise ValueError("Assignments require both 'stand_id' and 'bootstrap_id'.")
        assignments.append(
            StandBootstrapAssignment(
                stand_id=str(stand_id_raw),
                vegetation_type=str(record.get("vegetation_type", "")),
                age_class=str(record.get("age_class", "")),
                area=float(record.get("area", 0.0)),
                bootstrap_id=str(bootstrap_id_raw),
            )
        )
    raw_bootstraps = payload.get("bootstraps")
    if not isinstance(raw_bootstraps, Mapping):
        raise ValueError("Bootstrap manifest must include a 'bootstraps' mapping.")
    library: dict[str, StandBootstrapLibraryEntry] = {}
    for identifier_raw, entry in raw_bootstraps.items():
        identifier = str(identifier_raw)
        if not isinstance(entry, Mapping):
            raise ValueError("Each bootstrap entry must be a mapping.")
        metadata = entry.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Bootstrap '{identifier}' is missing metadata.")
        dbh_vectors_raw = entry.get("dbh_vectors")
        if not isinstance(dbh_vectors_raw, Mapping):
            raise ValueError(f"Bootstrap '{identifier}' is missing dbh_vectors.")
        vector_map: dict[str, list[float]] = {}
        for key, values in dbh_vectors_raw.items():
            if not isinstance(values, Sequence):
                raise ValueError("Each dbh_vectors entry must be a sequence.")
            vector_map[str(key)] = [float(value) for value in values]
        source_value = entry.get("source")
        library[identifier] = StandBootstrapLibraryEntry(
            identifier=identifier,
            source=str(source_value) if source_value is not None else identifier,
            metadata=dict(metadata),
            dbh_vectors=vector_map,
        )
    attributes_source = payload.get("attributes_source")
    plan_source = payload.get("plan_source")
    return StandBootstrapManifest(
        attributes_source=str(attributes_source) if attributes_source else None,
        plan_source=str(plan_source) if plan_source else None,
        assignments=tuple(assignments),
        bootstraps=library,
    )


def build_bootstrap_assignments(
    samples: Sequence[StandAttributeSample],
    plan: StandBootstrapPlan,
    *,
    id_prefix: str = "stand",
    start_index: int = 1,
) -> tuple[list[StandBootstrapAssignment], dict[str, StandBootstrapLibraryEntry]]:
    """Resolve bootstrap payloads for each stand sample."""

    if start_index < 0:
        raise ValueError("start_index must be non-negative.")
    prefix = id_prefix.strip()
    assignments: list[StandBootstrapAssignment] = []
    library: dict[str, StandBootstrapLibraryEntry] = {}
    counter = start_index
    for sample in samples:
        rule = plan.select_rule(sample)
        if rule.identifier not in library:
            metadata, dbh_vectors = _load_bootstrap_payload(rule.bootstrap_path)
            library[rule.identifier] = StandBootstrapLibraryEntry(
                identifier=rule.identifier,
                source=rule.source,
                metadata=metadata,
                dbh_vectors=dbh_vectors,
            )
        stand_id = f"{prefix}-{counter:04d}" if prefix else f"{counter:04d}"
        counter += 1
        assignments.append(
            StandBootstrapAssignment(
                stand_id=stand_id,
                vegetation_type=sample.vegetation_type,
                age_class=sample.age_class,
                area=sample.area,
                bootstrap_id=rule.identifier,
            )
        )
    return assignments, library


def _load_bootstrap_payload(path: Path) -> tuple[dict[str, object], dict[str, list[float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Bootstrap payload {path} is missing metadata.")
    dbh_vectors = payload.get("dbh_vectors")
    if not isinstance(dbh_vectors, Mapping):
        raise ValueError(f"Bootstrap payload {path} is missing dbh_vectors.")
    metadata_dict = dict(metadata)
    vector_map: dict[str, list[float]] = {}
    for key, values in dbh_vectors.items():
        if not isinstance(values, Sequence):
            raise ValueError("Each dbh_vectors entry must be a sequence.")
        vector_map[str(key)] = [float(value) for value in values]
    return metadata_dict, vector_map


def _ensure_unique_name(candidate: str, used: set[str]) -> str:
    name = candidate or "bootstrap"
    result = name
    suffix = 1
    while result in used:
        result = f"{name}-{suffix}"
        suffix += 1
    used.add(result)
    return result


def _resolve_plan_path(base_dir: Path, reference: str) -> Path:
    path = Path(reference)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if not path.exists():
        raise ValueError(f"Bootstrap payload not found: {path}")
    return path
