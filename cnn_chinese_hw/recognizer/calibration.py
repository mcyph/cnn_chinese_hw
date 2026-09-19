"""Separate checkpoint selection, temperature fitting and final corpus testing.

The current stroke corpus does not retain writer/device identities. This split
protects character variants and duplicate rounded geometry within the held-out
corpus; it must never be reported as writer- or device-disjoint validation.
"""
from __future__ import annotations

import hashlib
import json
from pydantic.dataclasses import dataclass
from iso_tools.inference.types import CONTRACT


def geometry_hash(strokes) -> str:
    geometry = [[[round(float(x), 6), round(float(y), 6)] for x, y in stroke] for stroke in strokes]
    return hashlib.sha256(json.dumps(geometry, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(config=CONTRACT, frozen=True)
class CalibrationPartitions:
    selection: tuple[int, ...]
    calibration: tuple[int, ...]
    test: tuple[int, ...]
    excluded_training_duplicates: tuple[int, ...]
    partition_sha256: str
    seed: int
    protected_fields: tuple[str, ...] = ("character_id", "rounded_geometry")
    unavailable_identities: tuple[str, ...] = ("writer_id", "device_id")
    scope: str = "heldout_corpus_selection_calibration_test"


def split_calibration_samples(train_samples, heldout_samples, *, seed: int = 13) -> CalibrationPartitions:
    from iso_tools.stt.evaluation.partitions import partition_rows, validate_partitions

    train_geometry = {geometry_hash(strokes) for _, strokes, _ in train_samples}
    included, excluded, rows = [], [], []
    for index, (label, strokes, _) in enumerate(heldout_samples):
        digest = geometry_hash(strokes)
        if digest in train_geometry:
            excluded.append(index)
            continue
        included.append(index)
        rows.append({"character_id": str(label), "rounded_geometry": digest})
    fields = ("character_id", "rounded_geometry")
    assignments = partition_rows(rows, identity_fields=fields, seed=seed,
        calibration_fraction=0.2, test_fraction=0.2)
    audit = validate_partitions([{**row, "partition": split} for row, split in zip(rows, assignments)], identity_fields=fields)
    indexes = {part: tuple(index for index, assigned in zip(included, assignments) if assigned == part)
               for part in ("fit", "calibration", "test")}
    return CalibrationPartitions(indexes["fit"], indexes["calibration"], indexes["test"], tuple(excluded), audit["partition_sha256"], seed)
