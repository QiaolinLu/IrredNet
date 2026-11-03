"""Apply explicit material-ID splits without reshuffling or moving samples."""

import hashlib
import json
from pathlib import Path

import numpy as np


SPLIT_VERSION = "material_id_split_v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_material_split(path, samples, *, dataset_path=None):
    """Return ordered dataset indices and provenance for a complete partition.

    A manifest contains dataset_ids, train/val/test ID lists under splits, and
    explicit unused_ids (e.g. remainders after ratio truncation).
    The optional dataset_sha256 binds both labels and structures to the source.
    """
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != SPLIT_VERSION:
        raise ValueError("Unsupported material-ID split version.")
    if set(manifest.get("splits", {})) != {"train", "val", "test"}:
        raise ValueError("Split manifest must contain exactly train, val and test.")

    def checked_ids(values, name, *, allow_empty=False):
        if not isinstance(values, list) or not all(isinstance(v, str) and v for v in values):
            raise ValueError(f"{name} must be a list of nonempty material IDs.")
        if not allow_empty and not values:
            raise ValueError(f"{name} cannot be empty.")
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate material IDs in {name}.")
        return values

    expected_ids = checked_ids(manifest.get("dataset_ids"), "dataset_ids")
    ids = []
    for sample in samples:
        identifier = next((sample[key] for key in ("JARVIS_ID", "jid", "id") if key in sample), None)
        if identifier is None or not str(identifier):
            raise ValueError("Every sample needs a material ID for an explicit split.")
        ids.append(str(identifier))
    checked_ids(ids, "dataset samples")
    if set(ids) != set(expected_ids):
        raise ValueError("Dataset material IDs do not match the split manifest.")
    expected_digest = manifest.get("dataset_sha256")
    if expected_digest:
        if dataset_path is None or file_sha256(dataset_path) != expected_digest:
            raise ValueError("Dataset SHA256 does not match the split manifest.")

    partitions = {name: checked_ids(values, name) for name, values in manifest["splits"].items()}
    partitions["unused"] = checked_ids(manifest.get("unused_ids", []), "unused_ids", allow_empty=True)
    assigned = set()
    for name, values in partitions.items():
        if assigned.intersection(values):
            raise ValueError(f"Material IDs overlap across partitions ({name}).")
        assigned.update(values)
    if assigned != set(ids):
        raise ValueError("Partitions plus unused_ids must cover the exact dataset.")
    lookup = {identifier: index for index, identifier in enumerate(ids)}
    splits = {name: np.asarray([lookup[jid] for jid in partitions[name]], dtype=np.int64)
              for name in ("train", "val", "test")}
    provenance = {
        "path": str(path.resolve()), "sha256": file_sha256(path), "version": SPLIT_VERSION,
        "dataset_sha256": expected_digest, "strategy": manifest.get("strategy"),
        "seed": manifest.get("seed"), "unused_ids": partitions["unused"],
        "num_unused": len(partitions["unused"]),
    }
    return splits, provenance
