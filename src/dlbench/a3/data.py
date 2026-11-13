"""Manifest validation, deterministic group splits, and candidate discovery."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile

CLASSES = (
    "GROCERY", "SHOES", "CHAIR", "RUG", "SOFA", "WALL_ART", "TABLE",
    "LIGHT_FIXTURE", "LAMP", "STOOL_SEATING",
)
SPLITS = ("train", "validation", "test")
FIELDS = ("sample_id", "image_id", "image_path", "title", "label", "group_id")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"Invalid CSV header: {path}")
        rows = list(reader)
    if not rows or any(None in row or None in row.values() for row in rows):
        raise ValueError(f"Expected nonempty, rectangular CSV: {path}")
    return rows


def write_csv(path, fields, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_rows(rows, *, split=True):
    """Validate one sample per image and a shared, group-disjoint split."""
    seen = set()
    images = set()
    paths = set()
    groups = {}
    labels_by_split = defaultdict(set)
    for row in rows:
        if any(not row.get(field, "").strip() for field in FIELDS):
            raise ValueError("Each row must have nonempty manifest fields")
        sample_id, image_id = row["sample_id"], row["image_id"]
        path = row["image_path"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id):
            raise ValueError(f"Invalid sample ID: {sample_id}")
        if not re.fullmatch(r"[a-z0-9_./-]+", path) or any(
            part in {"", ".", ".."} for part in path.split("/")
        ) or PurePosixPath(path).suffix not in {".jpg", ".jpeg", ".png"}:
            raise ValueError(f"Expected a portable relative image path: {path}")
        if sample_id in seen or image_id in images or path in paths:
            raise ValueError(f"Duplicate sample ID, image ID or path at {sample_id}")
        seen.add(sample_id)
        images.add(image_id)
        paths.add(path)
        label = row["label"]
        if label not in CLASSES:
            raise ValueError(f"Unknown product type: {label}")
        if split:
            partition = row.get("split")
            if partition not in SPLITS:
                raise ValueError(f"Invalid split for {sample_id}: {partition}")
            group = row["group_id"]
            if groups.setdefault(group, partition) != partition:
                raise ValueError(f"Product group appears across splits: {group}")
            labels_by_split[partition].add(label)
    if not seen:
        raise ValueError("Manifest is empty")
    if split:
        for partition in SPLITS:
            missing = set(CLASSES) - labels_by_split[partition]
            if missing:
                raise ValueError(f"{partition} missing classes: {sorted(missing)}")


def load_manifest(path):
    rows = read_csv(path)
    validate_rows(rows)
    return rows


def manifest_digest(rows):
    """Bind caches to membership, labels, pairing and split, independent of CSV order."""
    canonical = [
        {field: row[field] for field in (*FIELDS, "split")}
        for row in sorted(rows, key=lambda row: row["sample_id"])
    ]
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode("utf-8")).hexdigest()


def assign_splits(rows, seed=36):
    """Allocate intact groups, aiming for 70/15/15 separately per class.

    Classes can share a product group. Process rare classes first and prioritize
    uncovered classes. Coverage is checked after assignment before publishing.
    """
    validate_rows(rows, split=False)
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    totals = {label: sum(row["label"] == label for row in rows) for label in CLASSES}
    fractions = dict(zip(SPLITS, (0.70, 0.15, 0.15)))
    counts = {split: dict.fromkeys(CLASSES, 0) for split in SPLITS}
    group_order = sorted(groups, key=lambda group: (
        min(totals[row["label"]] for row in groups[group]),
        -len(groups[group]),
        hashlib.sha256(f"{seed}:{group}".encode()).digest(),
    ))
    result = []
    for group in group_order:
        labels = {row["label"] for row in groups[group]}
        partition = max(SPLITS, key=lambda split: (
            sum(counts[split][label] == 0 for label in labels),
            sum(fractions[split] - counts[split][label] / totals[label] for label in labels),
        ))
        for row in groups[group]:
            counts[partition][row["label"]] += 1
            result.append({**row, "split": partition})
    validate_rows(result)
    return sorted(result, key=lambda row: row["sample_id"])


def discover_candidates(listings, image_index):
    """Select the proposed cohort from the official archives without extracting TAR members."""
    with gzip.open(image_index, "rt", encoding="utf-8", newline="") as stream:
        images = {row["image_id"]: row["path"] for row in csv.DictReader(stream)}
    by_image = defaultdict(list)
    with tarfile.open(listings) as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".json.gz"):
                continue
            with gzip.GzipFile(fileobj=archive.extractfile(member)) as stream:
                for line in stream:
                    listing = json.loads(line)
                    image = listing.get("main_image_id")
                    if listing.get("domain_name") != "amazon.com" or image not in images:
                        continue
                    labels = {item["value"] for item in listing.get("product_type", []) if item.get("value")}
                    title = next((
                        item["value"].strip() for item in listing.get("item_name", [])
                        if item.get("language_tag") == "en_US" and item.get("value", "").strip()
                    ), "")
                    if title and len(labels) == 1:
                        by_image[image].append((listing["item_id"], next(iter(labels)), title))
    rows = []
    for image, listings_for_image in by_image.items():
        if len({label for _, label, _ in listings_for_image}) != 1:
            continue
        item, label, title = min(listings_for_image)
        if label in CLASSES:
            rows.append(dict(sample_id=item, image_id=image, image_path=images[image],
                             title=title, label=label, group_id=image))
    rows.sort(key=lambda row: row["sample_id"])
    validate_rows(rows, split=False)
    return rows


def publish_file(path, write):
    """Publish a complete cache or run file with an atomic same-directory replace."""
    path = Path(path)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        staged = Path(temp.name)
    try:
        write(staged)
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)
