"""Build a reviewed, group-disjoint ABO input cohort from the official release."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.request import urlopen

from .data import (
    CLASSES, FIELDS, assign_splits, discover_candidates, iter_listings, load_manifest,
    manifest_digest, publish_file, read_csv, validate_rows, write_csv,
)

ABO_URL = "https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com/"
SOURCES = {
    "abo-listings.tar": ("archives/abo-listings.tar",
                         "b7f7ceacb328fa5ab6e143b88e1f948443a877cfc95b67ff09c8ebabd50644e3"),
    "images.csv.gz": ("images/metadata/images.csv.gz",
                      "915263ef354d7a5d3503e899e610082f510c0011cccd5b4dc4f4c5efa86ef539"),
}
CANDIDATE_FIELDS = (*FIELDS, "brand", "model_number")
AUDIT_FIELDS = (*CANDIDATE_FIELDS, "pixel_sha256", "difference_hash", "decode_error")
CASE_FIELDS = ("sample_id", "related_id", "reason")
DECISION_FIELDS = (*CASE_FIELDS, "decision", "note")


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _metadata_value(listing, field):
    items = listing.get(field, [])
    return next((item["value"] for item in items
                 if item.get("language_tag") == "en_US" and item.get("value")), "")


def _download(url, destination, expected=None):
    def transfer(staged):
        with urlopen(url, timeout=45) as response, staged.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if expected and file_sha256(staged) != expected:
            raise ValueError(f"Official metadata SHA-256 differs: {url}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        publish_file(destination, transfer)
    elif expected and file_sha256(destination) != expected:
        raise ValueError(f"Existing metadata SHA-256 differs: {destination}")


def prepare_candidates(work, *, workers=12):
    """Fetch the pinned metadata and only the selected official small images."""
    work = Path(work)
    for name, (key, digest) in SOURCES.items():
        _download(ABO_URL + key, work / name, digest)
    listings = work / "abo-listings.tar"
    rows = discover_candidates(listings, work / "images.csv.gz")
    selected = {row["sample_id"]: row for row in rows}
    for listing in iter_listings(listings):
        row = selected.get(listing.get("item_id"))
        if (row is not None and listing.get("domain_name") == "amazon.com"
                and listing.get("main_image_id") == row["image_id"]):
            row["brand"] = _metadata_value(listing, "brand")
            row["model_number"] = _metadata_value(listing, "model_number")
    inventory = work / "candidates.csv"
    if inventory.exists():
        existing = read_csv(inventory)
        if existing != [{field: row.get(field, "") for field in CANDIDATE_FIELDS} for row in rows]:
            raise ValueError("Candidate rows differ from the existing inventory")
    else:
        write_csv(inventory, CANDIDATE_FIELDS, rows)

    def fetch(row):
        relative = row["image_path"]
        path = _image_path(work / "images", relative)
        try:
            _download(ABO_URL + "images/small/" + relative, path)
        except HTTPError as error:
            if error.code != 404:
                raise  # Allow a retry without treating a server failure as a bad pair.

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(fetch, rows):
            pass
    return inventory


def _image_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image path escapes the image root: {relative}")
    return path


def _image_signature(path):
    from PIL import Image, ImageOps

    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
        pixels = hashlib.sha256(f"{image.width}x{image.height}:".encode() + image.tobytes())
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).tobytes()
    bits = sum(int(gray[y * 9 + x] > gray[y * 9 + x + 1]) << (y * 8 + x)
               for y in range(8) for x in range(8))
    return pixels.hexdigest(), f"{bits:016x}"


def audit_candidates(candidates, image_root, output, *, sample_per_class=10, seed=36):
    """Check all decodes and create decisions for sampled pairs and suspected variants."""
    rows = sorted(read_csv(candidates), key=lambda row: row["sample_id"])
    validate_rows(rows, split=False)
    if sample_per_class < 1:
        raise ValueError("Select at least one QA sample per class")
    usable = []
    audit = []
    for row in rows:
        row = {field: row.get(field, "") for field in AUDIT_FIELDS}
        try:
            path = _image_path(image_root, row["image_path"])
            row["pixel_sha256"], row["difference_hash"] = _image_signature(path)
            usable.append(row)
        except (OSError, SyntaxError) as error:
            row["decode_error"] = type(error).__name__
        audit.append(row)

    cases = {}
    for label in CLASSES:
        matching = [row for row in usable if row["label"] == label]
        if len(matching) < sample_per_class:
            raise ValueError(f"{label} has only {len(matching)} readable images for QA")
        chosen = sorted(matching, key=lambda row: hashlib.sha256(
            f"a3-qa:{seed}:{row['sample_id']}".encode()).digest())[:sample_per_class]
        for row in chosen:
            cases[row["sample_id"], ""] = {"stratified_sample"}

    brands = defaultdict(list)
    title_stems = defaultdict(list)
    exact = defaultdict(list)
    for row in usable:
        exact[row["pixel_sha256"]].append(row)
        brand, model = (re.sub(r"[^a-z0-9]", "", row[key].casefold())
                        for key in ("brand", "model_number"))
        if brand and len(model) >= 3 and model not in {"unknown", "none", "notapplicable"}:
            brands[brand, model].append(row["sample_id"])
        # Many ABO names put size or color after the first comma.
        stem = tuple(re.findall(r"[a-z0-9]+", row["title"].split(",", 1)[0].casefold()))
        if len(stem) >= 5:
            title_stems[stem].append(row["sample_id"])
    for matches, reason in ((brands, "brand_model"), (title_stems, "title_stem")):
        for grouped in matches.values():
            for pair in combinations(grouped, 2):
                cases.setdefault(pair, set()).add(reason)

    # Compare representative pixel signatures. Exact duplicates are handled as a group.
    signatures = [(row_group[0]["sample_id"], int(row_group[0]["difference_hash"], 16))
                  for row_group in exact.values()]
    for (left, a), (right, b) in combinations(signatures, 2):
        if (a ^ b).bit_count() <= 4:
            cases.setdefault(tuple(sorted((left, right))), set()).add("near_image")
    for group in exact.values():
        if len({row["label"] for row in group}) > 1:
            for row in group:
                cases.setdefault((row["sample_id"], ""), set()).add("exact_label_conflict")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as staging:
        folder = Path(staging) / "audit"
        folder.mkdir()
        write_csv(folder / "inventory.csv", AUDIT_FIELDS, audit)
        write_csv(folder / "decisions.csv", DECISION_FIELDS, (
            dict(sample_id=left, related_id=right, reason="+".join(sorted(tags)),
                 decision="", note="") for (left, right), tags in sorted(cases.items())
        ))
        _json(folder / "audit.json", {
            "schema": 1, "sample_per_class": sample_per_class, "sample_seed": seed,
            "near_image_max_hamming": 4, "candidates_sha256": file_sha256(candidates),
            "inventory_sha256": file_sha256(folder / "inventory.csv"),
            "case_sha256": _case_digest(read_csv(folder / "decisions.csv")),
            "readable": len(usable), "candidate_pairs": len(rows),
            "review_cases": len(cases),
            "decode_errors": dict(Counter(row["decode_error"] for row in audit if row["decode_error"])),
        })
        if output.exists():
            raise FileExistsError(output)
        folder.rename(output)
    return output


def _case_digest(cases):
    return hashlib.sha256(json.dumps(sorted(tuple(row[key] for key in CASE_FIELDS)
                                            for row in cases)).encode()).hexdigest()


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def freeze_manifest(audit_dir, image_root, output, *, reviewer, seed=36):
    """Apply reviewed decisions, union duplicate/variant groups, then freeze the split."""
    reviewer = reviewer.strip()
    if not reviewer:
        raise ValueError("Provide the reviewer name")
    audit_dir = Path(audit_dir)
    inventory = audit_dir / "inventory.csv"
    audit = json.loads((audit_dir / "audit.json").read_text(encoding="utf-8"))
    if audit["inventory_sha256"] != file_sha256(inventory):
        raise ValueError("Audit inventory changed after image inspection")
    rows = read_csv(inventory)
    cases = read_csv(audit_dir / "decisions.csv")
    required, extra = [case for case in cases if case["reason"] != "manual"], [
        case for case in cases if case["reason"] == "manual"]
    if _case_digest(required) != audit["case_sha256"]:
        raise ValueError("Required review rows changed. Edit decision/note or append manual rows.")
    valid = {row["sample_id"]: row for row in rows if not row["decode_error"]}
    excluded = {row["sample_id"]: "decode:" + row["decode_error"] for row in rows if row["decode_error"]}
    edges, separate = [], []
    for case in required + extra:
        left, right, _, decision, note = (case[key].strip() for key in DECISION_FIELDS)
        # Pair keep retains both; only merge and separate constrain the groups.
        choices = {"keep", "merge", "separate"} if right else {"keep", "drop"}
        if (left not in valid or (right and (left == right or right not in valid))
                or decision not in choices or (decision == "drop" and not note)):
            raise ValueError(f"Unresolved/invalid decision: {left}, {right or 'sample review'}")
        if right and decision == "merge":
            edges.append((left, right))
        if right and decision == "separate":
            separate.append((left, right))
        if decision == "drop":
            excluded[left] = note

    parent = {sample: sample for sample in valid if sample not in excluded}

    def root(sample):
        while sample != parent[sample]:
            parent[sample] = parent[parent[sample]]
            sample = parent[sample]
        return sample

    def merge(left, right):
        a, b = sorted((root(left), root(right)))
        parent[b] = a

    pixels, image_files = {}, []
    for sample in parent:
        row = valid[sample]
        path = _image_path(image_root, row["image_path"])
        pixel, _ = _image_signature(path)
        if pixel != row["pixel_sha256"]:
            raise ValueError(f"Decoded image changed after audit: {sample}")
        image_files.append({"image_path": row["image_path"], "sha256": file_sha256(path)})
        same = pixels.setdefault(pixel, sample)
        if valid[same]["label"] != row["label"]:
            raise ValueError(f"Conflicting exact-image labels remain: {same}, {sample}")
        merge(same, sample)
    for left, right in edges:
        if left in parent and right in parent:
            merge(left, right)
    for left, right in separate:
        if left in parent and right in parent and root(left) == root(right):
            raise ValueError(f"Separate decision conflicts with a linked group: {left}, {right}")
    reviewed = [{**{field: valid[sample][field] for field in FIELDS}, "group_id": root(sample)}
                for sample in sorted(parent)]
    manifest = assign_splits(reviewed, seed)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as staging:
        folder = Path(staging) / "freeze"
        folder.mkdir()
        write_csv(folder / "reviewed.csv", FIELDS, reviewed)
        write_csv(folder / "manifest.csv", (*FIELDS, "split"), manifest)
        write_csv(folder / "exclusions.csv", ("sample_id", "reason"), (
            {"sample_id": key, "reason": excluded[key]} for key in sorted(excluded)
        ))
        write_csv(folder / "image_files.csv", ("image_path", "sha256"), image_files)
        _json(folder / "freeze.json", {
            "schema": 1, "reviewer": reviewer, "split_seed": seed,
            "manifest_sha256": manifest_digest(manifest),
            "image_files_sha256": file_sha256(folder / "image_files.csv"),
            "audit_sha256": file_sha256(audit_dir / "audit.json"),
            "decisions_sha256": file_sha256(audit_dir / "decisions.csv"),
            "candidate_pairs": len(rows), "excluded": len(excluded),
            "product_groups": len({row["group_id"] for row in manifest}),
            "split_counts": {split: dict(Counter(row["label"] for row in manifest
                                                 if row["split"] == split))
                             for split in ("train", "validation", "test")},
        })
        if output.exists():
            raise FileExistsError(output)
        folder.rename(output)
    return output / "manifest.csv"


def verified_freeze(folder, *, image_root=None):
    folder = Path(folder)
    receipt = json.loads((folder / "freeze.json").read_text(encoding="utf-8"))
    rows = load_manifest(folder / "manifest.csv")
    if receipt.get("schema") != 1 or receipt["manifest_sha256"] != manifest_digest(rows):
        raise ValueError("Frozen manifest differs from the curation receipt")
    if image_root is not None:
        inventory = folder / "image_files.csv"
        if file_sha256(inventory) != receipt["image_files_sha256"]:
            raise ValueError("Frozen image inventory changed")
        for image in read_csv(inventory):
            if file_sha256(_image_path(image_root, image["image_path"])) != image["sha256"]:
                raise ValueError(f"Frozen image file changed: {image['image_path']}")
    return rows
