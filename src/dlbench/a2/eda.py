"""Reproducible semantic-mask EDA and video-group split for TrashCan Material.

Source data is read only. Use this rasterize_semantic function for future model
targets too. This defines derived semantic labels; it is not a COCO AP evaluator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageDraw

SPLITS = ("train", "validation", "test")
RATIOS = np.array([0.70, 0.15, 0.15])
IGNORE = 255
CONVERTER_VERSION = "pillow_native_polygon_union_crossclass_ignore_v1"
DRIVE_URL = "https://drive.google.com/drive/folders/1VwT0UyleGYyF9yrCtXaOwZpnp-ZRrfGD"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_records(dataset_root):
    root = Path(dataset_root)
    records, seen_ids, seen_ann_ids, seen_names = [], set(), set(), set()
    categories, hashes, licenses = None, {}, {}
    for part in ("train", "val"):
        source = root / f"instances_{part}_trashcan.json"
        data = json.loads(source.read_text(encoding="utf-8"))
        category_map = {int(c["id"]): c["name"] for c in data["categories"]}
        if categories is None:
            categories = category_map
        elif categories != category_map:
            raise ValueError("Category mapping differs between source partitions")
        by_image = defaultdict(list)
        image_ids = {im["id"] for im in data["images"]}
        if len(image_ids) != len(data["images"]):
            raise ValueError("Duplicate source image IDs")
        for ann in data["annotations"]:
            if ann["id"] in seen_ann_ids:
                raise ValueError("Duplicate annotation IDs across source files")
            seen_ann_ids.add(ann["id"])
            if ann["image_id"] not in image_ids or ann["category_id"] not in category_map:
                raise ValueError("Invalid annotation reference")
            by_image[ann["image_id"]].append(ann)
        for info in data["images"]:
            if info["id"] in seen_ids or info["file_name"] in seen_names:
                raise ValueError("Duplicate image ID/name across source files")
            seen_ids.add(info["id"])
            seen_names.add(info["file_name"])
            match = re.fullmatch(r"(vid_\d+)_frame\d+\.jpg", info["file_name"])
            if not match:
                raise ValueError(f"Cannot infer video group: {info['file_name']}")
            if not by_image[info["id"]]:
                raise ValueError(f"Image has no annotation: {info['file_name']}")
            records.append({**info, "source_partition": part,
                            "relative_path": f"{part}/{info['file_name']}",
                            "video_id": match.group(1),
                            "annotations": sorted(by_image[info["id"]], key=lambda a: a["id"])})
        hashes[source.name] = sha256_file(source)
        licenses[part] = data.get("licenses", [])
    if sorted(categories) != list(range(1, 17)):
        raise ValueError("Expected the supplied TrashCan Material category IDs 1..16")
    return sorted(records, key=lambda r: r["id"]), categories, hashes, licenses


def rasterize_semantic(record):
    """Pillow native-canvas fill; same-class union; cross-class overlap -> 255.

    Polygon lists within an annotation are unioned. Pillow clips canvas coverage
    outside the image. Fractional vertices are passed to ImageDraw unchanged;
    pin the Pillow version. There is no overwrite priority between classes.
    """
    width, height = int(record["width"]), int(record["height"])
    labels = np.zeros((height, width), dtype=np.uint8)
    union = np.zeros((height, width), dtype=bool)
    conflicts = np.zeros((height, width), dtype=bool)
    empty_masks, border_annotations = [], []
    for ann in record["annotations"]:
        if not isinstance(ann["segmentation"], list) or not ann["segmentation"]:
            raise ValueError(f"Expected nonempty COCO polygon lists: ann {ann['id']}")
        canvas = Image.new("1", (width, height))
        draw = ImageDraw.Draw(canvas)
        outside = False
        for segment in ann["segmentation"]:
            coords = np.asarray(segment, dtype=np.float64)
            if coords.size < 6 or coords.size % 2 or not np.isfinite(coords).all():
                raise ValueError(f"Malformed polygon: ann {ann['id']}")
            points = coords.reshape(-1, 2)
            if abs(float(np.dot(points[:, 0], np.roll(points[:, 1], -1)) -
                         np.dot(points[:, 1], np.roll(points[:, 0], -1)))) < 1e-9:
                raise ValueError(f"Degenerate polygon: ann {ann['id']}")
            outside |= bool((points < 0).any() or (points[:, 0] > width).any() or
                            (points[:, 1] > height).any())
            draw.polygon([(float(x), float(y)) for x, y in points], fill=1)
        pixels = np.asarray(canvas, dtype=bool)
        if not pixels.any():
            empty_masks.append(ann["id"])
        if outside:
            border_annotations.append(ann["id"])
        cid = int(ann["category_id"])
        conflicts |= pixels & (labels != 0) & (labels != cid)
        # labels may store an arbitrary owner at conflicts, but conflicts is
        # sticky and final labels there are always IGNORE, regardless of order.
        labels[pixels & (labels == 0)] = cid
        union |= pixels
    labels[conflicts] = IGNORE
    return labels, union, conflicts, empty_masks, border_annotations


def letterbox_target(rgb, labels, size=512):
    """RGB padding is black; target padding is IGNORE, not real background."""
    height, width = labels.shape
    scale = size / max(width, height)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    left, top = (size - new_w) // 2, (size - new_h) // 2
    image_out = Image.new("RGB", (size, size), color=(0, 0, 0))
    image_out.paste(rgb.resize((new_w, new_h), Image.Resampling.BILINEAR), (left, top))
    target_out = np.full((size, size), IGNORE, dtype=np.uint8)
    resized = np.asarray(Image.fromarray(labels).resize((new_w, new_h), Image.Resampling.NEAREST))
    target_out[top:top+new_h, left:left+new_w] = resized
    return image_out, target_out


def measure_records(records, dataset_root, output_dir, write_masks=False):
    output_dir = Path(output_dir)
    mask_dir = output_dir / "semantic_masks"
    if write_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)
    measured = []
    for index, record in enumerate(records):
        with Image.open(Path(dataset_root) / record["relative_path"]) as rgb:
            if rgb.size != (record["width"], record["height"]):
                raise ValueError(f"Image dimensions differ: {record['file_name']}")
            rgb.verify()
        labels, union, conflicts, empty_masks, border = rasterize_semantic(record)
        counts = np.bincount(labels.ravel(), minlength=256)
        object_counts = np.bincount([a["category_id"] for a in record["annotations"]], minlength=17)
        measured.append({"image_id": record["id"], "file_name": record["file_name"],
                         "relative_path": record["relative_path"],
                         "source_partition": record["source_partition"], "video_id": record["video_id"],
                         "width": record["width"], "height": record["height"],
                         "annotation_counts": object_counts[:17].tolist(),
                         "pixel_counts": counts[:17].tolist(), "ignore_pixels": int(counts[IGNORE]),
                         "foreground_union_pixels": int(union.sum()),
                         "total_pixels": int(labels.size), "conflict_pixels": int(conflicts.sum()),
                         "empty_rasterized_annotation_ids": empty_masks,
                         "border_annotation_ids": border,
                         "target_sha256": hashlib.sha256(labels.tobytes()).hexdigest()})
        if write_masks:
            Image.fromarray(labels).save(mask_dir / (Path(record["file_name"]).stem + ".png"))
        if (index + 1) % 1000 == 0:
            print(f"Measured {index+1}/{len(records)} images", flush=True)
    return measured


def aggregate(rows, categories):
    pixel_counts = np.sum([r["pixel_counts"] for r in rows], axis=0, dtype=np.int64)
    annotation_counts = np.sum([r["annotation_counts"] for r in rows], axis=0, dtype=np.int64)
    total = sum(r["total_pixels"] for r in rows)
    ignore = sum(r["ignore_pixels"] for r in rows)
    output = {"images": len(rows), "videos": len({r["video_id"] for r in rows}),
              "annotations": int(annotation_counts.sum()), "total_pixels": total,
              "foreground_union_pixels": sum(r["foreground_union_pixels"] for r in rows),
              "foreground_supervised_pixels": int(pixel_counts[1:].sum()),
              "background_pixels": int(pixel_counts[0]), "ignore_pixels": ignore,
              "foreground_union_percent": 100 * sum(r["foreground_union_pixels"] for r in rows) / total,
              "foreground_supervised_percent": 100 * int(pixel_counts[1:].sum()) / total,
              "background_percent": 100 * int(pixel_counts[0]) / total,
              "ignore_percent": 100 * ignore / total,
              "empty_rasterized_annotations": sum(len(r["empty_rasterized_annotation_ids"]) for r in rows),
              "resolution_counts": dict(Counter(f"{r['width']}x{r['height']}" for r in rows)), "classes": {}}
    if int(pixel_counts.sum()) + ignore != total:
        raise AssertionError("Pixel accounting does not cover the image canvas")
    for cid, name in sorted(categories.items()):
        matching = [r for r in rows if r["pixel_counts"][cid] > 0]
        output["classes"][name] = {"category_id": cid,
            "annotations": int(annotation_counts[cid]),
            "annotated_images": sum(r["annotation_counts"][cid] > 0 for r in rows),
            "supervised_images": len(matching), "videos": len({r["video_id"] for r in matching}),
            "pixels": int(pixel_counts[cid]), "percent_of_all_pixels": 100 * int(pixel_counts[cid]) / total}
    return output


def make_split(rows, seed=36, restarts=24):
    """Greedy + local moves using metadata only; all randomness derives from seed.

    Balance image count, per-class supervised image/pixel counts and video class
    coverage. Require >=3 train and >=2 validation/test videos per class.
    """
    groups = sorted({r["video_id"] for r in rows})
    group_rows = defaultdict(list)
    for row in rows:
        group_rows[row["video_id"]].append(row)
    image_n, class_images, class_pixels, presence = [], [], [], []
    for group in groups:
        rr = group_rows[group]
        pp = np.array([r["pixel_counts"][1:] for r in rr], dtype=np.int64)
        image_n.append(len(rr))
        class_images.append((pp > 0).sum(axis=0))
        class_pixels.append(pp.sum(axis=0))
        presence.append((pp.sum(axis=0) > 0).astype(float))
    image_n = np.asarray(image_n, dtype=float)
    class_images = np.asarray(class_images, dtype=float)
    class_pixels = np.asarray(class_pixels, dtype=float)
    presence = np.asarray(presence, dtype=float)
    if (presence.sum(axis=0) < 7).any():
        raise ValueError("Insufficient distinct video support for 3/2/2 per-class coverage")
    minimum = np.array([3, 2, 2])[:, None]
    features = np.column_stack([image_n, class_images, class_pixels, np.ones(len(groups))])
    totals = features.sum(axis=0)
    # normalized squared differences measure absolute fraction deviations.
    weights = np.r_[100.0, np.full(16, 8.0 / 16), np.full(16, 2.0 / 16), 1.0]
    target = RATIOS[:, None]

    def objective(counts, support):
        return float(np.sum(((counts / totals) - target) ** 2 * weights) +
                     1000 * np.maximum(minimum - support, 0).sum())

    rng = np.random.default_rng(seed)
    rarity = (presence / presence.sum(axis=0)).sum(axis=1)
    best = None
    for restart in range(restarts):
        order = np.argsort(-(rarity * rng.uniform(0.6, 1.4, len(groups)) +
                             image_n / image_n.sum() * 2), kind="stable")
        assignment = np.full(len(groups), -1, dtype=int)
        counts = np.zeros((3, features.shape[1]))
        support = np.zeros((3, 16))
        for g in order:
            costs = []
            for s in range(3):
                cc, vv = counts.copy(), support.copy()
                cc[s] += features[g]
                vv[s] += presence[g]
                costs.append(objective(cc, vv))
            s = int(np.argmin(costs))
            assignment[g] = s
            counts[s] += features[g]
            support[s] += presence[g]
        for sweep in range(30):
            changed = False
            for g in rng.permutation(len(groups)):
                old = int(assignment[g])
                best_move, best_score = old, objective(counts, support)
                for s in range(3):
                    if s == old:
                        continue
                    cc, vv = counts.copy(), support.copy()
                    cc[old] -= features[g]; cc[s] += features[g]
                    vv[old] -= presence[g]; vv[s] += presence[g]
                    cost = objective(cc, vv)
                    if cost < best_score - 1e-12:
                        best_move, best_score = s, cost
                if best_move != old:
                    counts[old] -= features[g]; counts[best_move] += features[g]
                    support[old] -= presence[g]; support[best_move] += presence[g]
                    assignment[g] = best_move
                    changed = True
            if not changed:
                break
        score = objective(counts, support)
        feasible = bool((support >= minimum).all())
        if feasible and (best is None or score < best[0] - 1e-12):
            best = (score, assignment.copy())
        if (restart + 1) % 8 == 0:
            print(f"Split search {restart+1}/{restarts}; feasible best={None if best is None else best[0]:}", flush=True)
    if best is None:
        raise ValueError("No split meets required video support; revise the split constraints explicitly")
    return {group: SPLITS[int(best[1][i])] for i, group in enumerate(groups)}, best[0]


def make_manifest(rows, group_split, source_hashes, categories, seed, score):
    entries = [{"image_id": r["image_id"], "file_name": r["file_name"],
                "relative_path": r["relative_path"], "source_partition": r["source_partition"],
                "video_id": r["video_id"], "split": group_split[r["video_id"]],
                "target_sha256": r["target_sha256"]} for r in rows]
    manifest = {"protocol_version": "trashcan_material_semantic_v1", "split_seed": seed,
                "split_unit": "filename_video_prefix", "target_image_ratios": dict(zip(SPLITS, RATIOS.tolist())),
                "group_split_algorithm": "metadata_greedy_local_moves_24_restarts_v1",
                "minimum_class_video_support": dict(zip(SPLITS, (3, 2, 2))),
                "objective": score, "source_annotation_sha256": source_hashes,
                "rasterizer": {"version": CONVERTER_VERSION, "pillow_version": PIL.__version__,
                    "overlap_policy": "same_class_union_cross_class_ignore", "ignore_id": IGNORE,
                    "background_id": 0, "padding_id": IGNORE},
                "categories": [{"id": 0, "name": "background"}] +
                    [{"id": cid, "name": name} for cid, name in sorted(categories.items())],
                "groups": [{"video_id": v, "split": s} for v, s in sorted(group_split.items())],
                "images": entries}
    # Digest is canonical content before the digest field itself is added.
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    manifest["content_sha256"] = hashlib.sha256(canonical).hexdigest()
    return manifest


def verify_outputs(rows, manifest, summary):
    assigned = manifest["images"]
    if len(assigned) != len(rows) or len({r["image_id"] for r in assigned}) != len(rows):
        raise AssertionError("Missing/duplicated manifest entries")
    group_sets = {s: {r["video_id"] for r in assigned if r["split"] == s} for s in SPLITS}
    for i, s in enumerate(SPLITS):
        for other in SPLITS[i+1:]:
            if group_sets[s] & group_sets[other]:
                raise AssertionError("Video leakage in manifest")
        minimum = 3 if s == "train" else 2
        for cls in summary[s]["classes"].values():
            if cls["videos"] < minimum or cls["pixels"] <= 0:
                raise AssertionError("Class support constraint failed")
    payload = {k: v for k, v in manifest.items() if k != "content_sha256"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    if digest != manifest["content_sha256"]:
        raise AssertionError("Manifest content hash does not verify")
    return {"unique_images": len(assigned), "unique_videos": len(set.union(*group_sets.values())),
            "pairwise_video_overlap": 0, "all_classes_meet_video_minima": True,
            "manifest_content_sha256_verified": True}


def save_figures(records, rows, categories, summary, dataset_root, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    fig_dir = Path(output_dir) / "figures"
    qa_dir = Path(output_dir) / "qa_samples"
    fig_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)
    names = [categories[c] for c in sorted(categories)]
    trash = [n for n in names if n.startswith("trash_")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    pooled = aggregate(rows, categories)
    for ax, field, title in zip(axes, ("annotations", "pixels"), ("Annotated objects", "Supervised semantic pixels")):
        values = [pooled["classes"][n][field] for n in trash]
        ax.barh(trash, values, color="#31688e")
        ax.invert_yaxis(); ax.set_title(title); ax.set_xlabel("Count")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.suptitle("TrashCan Material: eight trash categories (pooled source data)")
    fig.tight_layout(); fig.savefig(fig_dir / "material_distribution.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 6))
    positions = np.arange(len(names))
    for i, s in enumerate(SPLITS):
        values = [summary[s]["classes"][n]["videos"] for n in names]
        ax.bar(positions + (i - 1) * .25, values, width=.25, label=s)
    ax.set_xticks(positions, names, rotation=50, ha="right")
    ax.set_ylabel("Distinct videos with supervised pixels"); ax.set_title("Frozen video split: class support")
    ax.legend(); fig.tight_layout(); fig.savefig(fig_dir / "split_class_video_support.png", dpi=160); plt.close(fig)
    cmap = plt.get_cmap("tab20")
    record_map = {r["id"]: r for r in records}
    qa_files = []
    for s in SPLITS:
        candidates = [r for r in rows if r["split"] == s and sum(r["annotation_counts"][9:]) > 0]
        selected, used_videos = [], set()
        # Include a sparse scene and a moderately annotated scene, no stale files.
        for r in sorted(candidates, key=lambda r: (abs(sum(r["annotation_counts"]) - 3), r["image_id"])):
            if r["video_id"] not in used_videos:
                selected.append(r); used_videos.add(r["video_id"])
            if len(selected) == 2:
                break
        for row in selected:
            record = record_map[row["image_id"]]
            target, _, _, _, _ = rasterize_semantic(record)
            with Image.open(Path(dataset_root) / record["relative_path"]) as rgb:
                rgb = rgb.convert("RGB")
                overlay = np.asarray(rgb).astype(float)
                for cid in sorted(categories):
                    active = target == cid
                    overlay[active] = .55 * overlay[active] + .45 * np.array(cmap((cid - 1) % 20)[:3]) * 255
                overlay[target == IGNORE] = (255, 0, 255)
                fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
                axes[0].imshow(rgb); axes[0].set_title("RGB")
                axes[1].imshow(overlay.astype(np.uint8)); axes[1].set_title("Semantic overlay")
                for ax in axes: ax.axis("off")
                handles = [mpatches.Patch(color=cmap((cid-1)%20), label=categories[cid]) for cid in sorted(categories) if (target == cid).any()]
                if (target == IGNORE).any():
                    handles.append(mpatches.Patch(color="magenta", label="ignore overlap"))
                fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8)
                fig.suptitle(f"{s}: {record['file_name']}")
                fig.tight_layout(rect=(0, .10, 1, .95))
                filename = f"{s}_{Path(record['file_name']).stem}_semantic_qa.png"
                fig.savefig(qa_dir / filename, dpi=150); plt.close(fig)
                qa_files.append("qa_samples/" + filename)
    return qa_files


def write_report(output_dir, metadata, summary, categories, qa_files):
    pooled = metadata["pooled"]
    lines = ["# TrashCan Material EDA and frozen video split", "",
        "Measured on 5 October 2026 from the supplied archive. Counts of semantic pixels use the documented Pillow converter; polygon area metadata is not substituted for pixel counts.", "",
        "## Dataset and labels", "",
        f"{pooled['images']:,} images, {pooled['annotations']:,} object annotations, 16 foreground classes plus background, {pooled['videos']} filename video groups.", "",
        "| ID | Class | Objects | Annotated images | Videos with supervised pixels | Supervised pixels |", "|---|---|---:|---:|---:|---:|"]
    for cid, name in sorted(categories.items()):
        v = pooled["classes"][name]
        lines.append(f"| {cid} | {name} | {v['annotations']} | {v['annotated_images']} | {v['videos']} | {v['pixels']} |")
    lines += ["", "## Semantic conversion policy", "",
        f"Converter `{CONVERTER_VERSION}`, Pillow {PIL.__version__}. Object polygons are filled on the native image canvas; coverage outside the canvas is clipped. Same-class masks are unioned. Cross-class overlaps receive 255 and are excluded from loss/evaluation. Background is 0. Letterbox padding also receives 255.", "",
        "## Original partitions and corrected pixel accounting", "",
        "| Source | Images | Videos | Foreground union % | Supervised foreground % | Background % | Ignore % |", "|---|---:|---:|---:|---:|---:|---:|"]
    for part, v in metadata["source_partitions"].items():
        lines.append(f"| {part} | {v['images']} | {v['videos']} | {v['foreground_union_percent']:.4f} | {v['foreground_supervised_percent']:.4f} | {v['background_percent']:.4f} | {v['ignore_percent']:.4f} |")
    lines += ["", "127 source video groups occur in both original train and validation. The original labels remain accessible in their original folders.", "",
        "## New split", "", "| Split | Images | Image share % | Videos | Annotations |", "|---|---:|---:|---:|---:|"]
    for s in SPLITS:
        v = summary[s]
        lines.append(f"| {s} | {v['images']} | {100*v['images']/pooled['images']:.2f} | {v['videos']} | {v['annotations']} |")
    lines += ["", "Seed 36; each class has at least three train videos and two videos in each of validation and test. Candidate assignment uses metadata only, with 24 deterministic greedy/local-search restarts. Exact percentages may differ from 70/15/15 because videos are indivisible.", "",
        "| Class | Train videos | Validation videos | Test videos |", "|---|---:|---:|---:|"]
    for name in categories.values():
        lines.append("| " + name + " | " + " | ".join(str(summary[s]["classes"][name]["videos"]) for s in SPLITS) + " |")
    lines += ["", "![Material distribution](figures/material_distribution.png)", "",
        "![Video support](figures/split_class_video_support.png)", "", "## QA from this execution", ""]
    for f in qa_files:
        lines.append(f"![{Path(f).stem}]({f})\n")
    lines += ["## Provenance", "", f"Working source: {DRIVE_URL}", "",
        "See metadata.json for source hashes, runtime versions and mask-accounting checks. See splits/trashcan_material_seed36_v1.json for image paths, video assignments and canonical manifest digest."]
    (Path(output_dir) / "eda_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(dataset_root, output_dir, seed=36, write_masks=False):
    root, out = Path(dataset_root), Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a new output directory to avoid stale outputs: {out}")
    out.mkdir(parents=True, exist_ok=True)
    records, categories, hashes, licenses = load_records(root)
    rows = measure_records(records, root, out, write_masks)
    group_split, score = make_split(rows, seed=seed)
    for row in rows:
        row["split"] = group_split[row["video_id"]]
    summary = {s: aggregate([r for r in rows if r["split"] == s], categories) for s in SPLITS}
    metadata = {"dataset": "TrashCan 1.0 Material", "working_source": DRIVE_URL,
        "measured_date": "2026-10-05", "rasterizer_version": CONVERTER_VERSION,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pillow": PIL.__version__},
        "source_annotation_sha256": hashes, "source_json_licenses": licenses,
        "source_partitions": {part: aggregate([r for r in rows if r["source_partition"] == part], categories) for part in ("train", "val")},
        "pooled": aggregate(rows, categories)}
    manifest = make_manifest(rows, group_split, hashes, categories, seed, score)
    verification = verify_outputs(rows, manifest, summary)
    save_json(out / "image_statistics.json", rows)
    save_json(out / "metadata.json", metadata)
    save_json(out / "split_summary.json", summary)
    save_json(out / "verification.json", verification)
    save_json(out / "splits/trashcan_material_seed36_v1.json", manifest)
    with (out / "split_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["image_id", "file_name", "relative_path", "source_partition", "video_id", "split"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in manifest["images"])
    qa_files = save_figures(records, rows, categories, summary, root, out)
    save_json(out / "qa_manifest.json", qa_files)
    write_report(out, metadata, summary, categories, qa_files)
    print(json.dumps({"verification": verification, "split": {s: {k: summary[s][k] for k in ("images", "videos", "annotations")} for s in SPLITS}, "content_sha256": manifest["content_sha256"]}, indent=2), flush=True)
    return metadata, summary, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--write-masks", action="store_true")
    args = parser.parse_args()
    run(args.dataset_root, args.output_dir, args.seed, args.write_masks)
