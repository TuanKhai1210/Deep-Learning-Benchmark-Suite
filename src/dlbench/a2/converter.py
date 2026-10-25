"""EDA-v1 converter port: function bodies preserved from the 2026-10-05 bundle.

Source: trashcan_material_tools.py; requires Pillow 12.3.0 for target identity.
No COCO label reduction. Same-class union, sticky cross-class ignore.
"""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import re
import numpy as np
from PIL import Image, ImageDraw

IGNORE = 255
CONVERTER_VERSION = "pillow_native_polygon_union_crossclass_ignore_v1"

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
