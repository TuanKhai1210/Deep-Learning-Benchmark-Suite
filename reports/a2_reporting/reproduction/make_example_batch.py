"""Regenerate the first seed-69 training batch using the frozen A2 data adapter."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reproduction/core/src"))
from dlbench.a2.data import CLASS_NAMES, TrashCanDataset, make_loader

PALETTE = np.array([
    [0, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
    [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
    [210, 245, 60], [250, 190, 212], [0, 128, 128], [220, 190, 255],
    [170, 110, 40], [255, 250, 200], [128, 0, 0], [170, 255, 195],
], dtype=np.uint8)


def show_rgb(tensor, preprocessing):
    mean = torch.tensor(preprocessing["mean"])[:, None, None]
    std = torch.tensor(preprocessing["std"])[:, None, None]
    return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def mask_image(mask):
    rgb = PALETTE[np.minimum(mask, 16)]
    rgb[mask == 255] = 128
    return rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "images")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = ROOT / "evidence"
    config = json.loads((evidence / "runs/69/S0/effective_config.json").read_text())
    prep = config["preprocessing"]
    seed = config["run"]["seed"]
    # Trainer passes a zero-based epoch to the data loader.
    epoch = 0
    batches = []
    for color_jitter in (False, True):
        data = TrashCanDataset(args.data_root, evidence / "manifest.json", "train",
                              prep, seed=seed, color_jitter=color_jitter)
        batches.append(next(iter(make_loader(data, batch_size=4, workers=0, epoch=epoch))))
    off, on = batches
    assert torch.equal(off["image_id"], on["image_id"])
    assert torch.equal(off["mask"], on["mask"])
    assert off["image"].shape == on["image"].shape == (4, 3, 512, 512)
    assert off["mask"].shape == (4, 512, 512)
    changed = (off["image"] != on["image"]).flatten(1).any(dim=1)
    assert changed.any()

    fig, axes = plt.subplots(3, 4, figsize=(10, 7.3), layout="constrained")
    rows = ("S0 RGB", "S1 RGB", "Shared target")
    samples = []
    for i, image_id in enumerate(off["image_id"].tolist()):
        mask = off["mask"][i].numpy()
        images = (show_rgb(off["image"][i], prep), show_rgb(on["image"][i], prep), mask_image(mask))
        axes[0, i].set_title(f"Image {image_id}", fontsize=11, weight="bold")
        samples.append({
            "image_id": image_id, "video_id": off["video_id"][i],
            "rgb_changed": bool(changed[i]), "target_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
            "classes": [CLASS_NAMES[v] if v != 255 else "ignore" for v in np.unique(mask)],
        })
        for j, image in enumerate(images):
            axis = axes[j, i]
            axis.imshow(image, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if i == 0:
                axis.set_ylabel(rows[j], weight="bold")
    fig.savefig(args.output / "a2_training_batch.png", dpi=200, facecolor="white")
    plt.close(fig)
    record = {
        "split": "train", "seed": seed, "loader_epoch_zero_based": epoch,
        "image_shape": list(off["image"].shape), "mask_shape": list(off["mask"].shape),
        "all_masks_equal": True, "images": samples,
        "display": "RGB denormalized; ignore gray and background black",
    }
    (args.output / "example_batch.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
