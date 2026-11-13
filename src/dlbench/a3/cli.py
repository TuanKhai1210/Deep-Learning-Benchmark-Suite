"""Command line for the ABO base code."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from dlbench.common.config import _read_python_config

from .data import (
    CLASSES, FIELDS, SPLITS, assign_splits, discover_candidates,
    load_manifest, read_csv, write_csv,
)


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def _device(name):
    import torch

    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    return torch.device(name)


def _config(path, model, seed):
    config = _read_python_config(path)
    expected = {"batch_size", "epochs", "learning_rate", "projection"}
    if set(config) != expected:
        raise ValueError(f"Config must contain exactly {sorted(expected)}")
    if any(type(config[key]) is not int or config[key] < 1
           for key in ("batch_size", "epochs", "projection")):
        raise ValueError("Batch size, epochs and projection must be positive integers")
    lr = config["learning_rate"]
    if type(lr) not in {int, float} or not math.isfinite(lr) or lr <= 0:
        raise ValueError("Learning rate must be positive and finite")
    if not 0 <= seed < 2**32:
        raise ValueError("Run seed must be between 0 and 2**32 - 1")
    return {**config, "model": model, "seed": seed}


def _summary(rows):
    return {split: dict(Counter(row["label"] for row in rows if row["split"] == split))
            for split in SPLITS}


def _smoke(output):
    import numpy as np
    import torch

    from .engine import evaluate_test, train
    from .features import WIDTHS, save_cache
    from .model import MODALITIES

    output.mkdir(parents=True, exist_ok=False)
    rows = [
        dict(sample_id=f"{index}-{split}", image_id=f"{index}-{split}",
             image_path=f"{index}-{split}.jpg", title=f"Synthetic product {index}",
             label=label, group_id=f"{index}-{split}", split=split)
        for split in SPLITS for index, label in enumerate(CLASSES)
    ]
    manifest = output / "manifest.csv"
    write_csv(manifest, (*FIELDS, "split"), rows)
    rng = np.random.default_rng(36)
    caches = {}
    for modality, width in WIDTHS.items():
        caches[modality] = output / f"{modality}.npz"
        ordered = list(reversed(rows)) if modality == "title" else rows
        save_cache(caches[modality], ordered, modality,
                   rng.standard_normal((len(rows), width), dtype=np.float32),
                   {"model": "synthetic-smoke"})
    device = torch.device("cpu")
    for model in MODALITIES:
        config = {"model": model, "seed": 36, "batch_size": 5, "epochs": 2,
                  "projection": 8, "learning_rate": 0.001}
        run = train(manifest, caches, config, output, model, device)
        test_dir = evaluate_test(manifest, caches, run, device)
        print(test_dir / "metrics.json")


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    discover = commands.add_parser("candidates", help="read ABO metadata and list candidate pairs")
    discover.add_argument("--listings", type=Path, required=True)
    discover.add_argument("--image-index", type=Path, required=True)
    discover.add_argument("--output", type=Path, required=True)
    split = commands.add_parser("split", help="split a reviewed CSV by its product group IDs")
    split.add_argument("--reviewed", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--seed", type=int, default=36)
    validate = commands.add_parser("validate-manifest")
    validate.add_argument("--manifest", type=Path, required=True)
    cache = commands.add_parser("encode", help="extract one frozen modality")
    cache.add_argument("--manifest", type=Path, required=True)
    cache.add_argument("--modality", choices=("image", "title"), required=True)
    cache.add_argument("--image-root", type=Path)
    cache.add_argument("--raw-titles", action="store_true")
    cache.add_argument("--extra-mask-phrases", type=Path,
                       help="one train-derived phrase per line")
    cache.add_argument("--batch-size", type=_positive_int, default=32)
    cache.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    cache.add_argument("--output", type=Path, required=True)
    for name in ("train", "test"):
        run = commands.add_parser(name)
        run.add_argument("--manifest", type=Path, required=True)
        run.add_argument("--image-cache", type=Path)
        run.add_argument("--title-cache", type=Path)
        run.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        if name == "train":
            run.add_argument("--model", choices=(
                "image-only", "title-only", "concat", "product", "mean-ablation",
            ), required=True)
            run.add_argument("--config", type=Path, required=True)
            run.add_argument("--seed", type=int, required=True)
            run.add_argument("--output-root", type=Path, default=Path("runs/a3"))
            run.add_argument("--run-id", required=True)
        else:
            run.add_argument("--run-dir", type=Path, required=True)
    smoke = commands.add_parser("smoke", help="run all five variants on synthetic CPU inputs")
    smoke.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "candidates":
            rows = discover_candidates(args.listings, args.image_index)
            write_csv(args.output, FIELDS, rows)
            print(json.dumps(dict(Counter(row["label"] for row in rows)), indent=2))
        elif args.command == "split":
            rows = assign_splits(read_csv(args.reviewed), args.seed)
            write_csv(args.output, (*FIELDS, "split"), (
                {field: row[field] for field in (*FIELDS, "split")} for row in rows
            ))
            print(json.dumps(_summary(rows), indent=2))
        elif args.command == "validate-manifest":
            print(json.dumps(_summary(load_manifest(args.manifest)), indent=2))
        elif args.command == "encode":
            from .features import encode_images, encode_titles, save_cache

            rows = load_manifest(args.manifest)
            device = _device(args.device)
            if args.modality == "image":
                if args.image_root is None or args.raw_titles or args.extra_mask_phrases:
                    raise ValueError("Image encoding requires --image-root and takes no title options")
                vectors, provenance = encode_images(rows, args.image_root, args.batch_size, device)
            else:
                if args.image_root or (args.raw_titles and args.extra_mask_phrases):
                    raise ValueError("Use image root only for images and extra phrases only for masked titles")
                phrases = (args.extra_mask_phrases.read_text(encoding="utf-8-sig").splitlines()
                           if args.extra_mask_phrases else ())
                vectors, provenance = encode_titles(rows, args.batch_size, device,
                                                    raw=args.raw_titles, phrases=phrases)
            save_cache(args.output, rows, args.modality, vectors, provenance)
            print(args.output)
        elif args.command == "smoke":
            _smoke(args.output)
        else:
            from .engine import evaluate_test, train

            caches = {"image": args.image_cache, "title": args.title_cache}
            device = _device(args.device)
            if args.command == "train":
                config = _config(args.config, args.model, args.seed)
                print(train(args.manifest, caches, config, args.output_root, args.run_id, device))
            else:
                print(evaluate_test(args.manifest, caches, args.run_dir, device))
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
