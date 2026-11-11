"""Initial-code entry points. All data/model work is explicit, never on import."""
import argparse
from copy import copy
from pathlib import Path
import json

import torch
from .config import load_config, validate_config
from .data import TrashCanDataset, read_manifest
from .engine import run_epoch
from .losses import SegmentationLoss
from .models import build_model, count_parameters
from .trainer import fit, evaluate_checkpoint, save_json
from dlbench.common.reproducibility import seed_everything


def make_dataset(args, config, split):
    dataset = TrashCanDataset(args.data_root, args.manifest, split, config["preprocessing"],
                              seed=config["run"]["seed"],
                              color_jitter=config["run"]["color_jitter"])
    if getattr(args, "limit", None):
        dataset = copy(dataset)
        dataset.rows = dataset.rows[:args.limit]
    return dataset


def synthetic_smoke(config, *, size=64, device="cpu", pretrained=False):
    """One synthetic forward/backward step; explicitly not a data integration run."""
    seed_everything(config["run"]["seed"])
    model = build_model(config["model"], pretrained=pretrained).to(device)
    image = torch.rand(2, 3, size, size)
    target = torch.randint(0, 17, (2, size, size))
    target[:, :4, :] = 255
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"])
    result = run_epoch(model, [{"image": image, "mask": target}],
                       SegmentationLoss(**config["loss"]), device, optimizer)
    return {"kind": "synthetic_development_check", "model": config["model"]["name"],
            "pretrained": pretrained and config["model"]["name"] == "segformer",
            "shape": [2, 17, size, size], **count_parameters(model),
            "loss": result["loss"], "valid_pixels": result["valid_pixels"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-config")
    validate.add_argument("--config", required=True)
    validate.add_argument("--main", action="store_true")
    manifest = commands.add_parser("check-manifest")
    manifest.add_argument("--manifest", required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--pretrained", action="store_true",
                       help="Explicitly download/load pinned pretrained weights for SegFormer")
    for command in ("check-data", "train", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("--data-root", required=True)
        sub.add_argument("--manifest", required=True)
        if command != "evaluate":
            sub.add_argument("--config", required=True)
        if command == "check-data":
            sub.add_argument("--split", choices=("train", "validation", "test"), default="train")
            sub.add_argument("--limit", type=int, default=4)
        else:
            sub.add_argument("--device", default="cpu")
            sub.add_argument("--workers", type=int, default=0)
            sub.add_argument("--output", required=True)
        if command == "train":
            sub.add_argument("--development", action="store_true")
            sub.add_argument("--limit", type=int)
            sub.add_argument("--epochs", type=int)
            sub.add_argument("--resume", action="store_true")
        if command == "evaluate":
            sub.add_argument("--run-dir", required=True)
            sub.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args(argv)
    if getattr(args, "limit", None) is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if getattr(args, "workers", 0) < 0:
        parser.error("--workers must be nonnegative")
    if args.command == "check-manifest":
        value = read_manifest(args.manifest)
        result = {"content_sha256": value["content_sha256"], "images": len(value["images"]),
                  "groups": len(value["groups"]), "kind": "manifest_only_check"}
    elif args.command == "evaluate":
        config = json.loads((Path(args.run_dir) / "effective_config.json").read_text(encoding="utf-8"))
        if Path(args.output).exists():
            raise FileExistsError("Evaluation output already exists")
        data = make_dataset(args, config, args.split)
        result = evaluate_checkpoint(args.run_dir, data, device=args.device, workers=args.workers)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        save_json(args.output, result)
    else:
        config = load_config(args.config)
        if args.command == "validate-config":
            result = {"name": config["run"]["name"], "pending": validate_config(config, main=args.main)}
        elif args.command == "smoke":
            result = synthetic_smoke(config, device=args.device, pretrained=args.pretrained)
        elif args.command == "check-data":
            data = make_dataset(args, config, args.split)
            checked = [data[i] for i in range(len(data))]
            result = {"kind": "sampled_real_data_check", "split": args.split,
                      "checked_images": len(checked), "image_ids": [b["image_id"] for b in checked],
                      "shape": list(checked[0]["image"].shape),
                      "target_ids": sorted(set(int(v) for b in checked for v in b["mask"].unique()))}
        elif args.command == "train":
            validate_config(config, main=not args.development)
            if args.limit and not args.development:
                parser.error("--limit is development-only")
            if args.development and (not args.limit or not args.epochs):
                parser.error("--development requires explicit --limit and --epochs")
            result = fit(config, make_dataset(args, config, "train"),
                         make_dataset(args, config, "validation"), args.output,
                         device=args.device, workers=args.workers,
                         development=args.development, epochs=args.epochs, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    main()
