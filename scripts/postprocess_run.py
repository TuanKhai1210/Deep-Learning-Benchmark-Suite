"""Analyze one trusted saved run on its data/device; never train or use test data."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New output folder")
    parser.add_argument("--trust-checkpoint", action="store_true", required=True,
                        help="Only use checkpoints produced by your trusted team")
    parser.add_argument("--benchmark", action="store_true", help="Measure on the configured timing device; no fallback")
    args = parser.parse_args()
    import torch
    from dlbench.a1.analysis import analyze_run, analyze_predictions
    from dlbench.a1.benchmark import benchmark_inference
    from dlbench.a1.checkpoint import load_checkpoint
    from dlbench.a1.models.registry import build_model
    from dlbench.a1.trainer import evaluate_checkpoint
    from dlbench.common.reproducibility import collect_environment

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint = args.run_dir / "best.pt"
    # Evaluation verifies the saved model/config, split hash and normalization hash.
    result = evaluate_checkpoint(config, checkpoint, split="validation",
                                 smoke=config["run"].get("mode") == "smoke")
    saved = json.loads((args.run_dir / "metrics.json").read_text(encoding="utf-8"))
    import math
    for name in ("loss", "accuracy", "macro_f1"):
        if not math.isclose(getattr(result.metrics, name), saved[f"val_{name}"], rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(f"Re-evaluated {name} differs from saved best metrics")
    analyze_run(args.run_dir, output_dir=args.output_dir)
    analyze_predictions(result.predictions, args.output_dir / "validation", split="validation")
    if args.benchmark:
        device = torch.device(config["timing"]["device"])
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("Configured CUDA timing requires a GPU runtime; no CPU fallback")
        payload = load_checkpoint(checkpoint, map_location=device)
        model = build_model(config["model"]).to(device)
        model.load_state_dict(payload["model_state_dict"])
        # Fixed shape, preallocated normalized-input-domain zeros; no loading/transfer in timed region.
        shape = [config["timing"]["batch_size"], config["preprocessing"]["channels"],
                 *config["preprocessing"]["image_size"]]
        batch = torch.zeros(shape, dtype=torch.float32, device=device)
        record = benchmark_inference(model, batch, config["timing"])
        record.update({"run_id": args.run_dir.name,
                       "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                       "split_hash": payload["split_hash"], "statistics_hash": payload["statistics_hash"],
                       "input_policy": "Preallocated zeros of configured image shape; identical across models",
                       "environment": collect_environment()})
        with (args.output_dir / "benchmark.json").open("x", encoding="utf-8") as file:
            json.dump(record, file, indent=2, allow_nan=False)
    print(f"Saved validation analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
