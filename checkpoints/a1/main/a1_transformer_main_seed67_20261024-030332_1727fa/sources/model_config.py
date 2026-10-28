"""TRANSFORMER settings — owner C (Khải). Architecture values are starting proposals."""

CONFIG = {
    "protocol_file": "../protocol.py",
    "model": {
        "name": "transformer",
        "parameters": {
            "num_classes": 10,
            "patch_size": 4,
            "embed_dim": 64,
            "num_heads": 4,
            "num_layers": 2,
            "dropout": 0.1,
        },
    },
    "training": {
        # Match the shared batch size after the GPU batch-64 pilot.
        "batch_size": 64,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        # 0 disables early stopping; positive values are patience in epochs.
        "early_stopping_patience": 0,
    },
    "run": {
        # Repeat with run seeds 69420, 67 and 69 on the split created with seed 36.
        "seed": 69420,
        "device": "cuda",
        "output_root": "runs/a1",
    },
}
