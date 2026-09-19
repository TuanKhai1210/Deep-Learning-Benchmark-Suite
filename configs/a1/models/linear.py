"""LINEAR settings — owner A (Thiên). Architecture values are starting proposals."""

CONFIG = {
    "protocol_file": "../protocol.py",
    "model": {
        "name": "linear",
        "parameters": {
            "input_dim": 784,
            "num_classes": 10,
        },
    },
    "training": {
        # Shared batch size for the Linear/MLP draft runs.
        "batch_size": 64,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "momentum": 0.9, 
        # 0 disables early stopping; positive values are patience in epochs.
        "early_stopping_patience": 15,
        # Choose "macro_f1" or "loss" for the early stopping monitor.
        "early_stopping_monitor": "macro_f1",
        "min_delta": 0.001,
        "save_frequency": 1000,
        "scheduler": {
            "name": "cosine",
            "parameters": {
                "T_max": 100,
                "eta_min": 0.00001
            }
        }
    },
    "run": {
        # Repeat with run seeds 69420, 67 and 69 on the split created with seed 36.
        "seed": 69420,
        "device": "cpu",
        "output_root": "runs/a1",
    },
}
