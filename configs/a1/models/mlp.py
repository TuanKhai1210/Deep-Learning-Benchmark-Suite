"""MLP settings — owner B (Khoa). Architecture values are starting proposals."""

CONFIG = {
    "protocol_file": "../protocol.py",
    "model": {
        "name": "mlp",
        "parameters": {
            "input_dim": 784,
            "num_classes": 10,
            "hidden_dims": [256,128],
            "dropout": 0.2,
            "hidden_activation": "relu" # choose between tanh, sigmoid, relu, leakyrelu, gelu (default: relu)
        },
    },
    "training": {
        # Shared batch size for the Linear/MLP draft runs.
        "batch_size": 64,
        "optimizer": "adamw",
        "learning_rate": 0.0005,
        "weight_decay": 0.0001,
        "momentum": 0.9, 
        # 0 disables early stopping; positive values are patience in epochs.
        "early_stopping_patience": 15,
        # Choose "macro_f1" or "loss" for the early stopping monitor.
        "early_stopping_monitor": "macro_f1",
        "min_delta": 0.0005,
        "save_frequency": 10,
        "scheduler": {
            "name": "cosine",
            "parameters": {
                "T_max": 200,
                "eta_min": 0.00001
            }
        }
    },
    "run": {
        # Repeat with run seeds 69420, 67 and 69 on the split created with seed 36.
        "seed": 69420,
        "device": "cuda",
        "output_root": "runs/a1",
    },
}
