"""MLP settings — owner B (Khoa). Architecture values are starting proposals."""

CONFIG = {
    "protocol_file": "../protocol.py",
    "model": {
        "name": "mlp",
        "parameters": {
            "input_dim": 784,
            "num_classes": 10,
            "hidden_dims": [256,128],
            "dropout": 0.1,
            "hidden_activation": "relu" # choose between tanh, sigmoid, relu, leakyrelu, gelu (default: relu)
        },
    },
    "training": {
        # TODO team: choose after a memory/time check. 0 means undecided.
        "batch_size": 0,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        # 0 disables early stopping; positive values are patience in epochs.
        "early_stopping_patience": 0,
    },
    "run": {
        # Repeat with run seeds 69420, 67 and 69 on the split created with seed 36.
        "seed": 69420,
        "device": "cpu",
        "output_root": "runs/a1",
    },
}
