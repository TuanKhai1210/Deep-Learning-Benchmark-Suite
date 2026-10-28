"""RNN settings — owner B (Khoa). Architecture values are starting proposals."""

CONFIG = {
    "protocol_file": "../protocol.py",
    "model": {
        "name": "rnn",
        "parameters": {
            "input_size": 28, # NOT 784
            "num_classes": 10,
            "cell": "gru",
            "representation": "rows",
            "hidden_size": 128,
            "num_layers": 1,
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
