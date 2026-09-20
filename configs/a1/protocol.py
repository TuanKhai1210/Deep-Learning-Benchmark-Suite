"""Shared A1 policy. The team agreed on one split seed and three run seeds.

Other experimental choices remain proposals until reviewed and measured.
Use one literal CONFIG dictionary: no imports, functions or computations.
"""

CONFIG = {
    "protocol": {
        "id": "a1-v0",
        "status": "frozen",
        "approved_by": [
            "Nguyễn Hạo Thiên",
            "Nguyễn Anh Khoa",
            "Tạ Tuấn Khải",
        ],
    },
    "data": {
        "dataset": "fashion_mnist",
        "root": "data",
        # Fixed split seed; never resplit when the model run seed changes.
        "split_seed": 36,
        "train_size": 50000,
        "validation_size": 10000,
        "test_size": 10000,
        "stratified": True,
        "split_file": "configs/a1/splits/fashion_mnist_split.json",
        "download": True
    },
    "preprocessing": {
        "image_size": [28, 28],
        "channels": 1,
        # Measured on the unaugmented 50k TRAIN partition (split_seed=36)
        "mean": [0.28585961086061534],
        "std": [0.352790375749525],
        "augmentations": [
            {
                "name": "random_crop",
                "size": [28, 28],
                "padding": 2,
                "padding_mode": "constant",
                "fill": 0,
            },
        ],
    },
    "evaluation": {
        "metrics": ["accuracy", "macro_f1"],
        "labels": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "zero_division": 0,
    },
    "checkpoint": {
        "monitor": "val_macro_f1",
        "mode": "max",
        "tie_breaker": ["val_loss_min", "earlier_epoch"],
    },
    "budget": {
        # Shared epoch cap; early stopping may finish a run sooner.
        "max_epochs": 200,
        # One planned configuration per model for the initial draft runs.
        "tuning_trials_per_model": 1,
        "run_seeds": [69420, 67, 69],
    },
    "timing": {
        "device": "cuda",
        "batch_size": 64,
        "warmup_steps": 20,
        "measurement_steps": 100,
        "precision": "float32",
        "scope": "forward_only",
    },
}
