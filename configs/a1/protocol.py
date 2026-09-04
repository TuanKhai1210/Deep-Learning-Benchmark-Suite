"""Shared A1 policy. Thiên/Khoa/Khải agreed the four run seeds below.

Other experimental choices remain proposals until reviewed and measured.
Use one literal CONFIG dictionary: no imports, functions or computations.
"""

CONFIG = {
    "protocol": {
        "id": "a1-v0",
        "status": "draft",
        "approved_by": [],
    },
    "data": {
        "dataset": "fashion_mnist",
        "root": "data",
        # Separate split seed, retained as a proposal; never resplit per run.
        "split_seed": 42,
        "train_size": 50000,
        "validation_size": 10000,
        "test_size": 10000,
        "stratified": True,
        "split_file": "configs/a1/splits/fashion_mnist_seed42.json",
    },
    "preprocessing": {
        "image_size": [28, 28],
        "channels": 1,
        # TODO A (Thiên): measure on the unaugmented 50k TRAIN partition.
        "mean": [],
        "std": [],
        "augmentation": "random_crop",
        "crop_padding": 2,
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
        # TODO team: 0 means UNDECIDED, never a completed zero-epoch run.
        "max_epochs": 0,
        "tuning_trials_per_model": 0,
        "run_seeds": [36, 69420, 67, 69],
    },
    "timing": {
        "device": "TBD",
        "batch_size": 0,
        "warmup_steps": 20,
        "measurement_steps": 100,
        "precision": "float32",
        "scope": "forward_only",
    },
}
