"""S1: paired with S0; only run name and train color augmentation differ."""
CONFIG = {
    "protocol_file": "protocol.py",
    "model": {
        "name": "segformer", "checkpoint": "nvidia/segformer-b0-finetuned-ade-512-512",
        "revision": "489d5cd81a0b59fab9b7ea758d3548ebe99677da",
    },
    "training": {"optimizer": "adamw", "lr": 0.00006, "weight_decay": 0.01},
    "run": {"name": "S1", "seed": 69420, "color_jitter": True},
}
