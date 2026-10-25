"""S0: pinned ADE20K B0, color off; optimizer defaults await pilot."""
CONFIG = {
    "protocol_file": "protocol.py",
    "model": {
        "name": "segformer", "checkpoint": "nvidia/segformer-b0-finetuned-ade-512-512",
        "revision": "489d5cd81a0b59fab9b7ea758d3548ebe99677da",
    },
    "training": {"optimizer": "adamw", "lr": 0.00006, "weight_decay": 0.01},
    "run": {"name": "S0", "seed": 69420, "color_jitter": False},
}
