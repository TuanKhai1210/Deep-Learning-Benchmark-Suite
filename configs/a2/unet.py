"""U: scratch U-Net, development optimizer defaults."""
CONFIG = {
    "protocol_file": "protocol.py",
    "model": {"name": "unet", "base_channels": 16},
    "training": {"optimizer": "adamw", "lr": 0.001, "weight_decay": 0.01},
    "run": {"name": "U", "seed": 69420, "color_jitter": False},
}
