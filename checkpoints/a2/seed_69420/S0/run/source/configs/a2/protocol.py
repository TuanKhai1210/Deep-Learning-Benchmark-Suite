"""Executable INITIAL defaults; not pilot-confirmed or frozen for main runs."""
CONFIG = {
    "protocol": {
        "version": "a2-initial-v1", "status": "draft",
        "approval_reference": None, "pilot_reference": None,
    },
    "data": {
        "split_seed": 36,
        "manifest_sha256": "87905f1a97dd2c04194e049ca61d1a7f3bbe2942e9fe17dc2d737c371cdce943",
        "num_classes": 17, "ignore_id": 255,
        "converter": "pillow_native_polygon_union_crossclass_ignore_v1",
        "pillow_version": "12.3.0",
    },
    "preprocessing": {
        "size": 512, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    },
    "loss": {"ce_weight": 1.0, "dice_weight": 1.0},
    "budget": {"batch_size": 4, "epochs": 40, "patience": 8},
}
