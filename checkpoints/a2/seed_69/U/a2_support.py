"""Colab orchestration around pinned A2 modules; no second epoch trainer.

This helper is embedded in the notebook. Dataset and main training are performed
only when the user executes that notebook, never at helper import.
"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import gc
import hashlib
import json
import os
import shutil
import stat
import time
import uuid
import zipfile

CODE_REVISION = "7df323698d1ae4d5122850a54eb38b01609d80f1"
DATA_FILE_ID = "1a7C5WOkM5tUqsov0pIaYKTOjp0b5AitN"
DATA_BYTES = 553029970
DATA_SHA256 = "caf93d992b85343cfc500386c94ca9ffa6c63e2284e04127753a40db8f33ba4c"
APPROVAL = "User confirmed instructor approval on 2026-10-26 (Asia/Bangkok)"
CUDA_POLICY = {"deterministic_algorithms": True, "warn_only": True,
               "reason": "CUDA bilinear interpolation backward has no strict deterministic implementation"}


def prepare_base_config(config):
    config = deepcopy(config)
    if config["model"]["name"] != "unet":
        raise ValueError("This setup is specifically for U-Net")
    config["protocol"]["cuda_determinism"] = deepcopy(CUDA_POLICY)
    return config


def seed_for_unet(seed, *, deterministic=True):
    """Keep shared seeds; allow documented CUDA interpolation warnings explicitly."""
    import torch
    from dlbench.common.reproducibility import seed_everything
    seed_everything(seed, deterministic=deterministic)
    torch.use_deterministic_algorithms(deterministic, warn_only=deterministic)


def install_runtime_policy(config):
    import torch
    import dlbench.a2.trainer as trainer
    if config["protocol"].get("cuda_determinism") != CUDA_POLICY:
        raise ValueError("Declare the U-Net CUDA warning policy in the effective config")
    torch.set_default_dtype(torch.float32)
    # The core trainer resets seeds internally; this explicit adapter survives
    # that reset. Effective config and helper snapshot/hash record the policy.
    trainer.seed_everything = seed_for_unet


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def acquire_archive(path, downloader, *, expected_sha=DATA_SHA256, expected_size=DATA_BYTES):
    """Verified cache; failed downloads never replace the authoritative archive."""
    path = Path(path)
    if path.exists():
        if path.stat().st_size != expected_size or sha256_file(path) != expected_sha:
            raise ValueError("Existing archive differs from expected EDA source; preserve and inspect it")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial-" + uuid.uuid4().hex)
    downloader(temporary)
    if temporary.stat().st_size != expected_size or sha256_file(temporary) != expected_sha:
        raise ValueError(f"Downloaded bytes differ from EDA source; partial retained: {temporary}")
    os.replace(temporary, path)
    return path


def download_from_drive(destination):
    """Try public Drive download, then authenticated read-only Drive API in Colab."""
    import gdown
    try:
        result = gdown.download(id=DATA_FILE_ID, output=str(destination), quiet=False)
        if result and Path(destination).stat().st_size == DATA_BYTES:
            return
    except Exception as error:
        print("Public Drive download unavailable:", type(error).__name__)
    print("Using authenticated Drive read access; choose the account that can open the team folder.")
    from google.colab import auth
    import google.auth
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    auth.authenticate_user()
    credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/drive.readonly"])
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    request = service.files().get_media(fileId=DATA_FILE_ID, supportsAllDrives=True)
    with Path(destination).open("wb") as stream:
        download = MediaIoBaseDownload(stream, request, chunksize=16 * 1024 * 1024)
        done = False
        while not done:
            progress, done = download.next_chunk(num_retries=3)
            if progress:
                print(f"Dataset download: {progress.progress():.0%}")


def extract_material(archive_path, destination, *, expected_sha=DATA_SHA256):
    """Extract only material_version + license/readme; preserve original source paths."""
    archive_path, destination = Path(archive_path), Path(destination)
    if sha256_file(archive_path) != expected_sha:
        raise ValueError("Archive hash mismatch before extraction")
    marker = destination / "extraction.json"
    if destination.exists():
        if not marker.is_file():
            raise ValueError("Incomplete local extraction exists; use a fresh extraction directory")
        record = json.loads(marker.read_text(encoding="utf-8"))
        if record["archive_sha256"] != expected_sha:
            raise ValueError("Extraction cache identity mismatch")
        root = destination / record["material_root"]
        if not (root / "instances_val_trashcan.json").is_file():
            raise ValueError("Extraction cache is incomplete")
        return root
    temporary = destination.with_name(destination.name + ".partial-" + uuid.uuid4().hex)
    temporary.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive_path) as archive:
        candidates = []
        for info in archive.infolist():
            name = info.filename.replace("\\", "/")
            parts = PurePosixPath(name)
            if parts.is_absolute() or ".." in parts.parts or any(":" in p for p in parts.parts):
                raise ValueError("Unsafe archive member")
            if stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK:
                raise ValueError("Archive symlinks are unsupported")
            if parts.name == "instances_train_trashcan.json" and "material_version" in parts.parts:
                candidates.append(parts.parent)
        if len(candidates) != 1:
            raise ValueError("Expected exactly one material_version annotation directory")
        prefix = candidates[0]
        extracted = 0
        for info in archive.infolist():
            path = PurePosixPath(info.filename.replace("\\", "/"))
            selected = path.is_relative_to(prefix) or path.name.lower().startswith(("license", "readme"))
            if not selected or info.is_dir():
                continue
            target = temporary / str(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)  # ZIP CRC is checked on complete read.
            extracted += 1
    if not (temporary / str(prefix) / "instances_val_trashcan.json").is_file():
        raise ValueError("Missing material validation annotations")
    write_json(temporary / "extraction.json", {"archive_sha256": expected_sha,
               "material_root": str(prefix), "files": extracted})
    os.replace(temporary, destination)
    return destination / str(prefix)


def make_datasets(root, manifest, config):
    from dlbench.a2.data import TrashCanDataset
    args = dict(seed=config["run"]["seed"], color_jitter=False)
    train = TrashCanDataset(root, manifest, "train", config["preprocessing"], **args)
    validation = TrashCanDataset(root, manifest, "validation", config["preprocessing"], **args)
    return train, validation


def verify_training_data(train, validation):
    """All train/validation native masks and RGB dimensions, with no test evaluation."""
    from PIL import Image
    from dlbench.a2.converter import rasterize_semantic
    from tqdm.auto import tqdm
    checked = {}
    for dataset in (train, validation):
        for row in tqdm(dataset.rows, desc=f"Verify {dataset.split}"):
            record = dataset.records[row["image_id"]]
            with Image.open(dataset.root / row["relative_path"]) as image:
                if image.size != (record["width"], record["height"]):
                    raise ValueError(f"RGB dimension mismatch: {row['image_id']}")
                image.verify()
            target = rasterize_semantic(record)[0]
            if hashlib.sha256(target.tobytes()).hexdigest() != row["target_sha256"]:
                raise ValueError(f"Native target mismatch: {row['image_id']}")
        checked[dataset.split] = len(dataset)
    return {"checked": checked, "manifest_sha256": train.manifest_digest,
            "test_evaluated": False}


def pilot(config, dataset, *, steps=20, device="cuda", require_cuda=True):
    """Fixed real batch diagnostic and train-step/VRAM pilot, discarded before main."""
    import torch
    from dlbench.a2.data import make_loader
    from dlbench.a2.models import build_model, count_parameters
    from dlbench.a2.losses import SegmentationLoss
    from dlbench.common.reproducibility import collect_environment
    if steps < 2:
        raise ValueError("Pilot needs at least two optimization steps")
    if dataset.split != "train":
        raise ValueError("Optimize the pilot on train only")
    if require_cuda and (device != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("Select a Colab GPU runtime before the pilot/main run")
    if config["protocol"].get("cuda_determinism") != CUDA_POLICY:
        raise ValueError("Pilot config must declare the CUDA warning policy")
    seed_for_unet(config["run"]["seed"])
    loader = make_loader(dataset, batch_size=config["budget"]["batch_size"], workers=0)
    batch = next(iter(loader))
    images, targets = batch["image"].to(device), batch["mask"].to(device)
    if not (targets != 255).any():
        raise ValueError("Pilot batch contains no supervised pixels")
    model = build_model(config["model"]).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"],
                                 weight_decay=config["training"]["weight_decay"])
    criterion = SegmentationLoss(**config["loss"])
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    losses, durations = [], []
    for step in range(steps):
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if logits.shape != (images.shape[0], 17, *targets.shape[1:]):
            raise ValueError("Pilot logits contract mismatch")
        loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Pilot loss is non-finite")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("Pilot gradient is non-finite")
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
        losses.append(float(loss.detach()))
        if step in (0, steps - 1) or (step + 1) % 5 == 0:
            print(f"Pilot {step + 1}/{steps}: loss={losses[-1]:.4f}", flush=True)
    total = torch.cuda.get_device_properties(0).total_memory if device == "cuda" else None
    reserved = torch.cuda.max_memory_reserved() if device == "cuda" else None
    passed = losses[-1] < losses[0] and (reserved is None or reserved < 0.9 * total)
    result = {"kind": "real_batch_gpu_pilot" if device == "cuda" else "cpu_development_check",
              "device": device, "passed": passed, "config_sha256": digest(config),
              "code_revision": CODE_REVISION, "steps": steps,
              "cuda_determinism": deepcopy(CUDA_POLICY),
              "batch_size": int(images.shape[0]), "canvas": list(targets.shape[-2:]),
              "image_ids": [int(v) for v in batch["image_id"]],
              "losses": losses, "train_step_seconds_excluding_io": durations[2:],
              "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
              "peak_reserved_bytes": reserved, "gpu_total_bytes": total,
              "environment": collect_environment(), "parameter_counts": count_parameters(model),
              "note": "Fixed-batch loss decrease and capacity check, not full-data overfit or benchmark quality"}
    del model, optimizer, criterion, batch, images, targets, logits, loss
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def freeze_config(base, pilot_record, *, approval=APPROVAL):
    from dlbench.a2.config import validate_config
    if not approval:
        raise ValueError("Record instructor approval before main")
    if base["protocol"].get("cuda_determinism") != CUDA_POLICY:
        raise ValueError("Missing CUDA determinism policy")
    if (not pilot_record["passed"] or pilot_record["device"] != "cuda"
            or pilot_record["code_revision"] != CODE_REVISION
            or pilot_record.get("cuda_determinism") != CUDA_POLICY
            or pilot_record["config_sha256"] != digest(base)
            or pilot_record["batch_size"] != base["budget"]["batch_size"]):
        raise ValueError("Matching successful GPU pilot required before main")
    result = deepcopy(base)
    result["protocol"].update(status="frozen", approval_reference=approval,
                              pilot_reference="sha256:" + digest(pilot_record))
    validate_config(result, main=True)
    return result


def pin_setup(path, base, pilot_record, helper_sha):
    """Store settings once; reject changed options rather than silently overwrite."""
    path = Path(path)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if digest(saved["payload"]) != saved["sha256"]:
            raise ValueError("Setup ledger hash mismatch")
        value = saved["payload"]
        if value["base_sha256"] != digest(base) or value["helper_sha256"] != helper_sha:
            raise ValueError("Settings/helper changed; use a new RUN_TAG, preserve the existing run")
        if value["config"] != freeze_config(base, value["pilot"]):
            raise ValueError("Frozen config does not match pilot/approval")
        return value
    if pilot_record is None:
        raise ValueError("A fresh setup needs its GPU pilot")
    value = {"created_utc": datetime.now(timezone.utc).isoformat(), "code_revision": CODE_REVISION,
             "base_sha256": digest(base), "helper_sha256": helper_sha,
             "pilot": pilot_record, "config": freeze_config(base, pilot_record)}
    write_json(path, {"payload": value, "sha256": digest(value)})
    return value


def install_progress(run_root):
    """Decorate the shared engine for visibility; core checkpoint/training is unchanged."""
    import dlbench.a2.trainer as trainer
    from dlbench.a2.engine import run_epoch
    from tqdm.auto import tqdm
    journal = Path(run_root) / "epoch_progress.jsonl"
    def visible_epoch(model, loader, criterion, device, optimizer=None):
        phase = "train" if optimizer is not None else "validation"
        started = time.perf_counter()
        result = run_epoch(model, tqdm(loader, desc=phase), criterion, device, optimizer)
        elapsed = time.perf_counter() - started
        print(f"{phase}: loss={result['loss']:.4f}, fg mIoU={result['foreground_miou']}, seconds={elapsed:.1f}", flush=True)
        with journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"phase": phase, "seconds": elapsed, "loss": result["loss"],
                                     "foreground_miou": result["foreground_miou"]}) + "\n")
        return result
    trainer.run_epoch = visible_epoch


def train_or_resume(config, train, validation, output, *, device="cuda"):
    from dlbench.a2.trainer import fit
    output = Path(output)
    resume = output.exists()
    if resume and not all((output / name).is_file() for name in ("metadata.json", "last.pt")):
        raise ValueError("Run directory has no complete epoch checkpoint; preserve it and use a new RUN_TAG")
    install_runtime_policy(config)
    # The core trainer enforces exact config/source/device/environment identity.
    return fit(config, train, validation, output, device=device, workers=0, resume=resume)
