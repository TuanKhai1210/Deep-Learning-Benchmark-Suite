"""Exact mismatch diagnosis and provenance-preserving host-only continuation.

Original run/checkpoint identities are never edited. The pinned trainer still
uses its strict identity check on every resumed run, including derived copies.
"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
import uuid

HOST_ONLY_FIELDS = frozenset({
    "environment.os_release", "environment.processor",
    "environment.system_ram.total_bytes", "environment.system_ram.total_gb",
})


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def differences(previous, current, prefix=""):
    result = []
    if isinstance(previous, dict) and isinstance(current, dict):
        for key in sorted(set(previous) | set(current)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in previous or key not in current:
                result.append({"field": path, "saved": previous.get(key, "<missing>"),
                               "current": current.get(key, "<missing>")})
            else:
                result.extend(differences(previous[key], current[key], path))
    elif type(previous) is not type(current) or previous != current:
        result.append({"field": prefix, "saved": previous, "current": current})
    return result


def mismatch_context(error):
    """Read the actual failed fit frame; never approximate its fingerprint."""
    trace = error.__traceback__
    while trace:
        values = trace.tb_frame.f_locals
        if (trace.tb_frame.f_code.co_name == "fit"
                and {"identity", "metadata", "fingerprint", "output"} <= set(values.keys())):
            previous = deepcopy(values["metadata"])
            current = deepcopy(values["identity"])
            fingerprint = values["fingerprint"]
            if digest(previous["identity"]) != previous["identity_sha256"]:
                raise ValueError("Original metadata identity hash is invalid")
            if digest(current) != fingerprint:
                raise ValueError("Current fit identity hash is invalid")
            return {"metadata": previous, "identity": current, "fingerprint": fingerprint,
                    "run": Path(values["output"]).resolve(),
                    "source": deepcopy(values.get("source", previous["source"]))}
        trace = trace.tb_next
    raise ValueError("Cannot find the exact failed fit identity frame; rerun the original failing cell")


def origin_root(root):
    root = Path(root).resolve()
    provenance = root / "continuation.json"
    if not provenance.is_file():
        return root
    value = json.loads(provenance.read_text(encoding="utf-8"))
    origin = Path(value["origin_root"]).resolve()
    if not root.is_relative_to(origin / "continuations"):
        raise ValueError("Continuation origin path is invalid")
    return origin


def resolve_target(origin):
    origin = Path(origin).resolve()
    pointer = origin / "resume_pointer.json"
    if not pointer.is_file():
        return origin
    value = json.loads(pointer.read_text(encoding="utf-8"))
    target = (origin / value["target_relative"]).resolve()
    if not target.is_relative_to(origin / "continuations"):
        raise ValueError("Resume pointer escapes this run's continuation directory")
    metadata = json.loads((target / "run/metadata.json").read_text(encoding="utf-8"))
    if digest(metadata["identity"]) != metadata["identity_sha256"]:
        raise ValueError("Continuation metadata identity hash is invalid")
    if metadata["identity_sha256"] != value["identity_sha256"]:
        raise ValueError("Resume pointer identity mismatch")
    if not (target / "run/last.pt").is_file():
        raise ValueError("Continuation pointer has no epoch checkpoint")
    return target


def fork_host_continuation(context, origin, changes):
    import torch
    origin = Path(origin).resolve()
    source_run = context["run"]
    if not source_run.is_relative_to(origin) or source_run.name != "run":
        raise ValueError("Source run is outside the selected experiment")
    if not changes or any(item["field"] not in HOST_ONLY_FIELDS for item in changes):
        raise ValueError("Continuation permits only RAM/kernel/processor descriptor changes")
    old = context["metadata"]
    current = context["identity"]
    fingerprint = context["fingerprint"]
    actual = differences(old["identity"], current)
    if changes != actual:
        raise ValueError("Mismatch report does not match the identities")
    checkpoint = torch.load(source_run / "last.pt", map_location="cpu", weights_only=False)
    if checkpoint["identity_sha256"] != old["identity_sha256"]:
        raise ValueError("Original checkpoint identity mismatch")
    if checkpoint["epoch"] < 1 or len(checkpoint["history"]) != checkpoint["epoch"]:
        raise ValueError("Original epoch/history is inconsistent")
    best = checkpoint["best"]
    if best is None:
        raise ValueError("Original checkpoint has no selected validation checkpoint")
    # Recover the best copy from the authoritative last state if its write was
    # interrupted after last.pt was saved. Never write into the parent run.
    if best["epoch"] == checkpoint["epoch"]:
        best_checkpoint = checkpoint
    else:
        best_checkpoint = torch.load(source_run / "best.pt", map_location="cpu", weights_only=False)
        if (best_checkpoint["identity_sha256"] != old["identity_sha256"]
                or best_checkpoint["epoch"] != best["epoch"]):
            raise ValueError("Original best checkpoint disagrees with last.pt")
    inputs = {name: file_hash(source_run / name) for name in
              ("metadata.json", "last.pt", "best.pt", "effective_config.json")
              if (source_run / name).is_file()}
    if json.loads((source_run / "effective_config.json").read_text(encoding="utf-8")) != current["config"]:
        raise ValueError("Effective config does not match the checkpoint")
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    target = origin / "continuations" / suffix
    run = target / "run"
    run.mkdir(parents=True, exist_ok=False)
    for name in ("setup.json", "frozen_config.json", "pilot.json", "manifest.json",
                 "data_checks.json", "data_overlay.png", "helper_snapshot.py"):
        path = origin / name
        if path.is_file():
            shutil.copy2(path, target / name)
    if (source_run / "source").is_dir():
        shutil.copytree(source_run / "source", run / "source")
    record = {"kind": "same_experiment_host_continuation", "origin_root": str(origin),
              "parent_run": str(source_run), "parent_run_id": old.get("run_id"),
              "parent_identity_sha256": old["identity_sha256"],
              "current_identity_sha256": fingerprint, "differences": changes,
              "resumed_epoch": checkpoint["epoch"], "original_input_sha256": inputs,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "recovery_source_sha256": file_hash(Path(__file__)),
              "note": "Same config/source/data/GPU/libraries; host descriptors differ. Not a new seed or independent run; no bitwise GPU claim."}
    metadata = deepcopy(old)
    metadata.update(run_id=str(uuid.uuid4()), identity=current, identity_sha256=fingerprint,
                    environment=current["environment"], source=context["source"], continuation=record)
    metadata["experiment_id"] = old.get("experiment_id", old.get("run_id"))
    write_json(run / "metadata.json", metadata)
    write_json(run / "effective_config.json", current["config"])
    for name, state in (("last.pt", checkpoint), ("best.pt", best_checkpoint)):
        copied = dict(state)
        copied["identity_sha256"] = fingerprint  # Only the derived copy changes identity.
        torch.save(copied, run / name)
    write_json(run / "history.json", checkpoint["history"])
    write_json(run / "metrics.json", {"best": best, "epochs_completed": checkpoint["epoch"],
                                      "kind": metadata["kind"]})
    write_json(target / "continuation.json", record)
    shutil.copy2(Path(__file__), target / "resume_recovery_snapshot.py")
    for name, previous_hash in inputs.items():
        if file_hash(source_run / name) != previous_hash:
            raise ValueError("Original input changed during copying; continuation not published")
    # Publish a routing pointer only after all derived checkpoint files exist.
    write_json(origin / "resume_pointer.json", {"target_relative": target.relative_to(origin).as_posix(),
               "identity_sha256": fingerprint})
    return target


def train_with_recovery(config, train, validation, run_root, setup, *, runner=None):
    """Keep strict resume; branch only when exact drift is entirely descriptive."""
    root = origin_root(run_root)
    target = resolve_target(root)
    invoke = runner or setup.train_or_resume
    setup.install_progress(target)
    try:
        return target, invoke(config, train, validation, target / "run")
    except ValueError as error:
        if str(error) != "Resume config/source/device identity mismatch":
            raise
        context = mismatch_context(error)
        changes = differences(context["metadata"]["identity"], context["identity"])
        report = {"differences": changes, "host_only": bool(changes) and all(
                  item["field"] in HOST_ONLY_FIELDS for item in changes)}
        write_json(root / "resume_diagnostic.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if not report["host_only"]:
            raise ValueError("Resume stopped: numerical/config/source/data identity differs. Send resume_diagnostic.json; all checkpoints preserved.") from error
        target = fork_host_continuation(context, root, changes)
        print("Continuing the same experiment from the saved epoch in:", target, flush=True)
        setup.install_progress(target)
        return target, invoke(config, train, validation, target / "run")


def launch(namespace):
    required = ("setup", "CONFIG", "train_data", "validation_data", "RUN_ROOT")
    if any(name not in namespace for name in required):
        raise ValueError("Run notebook setup/data/pilot cells first, then this recovery cell")
    if not namespace.get("RUN_MAIN", False):
        raise ValueError("RUN_MAIN=False; enable it before resuming training")
    target, result = train_with_recovery(namespace["CONFIG"], namespace["train_data"],
                    namespace["validation_data"], namespace["RUN_ROOT"], namespace["setup"])
    namespace["RUN_ROOT"] = target
    namespace["outcome"] = result
    namespace["setup"].write_json(target / "outcome.json", result)
    print(result)
    print("Then run the original notebook's best-validation/curves/ZIP cell.")
