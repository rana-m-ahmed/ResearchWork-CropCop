from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.data import CropCopManifestDataset
from cropcop_je.environment import capture_environment, software_stack_identity, validate_locked_core
from cropcop_je.frozen_v1_manifest import load_frozen_v1_rows
from cropcop_je.g1 import validate_dependency_environment, validate_dependency_lock_object
from cropcop_je.hashing import sha256_json
from cropcop_je.models import build_projection_without_state_drift, load_exact_teacher, load_pair_initialization
from cropcop_je.persistence import build_store
from cropcop_je.secondary import (
    CLASS_MAP_SHA256,
    MANIFEST_SHA256,
    S1_PAIR_ID,
    S1_SEED,
    load_baseline_initialization,
    load_json,
    validate_secondary_g1_bundle,
)
from cropcop_je.session import SessionBudget
from cropcop_je.train import run_training

LOCKED_STEPS = {
    "CAL-MNV4-DIRECT": 200,
    "CAL-MNV4-TEACHER": 100,
    "CAL-EFFB0": 100,
    "CAL-CNXTT": 100,
}
QUAL_STEPS = 10
QUAL_RESUME_STEPS = 5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration-id", choices=tuple(LOCKED_STEPS), required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--class-map", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--secondary-g1-bundle", required=True)
    ap.add_argument("--source-git-commit", required=True)
    ap.add_argument("--lane-id", choices=["K1", "K2", "K3"], required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dependency-lock", default="journal_extension/locks/execution_dependency_lock.json")
    ap.add_argument("--row-id-column", required=True)
    ap.add_argument("--path-column", required=True)
    ap.add_argument("--split-column", required=True)
    ap.add_argument("--label-column", required=True)
    ap.add_argument("--class-index-column", default="")
    ap.add_argument("--train-split-value", default="train")
    ap.add_argument("--val-split-value", default="val")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--min-free-gb", type=float, default=5.0)
    ap.add_argument("--durable-store-kind", choices=["filesystem", "kaggle-dataset"], required=True)
    ap.add_argument("--durable-store-locator", required=True)
    ap.add_argument("--durable-required", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    env = capture_environment()
    drift = validate_locked_core(env)
    if drift:
        raise SystemExit(f"locked software identity mismatch: {json.dumps(drift, sort_keys=True)}")
    if not env.get("cuda_available") or int(env.get("cuda_device_count", 0)) != 1:
        raise SystemExit("secondary G2 child requires exactly one visible CUDA device")
    stack_sha = sha256_json(software_stack_identity(env))

    dep_path = Path(args.dependency_lock)
    if not dep_path.is_absolute():
        dep_path = repo / dep_path
    dep = load_json(dep_path)
    dep_errors = validate_dependency_lock_object(dep) + validate_dependency_environment(dep)
    if dep_errors:
        raise SystemExit("dependency lock mismatch: " + "; ".join(dep_errors))

    seal, errors = validate_secondary_g1_bundle(args.secondary_g1_bundle)
    if errors:
        raise SystemExit("secondary G1 invalid: " + "; ".join(errors))
    if seal.get("source_git_sha") != args.source_git_commit:
        raise SystemExit("secondary G1 source differs from calibration source")
    if seal.get("dependency_lock_sha256") != dep.get("dependency_lock_sha256"):
        raise SystemExit("secondary G1 dependency lock differs from calibration host")

    train_rows = load_frozen_v1_rows(
        args.manifest, args.class_map,
        expected_manifest_sha256=MANIFEST_SHA256,
        expected_class_map_sha256=CLASS_MAP_SHA256,
        surface="DS-V1-TRAIN",
        row_id_column=args.row_id_column, path_column=args.path_column,
        split_column=args.split_column, label_column=args.label_column,
        class_index_column=args.class_index_column,
        train_split_value=args.train_split_value, val_split_value=args.val_split_value,
        expected_count=76376,
    )
    val_rows = load_frozen_v1_rows(
        args.manifest, args.class_map,
        expected_manifest_sha256=MANIFEST_SHA256,
        expected_class_map_sha256=CLASS_MAP_SHA256,
        surface="DS-V1-VAL",
        row_id_column=args.row_id_column, path_column=args.path_column,
        split_column=args.split_column, label_column=args.label_column,
        class_index_column=args.class_index_column,
        train_split_value=args.train_split_value, val_split_value=args.val_split_value,
        expected_count=16368,
    )
    train_ds = CropCopManifestDataset(train_rows, args.image_root, training_seed=S1_SEED, train=True)
    val_ds = CropCopManifestDataset(val_rows, args.image_root, training_seed=S1_SEED, train=False)
    ctc = load_json(repo / "journal_extension/configs/common/ctc_v2.json")
    bundle = Path(args.secondary_g1_bundle)
    private = bundle / "private"
    evidence = bundle / "evidence"
    cid = args.calibration_id

    def construct():
        teacher = None
        projection = None
        teacher_sha = None
        teacher_factory_bundle_sha = None
        if cid.startswith("CAL-MNV4"):
            pair = seal["mnv4_s1"]
            student, payload = load_pair_initialization(private / pair["student_init_basename"], expected_sha256=pair["student_init_sha256"], pair_id=S1_PAIR_ID, seed=S1_SEED)
            if payload.get("pretrained_sha256") != pair["pretrained_sha256"]:
                raise RuntimeError("calibration MNV4 initialization/pretrained mismatch")
            objective = {"ce": 1.0, "kd": 0.0, "feature": 0.0}
            init_sha = pair["student_init_sha256"]
            pretrain_sha = pair["pretrained_sha256"]
            model_family = "mnv4"
            if cid == "CAL-MNV4-TEACHER":
                t = seal["teacher"]
                teacher, identity = load_exact_teacher(private / t["artifact_basename"], factory_spec=t["factory_entrypoint"], factory_bundle_manifest=evidence / "TEACHER_FACTORY_BUNDLE.json", repo_root=repo, factory_source_root=repo / "journal_extension/teacher_factory")
                teacher_sha = t["checkpoint_sha256"]
                teacher_factory_bundle_sha = identity["bundle_sha256"]
                projection = build_projection_without_state_drift(student, teacher, seed=S1_SEED)
                objective = {"ce": 0.5, "kd": 0.35, "feature": 0.15}
        elif cid == "CAL-EFFB0":
            row = seal["baselines"]["effb0"]
            student, payload = load_baseline_initialization(private / row["init_basename"], expected_sha256=row["init_sha256"], model_key="effb0", seed=S1_SEED, experiment_id="R06-EFFB0-CONTEXT-S1")
            init_sha = row["init_sha256"]
            pretrain_sha = row["pretrained_sha256"]
            objective = {"ce": 1.0, "kd": 0.0, "feature": 0.0}
            model_family = "effb0"
        else:
            row = seal["baselines"]["cnxtt"]
            student, payload = load_baseline_initialization(private / row["init_basename"], expected_sha256=row["init_sha256"], model_key="cnxtt", seed=S1_SEED, experiment_id="R07-CNXTT-CONTEXT-S1")
            init_sha = row["init_sha256"]
            pretrain_sha = row["pretrained_sha256"]
            objective = {"ce": 1.0, "kd": 0.0, "feature": 0.0}
            model_family = "cnxtt"
        return student, teacher, projection, objective, init_sha, pretrain_sha, teacher_sha, teacher_factory_bundle_sha, model_family

    def identity(name: str, init_sha: str, pretrain_sha: str, teacher_sha, teacher_factory_bundle_sha, model_family: str):
        return {
            "run_id": name, "experiment_id": cid, "authority_id": seal["authority"]["id"],
            "source_git_commit": args.source_git_commit, "lane_id": args.lane_id,
            "config_sha256": sha256_json({"calibration_id": cid, "model_family": model_family, "seed": S1_SEED}),
            "ctc_v2_sha256": sha256_json(ctc), "manifest_sha256": MANIFEST_SHA256,
            "class_map_sha256": CLASS_MAP_SHA256, "seed": S1_SEED,
            "student_init_sha256": init_sha, "pretrained_sha256": pretrain_sha,
            "teacher_sha256": teacher_sha, "teacher_factory_sha256": None,
            "teacher_factory_bundle_sha256": teacher_factory_bundle_sha,
            "software_stack_sha256": stack_sha, "dependency_lock_sha256": dep["dependency_lock_sha256"],
            "g1_seal_sha256": seal["secondary_g1_seal_sha256"], "g2_barrier_sha256": None,
            "allowed_surfaces": ["DS-V1-TRAIN", "DS-V1-VAL"],
        }

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    def run_piece(name: str, steps: int, resume: bool):
        student, teacher, projection, objective, init_sha, pretrain_sha, teacher_sha, teacher_factory_bundle_sha, family = construct()
        rid = identity(name, init_sha, pretrain_sha, teacher_sha, teacher_factory_bundle_sha, family)
        return run_training(student=student, teacher=teacher, projection=projection, train_dataset=train_ds, val_dataset=val_ds, ctc=ctc, objective=objective, run_identity=rid, output_dir=root / name, num_workers=args.num_workers, resume=resume, max_optimizer_steps=steps, validation_enabled=False, checkpoint_every_steps=50, session_budget=SessionBudget.from_environment(require_global_clock=True), min_free_bytes=int(args.min_free_gb * 1024**3))

    if not args.durable_required:
        raise SystemExit("secondary G2 requires real durable checkpoint roundtrip")
    store = build_store(args.durable_store_kind, args.durable_store_locator)
    qual_name = f"{cid}-RESUME-QUAL"
    first = run_piece(qual_name, QUAL_STEPS, False)
    sync_started = time.perf_counter()
    sync = store.sync(root / qual_name, run_id=qual_name, segment_id="Q1")
    sync_seconds = time.perf_counter() - sync_started
    shutil.rmtree(root / qual_name)
    restore_started = time.perf_counter()
    restored = store.restore(root / qual_name, run_id=qual_name)
    restore_seconds = time.perf_counter() - restore_started
    if not restored:
        raise SystemExit("durable roundtrip restore returned no checkpoint generation")
    second = run_piece(qual_name, QUAL_RESUME_STEPS, True)
    resume_success = second.get("optimizer_steps_segment") == QUAL_RESUME_STEPS and float(second.get("checkpoint_load_seconds", 0.0) or 0.0) > 0.0
    if not resume_success:
        raise SystemExit("secondary durable save→restore→resume qualification failed")

    measured = run_piece(cid, LOCKED_STEPS[cid], False)
    if measured.get("optimizer_steps_segment") != LOCKED_STEPS[cid]:
        raise SystemExit("secondary measured calibration did not complete locked optimizer steps")
    measured["checkpoint_load_seconds"] = second["checkpoint_load_seconds"]
    measured["durable_sync_seconds"] = sync_seconds
    measured["durable_restore_seconds"] = restore_seconds
    measured["accelerator"] = (env.get("torch_cuda_devices") or [{}])[0].get("name")
    measured["cuda_driver_identity"] = env.get("nvidia_smi")

    teacher_sha = seal["teacher"]["checkpoint_sha256"] if cid == "CAL-MNV4-TEACHER" else None
    teacher_factory_bundle_sha = seal["teacher"]["factory_bundle_sha256"] if cid == "CAL-MNV4-TEACHER" else None
    coverage = {
        "CAL-MNV4-DIRECT": [],
        "CAL-MNV4-TEACHER": ["R12-MNV4-LOGITS-S1", "R12-MNV4-FEATURE-S1"],
        "CAL-EFFB0": ["R06-EFFB0-CONTEXT-S1"],
        "CAL-CNXTT": ["R07-CNXTT-CONTEXT-S1"],
    }[cid]
    summary = {
        "schema_version": "4.0", "status": "PASS", "calibration_id": cid,
        "locked_optimizer_steps": LOCKED_STEPS[cid], "resume_success": True,
        "durable_roundtrip_success": True, "calibration_weights_scientific": False,
        "source_git_commit": args.source_git_commit, "software_stack_sha256": stack_sha,
        "g1_seal_sha256": seal["secondary_g1_seal_sha256"], "dependency_lock_sha256": dep["dependency_lock_sha256"],
        "mnv4_pretrained_sha256": seal["mnv4_s1"]["pretrained_sha256"],
        "effb0_pretrained_sha256": seal["baselines"]["effb0"]["pretrained_sha256"],
        "cnxtt_pretrained_sha256": seal["baselines"]["cnxtt"]["pretrained_sha256"],
        "teacher_checkpoint_sha256": teacher_sha, "teacher_factory_bundle_sha256": teacher_factory_bundle_sha,
        "visible_cuda_device_count": int(env.get("cuda_device_count", 0)),
        "visible_gpu_name": (env.get("torch_cuda_devices") or [{}])[0].get("name"),
        "git_credentials_present": bool(__import__("os").environ.get("CROPCOP_GITHUB_TOKEN") or __import__("os").environ.get("GITHUB_TOKEN")),
        "allowed_surfaces": ["DS-V1-TRAIN", "DS-V1-VAL"], "v1_test_accessed": False,
        "external_protected_surface_accessed": False, "coverage": coverage,
        "durability": {
            "kind": args.durable_store_kind, "locator": args.durable_store_locator,
            "sync_status": sync.status, "sync_seconds": sync_seconds, "restore_seconds": restore_seconds,
            "restored_from_fresh_local_directory": True,
        },
        "measured": measured, "forecast_is_scheduling_only": True,
    }
    atomic_write_json(root / "calibration_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
