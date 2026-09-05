from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.data import CropCopManifestDataset, ManifestColumns, load_manifest_rows
from cropcop_je.environment import capture_environment, software_stack_identity, validate_locked_core
from cropcop_je.g1 import (
    CLASS_MAP_SHA256, MANIFEST_SHA256, validate_dependency_environment,
    validate_dependency_lock_object, validate_g1_seal_object,
)
from cropcop_je.hashing import require_sha256, sha256_file, sha256_json
from cropcop_je.models import create_convnext_tiny_from_pretrained
from cropcop_je.session import SessionBudget
from cropcop_je.train import run_training

SEED = 21270083
CAL_STEPS = 100
QUAL_STEPS = 10
QUAL_RESUME_STEPS = 5


def _env():
    env = capture_environment()
    drift = validate_locked_core(env)
    if drift:
        raise RuntimeError(f"locked software identity mismatch: {json.dumps(drift, sort_keys=True)}")
    if env.get("cuda_available") is not True:
        raise RuntimeError("CUDA unavailable")
    return env, sha256_json(software_stack_identity(env))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--class-map", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--source-git-commit", required=True)
    ap.add_argument("--g1-seal", required=True)
    ap.add_argument("--dependency-lock", default="journal_extension/locks/execution_dependency_lock.json")
    ap.add_argument("--lane-id", default="K3", choices=["K3"])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--row-id-column", required=True)
    ap.add_argument("--path-column", required=True)
    ap.add_argument("--split-column", required=True)
    ap.add_argument("--class-index-column", required=True)
    ap.add_argument("--train-split-value", default="train")
    ap.add_argument("--val-split-value", default="val")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--min-free-gb", type=float, default=5.0)
    args = ap.parse_args()

    env, stack_sha = _env()
    repo = Path(args.repo_root)
    dep_path = Path(args.dependency_lock)
    if not dep_path.is_absolute():
        dep_path = repo / dep_path
    dep = json.loads(dep_path.read_text(encoding="utf-8"))
    dep_errors = validate_dependency_lock_object(dep) + validate_dependency_environment(dep)
    if dep_errors:
        raise SystemExit("dependency lock mismatch: " + "; ".join(dep_errors))
    seal = json.loads(Path(args.g1_seal).read_text(encoding="utf-8"))
    seal_errors = validate_g1_seal_object(seal)
    if seal_errors:
        raise SystemExit("G1 seal invalid: " + "; ".join(seal_errors))
    if seal.get("source_git_sha") != args.source_git_commit:
        raise SystemExit("G1 source SHA differs from CAL-CNXTT source")
    if seal.get("dependency_lock_sha256") != dep.get("dependency_lock_sha256"):
        raise SystemExit("G1 dependency lock differs from CAL-CNXTT environment")

    require_sha256(args.manifest, MANIFEST_SHA256, "V1 manifest")
    require_sha256(args.class_map, CLASS_MAP_SHA256, "class map")
    cnxtt_pretrained_sha = sha256_file(args.pretrained)
    ctc_path = repo / "journal_extension/configs/common/ctc_v2.json"
    ctc = json.loads(ctc_path.read_text(encoding="utf-8"))
    cols = ManifestColumns(args.row_id_column, args.path_column, args.split_column, args.class_index_column)
    train_rows = load_manifest_rows(
        args.manifest, expected_sha256=MANIFEST_SHA256, surface="DS-V1-TRAIN", columns=cols,
        train_split_value=args.train_split_value, val_split_value=args.val_split_value, expected_count=76376
    )
    val_rows = load_manifest_rows(
        args.manifest, expected_sha256=MANIFEST_SHA256, surface="DS-V1-VAL", columns=cols,
        train_split_value=args.train_split_value, val_split_value=args.val_split_value, expected_count=16368
    )
    train_ds = CropCopManifestDataset(train_rows, args.image_root, training_seed=SEED, train=True)
    val_ds = CropCopManifestDataset(val_rows, args.image_root, training_seed=SEED, train=False)
    base_identity = {
        "run_id": "CAL-CNXTT",
        "experiment_id": "CAL-CNXTT",
        "authority_id": "EAAI-JE-SDL-v2.1-QA",
        "source_git_commit": args.source_git_commit,
        "lane_id": "K3",
        "config_sha256": sha256_json({"calibration_id": "CAL-CNXTT", "seed": SEED, "model": "torchvision.convnext_tiny"}),
        "ctc_v2_sha256": sha256_json(ctc),
        "manifest_sha256": MANIFEST_SHA256,
        "class_map_sha256": CLASS_MAP_SHA256,
        "seed": SEED,
        "student_init_sha256": sha256_json({"pretrained_sha256": cnxtt_pretrained_sha, "seed": SEED, "head_classes": 120}),
        "pretrained_sha256": cnxtt_pretrained_sha,
        "teacher_sha256": None,
        "teacher_factory_sha256": None,
        "teacher_factory_bundle_sha256": None,
        "software_stack_sha256": stack_sha,
        "dependency_lock_sha256": dep["dependency_lock_sha256"],
        "g1_seal_sha256": seal["g1_seal_sha256"],
        "g2_barrier_sha256": None,
        "allowed_surfaces": ["DS-V1-TRAIN", "DS-V1-VAL"],
    }
    root = Path(args.output_dir)

    def run_piece(name: str, steps: int, resume: bool):
        model = create_convnext_tiny_from_pretrained(args.pretrained, seed=SEED, num_classes=120)
        identity = dict(base_identity)
        identity["run_id"] = name
        budget = SessionBudget.from_environment(require_global_clock=True)
        return run_training(
            student=model, teacher=None, projection=None, train_dataset=train_ds, val_dataset=val_ds,
            ctc=ctc, objective={"ce": 1.0, "kd": 0.0, "feature": 0.0}, run_identity=identity,
            output_dir=root / name, num_workers=args.num_workers, resume=resume, max_optimizer_steps=steps,
            validation_enabled=False, checkpoint_every_steps=50, session_budget=budget,
            min_free_bytes=int(args.min_free_gb * 1024**3),
        )

    q1 = run_piece("CAL-CNXTT-RESUME-QUAL", QUAL_STEPS, False)
    q2 = run_piece("CAL-CNXTT-RESUME-QUAL", QUAL_RESUME_STEPS, True)
    measured = run_piece("CAL-CNXTT", CAL_STEPS, False)
    resume_success = q2.get("optimizer_steps_segment") == QUAL_RESUME_STEPS and q2.get("checkpoint_load_seconds", 0) > 0
    if not resume_success or measured.get("optimizer_steps_segment") != CAL_STEPS:
        raise SystemExit("ConvNeXt calibration qualification failed")
    measured["checkpoint_load_seconds"] = q2["checkpoint_load_seconds"]
    measured["durable_sync_seconds"] = 0.0
    measured["accelerator"] = (env.get("torch_cuda_devices") or [{}])[0].get("name")
    measured["cuda_driver_identity"] = env.get("nvidia_smi")
    summary = {
        "schema_version": "3.0",
        "status": "PASS",
        "calibration_id": "CAL-CNXTT",
        "condition": "context_direct",
        "locked_optimizer_steps": CAL_STEPS,
        "resume_success": True,
        "calibration_weights_scientific": False,
        "source_git_commit": args.source_git_commit,
        "software_stack_sha256": stack_sha,
        "g1_seal_sha256": seal["g1_seal_sha256"],
        "dependency_lock_sha256": dep["dependency_lock_sha256"],
        "mnv4_pretrained_sha256": seal["student"]["pretrained"]["sha256"],
        "teacher_checkpoint_sha256": None,
        "teacher_factory_bundle_sha256": None,
        "cnxtt_pretrained_sha256": cnxtt_pretrained_sha,
        "measured": measured,
        "observed_kaggle_constraints": {"quota_visibility": "not_exposed_programmatically_to_runner"},
    }
    atomic_write_json(root / "calibration_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
