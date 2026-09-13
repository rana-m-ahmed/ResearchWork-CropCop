from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.checkpointing import verify_selected
from cropcop_je.hashing import sha256_file, sha256_json
from cropcop_je.models import load_pair_initialization
from cropcop_je.source_state import verify_clean_source
from cropcop_je.tracka_v12 import PAIR_IDS, SEEDS
from cropcop_je.tracka_v12_evidence import validate_selected_checkpoint_replay
from cropcop_je.tracka_v12_historical import HISTORICAL_TRACKA_CLOSURE_SPECS, validate_historical_closure_identity
from cropcop_je.tracka_v12_posttraining import clean_replay, write_jsonl
from cropcop_je.tracka_v12_runtime import load_and_validate_g1a_bundle
from cropcop_je.train import _identity as checkpoint_identity
from run_tracka_v12_direct_evidence import load_json, restore_checkpoint_root, validation_rows

ALLOWED_PREFIXES = ("R05-MNV4-TEACHER-", "R12-MNV4-LOGITS-", "R12-MNV4-FEATURE-")


def seed_label(experiment_id: str) -> str:
    label = experiment_id.rsplit("-", 1)[-1]
    if label not in SEEDS:
        raise ValueError(f"cannot resolve seed label from {experiment_id}")
    return label


def load_auxiliary_student(args, run_record: dict):
    experiment_id = str(run_record.get("experiment_id", ""))
    if not experiment_id.startswith(ALLOWED_PREFIXES):
        raise RuntimeError(f"not an R05/R12 auxiliary Track-A state: {experiment_id}")
    config = load_json(args.config)
    if config.get("experiment_id") != experiment_id:
        raise RuntimeError("auxiliary config/run-record experiment mismatch")
    if sha256_json(config) != run_record.get("config_sha256"):
        raise RuntimeError("auxiliary config SHA differs from scientific run record")
    label = seed_label(experiment_id)
    seed = int(SEEDS[label])
    pair_id = str(config.get("pair_id", ""))
    if pair_id != PAIR_IDS[label] or int(config.get("seed", -1)) != seed:
        raise RuntimeError("auxiliary config pair/seed drift")

    if experiment_id.startswith("R12-") and label in {"S2", "S3"}:
        if not args.g1a_bundle:
            raise RuntimeError("R12 S2/S3 auxiliary replay requires the exact G1A bundle")
        seal, errors = load_and_validate_g1a_bundle(args.g1a_bundle, expected_source_sha=str(run_record["source_git_commit"]))
        if errors:
            raise RuntimeError("G1A bundle invalid for R12 auxiliary replay: " + "; ".join(errors))
        row = seal["r12_reused_pairs"][label]
        if experiment_id not in set(row.get("authorized_consumers", [])):
            raise RuntimeError("R12 auxiliary state is not an authorized reused-pair consumer")
        init_path = Path(args.g1a_bundle) / "private" / row["student_init_basename"]
        student, payload = load_pair_initialization(init_path, expected_sha256=row["student_init_sha256"], pair_id=pair_id, seed=seed)
        init_sha = row["student_init_sha256"]
    else:
        if not args.principal_pair_init or not args.principal_pair_evidence:
            raise RuntimeError("historical R05/R12-S1 replay requires principal pair init/evidence")
        evidence = load_json(args.principal_pair_evidence)
        if evidence.get("pair_id") != pair_id or int(evidence.get("seed", -1)) != seed:
            raise RuntimeError("principal pair evidence pair/seed mismatch")
        init_sha = str(evidence.get("student_init_sha256", ""))
        student, payload = load_pair_initialization(args.principal_pair_init, expected_sha256=init_sha, pair_id=pair_id, seed=seed)

    if init_sha != run_record.get("student_init_sha256"):
        raise RuntimeError("auxiliary initialization SHA differs from run record")
    if payload.get("pretrained_sha256") != run_record.get("pretrained_sha256"):
        raise RuntimeError("auxiliary pretrained identity differs from run record")

    checkpoint_root = restore_checkpoint_root(args)
    selected_sha = run_record.get("artifact_locators", {}).get("selected_checkpoint", {}).get("sha256")
    if not selected_sha:
        selected_sha = run_record.get("result_summary", {}).get("selected_checkpoint_sha256")
    if not selected_sha:
        raise RuntimeError("auxiliary run record lacks selected checkpoint SHA")
    checkpoint_path, checkpoint_payload = verify_selected(checkpoint_root, expected_identity=checkpoint_identity(run_record), expected_sha256=str(selected_sha))
    student.load_state_dict(checkpoint_payload["student"], strict=True)
    return student, config, checkpoint_path, checkpoint_payload, str(selected_sha)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--analysis-source-git-commit", required=True)
    ap.add_argument("--run-record", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--class-map", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--g1a-bundle", default="")
    ap.add_argument("--principal-pair-init", default="")
    ap.add_argument("--principal-pair-evidence", default="")
    ap.add_argument("--durable-store-kind", choices=["", "filesystem", "kaggle-dataset"], default="")
    ap.add_argument("--durable-store-locator", default="")
    ap.add_argument("--row-id-column", default="record_key")
    ap.add_argument("--path-column", default="portable_relpath")
    ap.add_argument("--split-column", default="split")
    ap.add_argument("--label-column", default="label")
    ap.add_argument("--class-index-column", default="")
    ap.add_argument("--train-split-value", default="train")
    ap.add_argument("--val-split-value", default="val")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = ap.parse_args()

    import torch

    repo = Path(args.repo_root).resolve()
    output = Path(args.output_dir).resolve()
    analysis_source = args.analysis_source_git_commit.strip()
    if len(analysis_source) != 40:
        raise SystemExit("analysis source Git commit must be a full 40-character SHA")
    verify_clean_source(repo, authorized_source_sha=analysis_source, output_roots=[output])

    run_record = load_json(args.run_record)
    if run_record.get("run_id") != args.run_id or run_record.get("status") != "PASS":
        raise SystemExit("auxiliary evidence requires the requested terminal PASS scientific run")
    experiment_id = str(run_record.get("experiment_id", ""))
    if experiment_id in HISTORICAL_TRACKA_CLOSURE_SPECS:
        historical_errors = validate_historical_closure_identity(run_record)
        if historical_errors:
            raise SystemExit("historical auxiliary run is not the canonical sealed Track-A state: " + "; ".join(historical_errors))
    if run_record.get("v1_test_accessed") not in {None, False}:
        raise SystemExit("auxiliary run record indicates V1-test access")
    if run_record.get("protected_external_surface_accessed") not in {None, False}:
        raise SystemExit("auxiliary run record indicates protected external-surface access")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA auxiliary replay requested but CUDA is unavailable")

    student, config, checkpoint_path, checkpoint_payload, selected_sha = load_auxiliary_student(args, run_record)
    rows = validation_rows(args)
    predictions, summary = clean_replay(student, rows, args.image_root, torch.device(args.device), batch_size=args.batch_size, num_workers=args.num_workers)
    expected_metrics = run_record.get("result_summary", {}).get("selected_metrics") or {}
    replay = validate_selected_checkpoint_replay(summary, expected_metrics)
    if replay.get("status") != "PASS":
        raise SystemExit("auxiliary selected-checkpoint replay failed frozen tolerance")

    private = output / "private_evidence"
    public = output / "public_evidence"
    private.mkdir(parents=True, exist_ok=False)
    public.mkdir(parents=True, exist_ok=False)
    predictions_path = private / "selected_checkpoint_validation_predictions.jsonl"
    write_jsonl(predictions_path, predictions)
    scientific_source = str(run_record["source_git_commit"])
    gate = {
        "schema_version": "1.1",
        "status": "PASS",
        "experiment_id": run_record["experiment_id"],
        "run_id": run_record["run_id"],
        "condition": config.get("condition"),
        "objective": config.get("objective"),
        "source_git_commit": scientific_source,
        "scientific_source_git_commit": scientific_source,
        "analysis_source_git_commit": analysis_source,
        "selected_checkpoint_sha256": selected_sha,
        "selected_checkpoint_file_sha256": sha256_file(checkpoint_path),
        "selected_checkpoint_epoch": int(checkpoint_payload["epoch"]),
        "replay_gate": replay,
        "clean_summary": summary,
        "classwise_pass": len(summary.get("classwise", [])) == 120,
        "training_performed": False,
        "optimizer_state_advanced": False,
        "v1_test_accessed": False,
        "external_surface_accessed": False,
        "private_predictions_sha256": sha256_file(predictions_path),
    }
    atomic_write_json(public / "AUXILIARY_STATE_EVIDENCE_GATE.json", gate)
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
