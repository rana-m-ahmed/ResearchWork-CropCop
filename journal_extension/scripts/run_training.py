from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.data import CropCopManifestDataset, ManifestColumns, load_manifest_rows
from cropcop_je.environment import capture_environment, software_stack_identity, validate_locked_core
from cropcop_je.g1 import (
    validate_dependency_environment, validate_dependency_lock_object,
    validate_g1_seal_object, validate_teacher_class_order_evidence,
)
from cropcop_je.g2 import validate_g2_barrier_object
from cropcop_je.hashing import require_sha256, sha256_file, sha256_json
from cropcop_je.models import build_projection_without_state_drift, load_exact_teacher, load_pair_initialization
from cropcop_je.persistence import build_store
from cropcop_je.runlog import claim_run_directory, write_run_record
from cropcop_je.segments import append_segment_event, host_identity, new_segment_id, utc_now
from cropcop_je.session import SessionBudget
from cropcop_je.surfaces import validate_training_config
from cropcop_je.train import run_training

TRAIN_COUNT = 76376
VAL_COUNT = 16368


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_software_identity() -> tuple[dict, str]:
    env = capture_environment()
    drift = validate_locked_core(env)
    if drift:
        raise RuntimeError(f"locked software identity mismatch: {json.dumps(drift, sort_keys=True)}")
    if env.get("cuda_available") is not True:
        raise RuntimeError("CUDA is unavailable; this host cannot qualify G2 for the locked FP16 path")
    return env, sha256_json(software_stack_identity(env))


def prepare(args, *, mode: str):
    config = load_json(args.config)
    validate_training_config(config)
    if not args.run_id.strip():
        raise ValueError("run_id must be explicit and non-empty")
    if args.lane_id not in {"K1", "K2", "K3"}:
        raise ValueError(f"invalid lane_id: {args.lane_id}")

    env, stack_sha = require_software_identity()

    dependency_path = Path(args.dependency_lock)
    if not dependency_path.is_absolute():
        dependency_path = Path(args.repo_root) / dependency_path
    dependency_lock = load_json(dependency_path)
    dep_errors = validate_dependency_lock_object(dependency_lock) + validate_dependency_environment(dependency_lock)
    if dep_errors:
        raise RuntimeError("execution dependency lock mismatch: " + "; ".join(dep_errors))

    if not args.g1_seal:
        raise RuntimeError("G1 seal is mandatory for calibration and scientific execution")
    g1_seal = load_json(args.g1_seal)
    g1_errors = validate_g1_seal_object(g1_seal)
    if g1_errors:
        raise RuntimeError("G1 seal invalid: " + "; ".join(g1_errors))
    if g1_seal.get("source_git_sha") != args.source_git_commit:
        raise RuntimeError("G1 seal source SHA differs from requested execution source")
    if g1_seal.get("dependency_lock_sha256") != dependency_lock.get("dependency_lock_sha256"):
        raise RuntimeError("G1 seal dependency lock differs from mounted execution lock")

    g2_barrier = None
    g2_sha = None
    if mode == "scientific":
        if not args.g2_barrier:
            raise RuntimeError("principal scientific execution requires a terminal G2 barrier")
        g2_barrier = load_json(args.g2_barrier)
        g2_errors = validate_g2_barrier_object(
            g2_barrier,
            expected_source_sha=args.source_git_commit,
            expected_g1_seal_sha256=g1_seal["g1_seal_sha256"],
        )
        if g2_errors:
            raise RuntimeError("G2 barrier invalid: " + "; ".join(g2_errors))
        if g2_barrier.get("dependency_lock_sha256") != dependency_lock.get("dependency_lock_sha256"):
            raise RuntimeError("G2 dependency lock differs from mounted execution lock")
        g2_sha = g2_barrier["barrier_sha256"]

    ctc_path = Path(args.repo_root) / config["ctc_config"]
    ctc = load_json(ctc_path)
    config_sha = sha256_json(config)
    ctc_sha = sha256_json(ctc)

    manifest_sha = require_sha256(args.manifest, config["manifest_sha256"], "V1 manifest")
    class_map_sha = require_sha256(args.class_map, config["class_map_sha256"], "120-way class map")
    cols = ManifestColumns(args.row_id_column, args.path_column, args.split_column, args.class_index_column)
    train_rows = load_manifest_rows(
        args.manifest,
        expected_sha256=config["manifest_sha256"],
        surface="DS-V1-TRAIN",
        columns=cols,
        train_split_value=args.train_split_value,
        val_split_value=args.val_split_value,
        expected_count=TRAIN_COUNT,
    )
    val_rows = load_manifest_rows(
        args.manifest,
        expected_sha256=config["manifest_sha256"],
        surface="DS-V1-VAL",
        columns=cols,
        train_split_value=args.train_split_value,
        val_split_value=args.val_split_value,
        expected_count=VAL_COUNT,
    )

    pair_evidence = load_json(args.pair_init_evidence)
    if pair_evidence.get("pair_id") != config["pair_id"] or int(pair_evidence.get("seed", -1)) != int(config["seed"]):
        raise ValueError("pair-init evidence does not match config pair/seed")
    seal_pair = next(
        (row for row in g1_seal.get("pair_initializations", {}).values() if row.get("pair_id") == config["pair_id"]),
        None,
    )
    if not seal_pair:
        raise ValueError("config pair is not authorized by the global G1 seal")
    if seal_pair.get("sha256") != pair_evidence.get("student_init_sha256"):
        raise ValueError("pair-init evidence SHA differs from the global G1 seal")
    if seal_pair.get("pretrained_sha256") != pair_evidence.get("pretrained_sha256"):
        raise ValueError("pair-init pretrained identity differs from the global G1 seal")
    if config["experiment_id"] not in set(seal_pair.get("authorized_consumers", [])):
        raise ValueError("experiment is not an authorized consumer of the sealed pair initialization")
    init_sha = pair_evidence["student_init_sha256"]
    student, init_payload = load_pair_initialization(
        args.pair_init,
        expected_sha256=init_sha,
        pair_id=config["pair_id"],
        seed=config["seed"],
    )
    pretrained_sha = pair_evidence["pretrained_sha256"]
    if init_payload.get("pretrained_sha256") != pretrained_sha:
        raise ValueError("pair-init payload/evidence disagree on pretrained SHA")

    teacher = None
    projection = None
    teacher_sha = None
    teacher_factory_sha = None
    teacher_factory_bundle_sha = None
    teacher_factory_identity = None
    if config["condition"] == "teacher":
        if not args.teacher_checkpoint or not args.teacher_factory or not args.teacher_evidence:
            raise ValueError("teacher run requires --teacher-checkpoint, --teacher-factory and --teacher-evidence")
        if not args.teacher_factory_manifest or not args.teacher_class_order_evidence:
            raise ValueError("teacher run requires sealed factory-bundle and class-order evidence")
        teacher_evidence = load_json(args.teacher_evidence)
        teacher_sha = require_sha256(
            args.teacher_checkpoint,
            config["teacher_checkpoint_sha256"],
            "historical DINO teacher",
        )
        if teacher_evidence.get("sha256") != teacher_sha:
            raise ValueError("teacher evidence SHA does not match actual teacher bytes")
        if teacher_evidence.get("class_map_sha256") != class_map_sha:
            raise ValueError("teacher evidence does not bind the frozen 120-way class map")
        teacher_order = load_json(args.teacher_class_order_evidence)
        order_errors = validate_teacher_class_order_evidence(
            teacher_order,
            factory_bundle_sha256=g1_seal.get("teacher", {}).get("factory_bundle_sha256"),
        )
        if order_errors:
            raise ValueError("teacher class-order evidence invalid: " + "; ".join(order_errors))
        if sha256_json(teacher_order) != g1_seal.get("teacher", {}).get("class_order_evidence_sha256"):
            raise ValueError("teacher class-order evidence differs from G1 seal")
        teacher, teacher_factory_identity = load_exact_teacher(
            args.teacher_checkpoint,
            factory_spec=args.teacher_factory,
            factory_bundle_manifest=args.teacher_factory_manifest,
            repo_root=args.repo_root,
            factory_source_root=(args.teacher_factory_root or args.repo_root),
        )
        teacher_factory_bundle_sha = teacher_factory_identity["bundle_sha256"]
        if teacher_factory_bundle_sha != g1_seal.get("teacher", {}).get("factory_bundle_sha256"):
            raise ValueError("teacher factory bundle differs from G1 seal")
        teacher_factory_sha = sha256_json(teacher_factory_identity)
        projection = build_projection_without_state_drift(student, teacher, seed=config["seed"])

    train_ds = CropCopManifestDataset(train_rows, args.image_root, training_seed=config["seed"], train=True)
    val_ds = CropCopManifestDataset(val_rows, args.image_root, training_seed=config["seed"], train=False)

    run_identity = {
        "run_id": args.run_id,
        "experiment_id": config["experiment_id"],
        "authority_id": config["authority_id"],
        "source_git_commit": args.source_git_commit,
        "lane_id": args.lane_id,
        "config_sha256": config_sha,
        "ctc_v2_sha256": ctc_sha,
        "manifest_sha256": manifest_sha,
        "class_map_sha256": class_map_sha,
        "seed": int(config["seed"]),
        "student_init_sha256": init_sha,
        "pretrained_sha256": pretrained_sha,
        "teacher_sha256": teacher_sha,
        "teacher_factory_sha256": teacher_factory_sha,
        "teacher_factory_bundle_sha256": teacher_factory_bundle_sha,
        "software_stack_sha256": stack_sha,
        "dependency_lock_sha256": dependency_lock["dependency_lock_sha256"],
        "g1_seal_sha256": g1_seal["g1_seal_sha256"],
        "g2_barrier_sha256": g2_sha,
        "allowed_surfaces": ["DS-V1-TRAIN", "DS-V1-VAL"],
    }
    return (
        config, ctc, student, teacher, projection, train_ds, val_ds,
        run_identity, env, teacher_factory_identity,
    )


def _durable_store(args):
    if not args.durable_store_kind:
        return None
    if not args.durable_store_locator:
        raise ValueError("durable store kind requires --durable-store-locator")
    return build_store(args.durable_store_kind, args.durable_store_locator)


def _public_locator(args) -> str | None:
    if not args.durable_store_locator:
        return None
    if args.durable_store_kind == "kaggle-dataset":
        return args.durable_store_locator
    return "restricted-filesystem-locator"


def execute(args, *, max_optimizer_steps=None, resume_mode="auto", mode="scientific") -> dict:
    (
        config, ctc, student, teacher, projection, train_ds, val_ds,
        run_identity, env, teacher_factory_identity,
    ) = prepare(args, mode=mode)

    output = Path(args.output_dir)
    claim_run_directory(
        output,
        run_id=args.run_id,
        experiment_id=config["experiment_id"],
        lane_id=args.lane_id,
    )
    checkpoint_root = output / "private_checkpoints"
    segment_ledger = output / "segments.jsonl"
    segment_id = new_segment_id(args.run_id)
    store = _durable_store(args)

    resume = False
    if resume_mode == "auto":
        if not (checkpoint_root / "checkpoint_index.json").exists() and store is not None:
            try:
                store.restore(checkpoint_root, run_id=args.run_id)
            except Exception as exc:
                append_segment_event(
                    segment_ledger,
                    {
                        "segment_id": segment_id,
                        "parent_run_id": args.run_id,
                        "state": "RESTORE_FAILED",
                        "timestamp_utc": utc_now(),
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                raise
        resume = (checkpoint_root / "checkpoint_index.json").exists()
    elif resume_mode == "required":
        if not (checkpoint_root / "checkpoint_index.json").exists() and store is not None:
            store.restore(checkpoint_root, run_id=args.run_id)
        if not (checkpoint_root / "checkpoint_index.json").exists():
            raise RuntimeError("resume required but no verified checkpoint index is available")
        resume = True
    elif resume_mode == "never":
        if (checkpoint_root / "checkpoint_index.json").exists():
            raise RuntimeError("fresh execution refused because checkpoint state already exists")
    else:
        raise ValueError(f"unsupported resume_mode: {resume_mode}")

    record_path = output / "run_record.json"
    record = {
        **run_identity,
        "status": "LAUNCHED",
        "mode": mode,
        "segment_id": segment_id,
        "environment": env,
        "teacher_factory_identity": teacher_factory_identity,
        "hardware_identity": {
            "accelerator": (env.get("torch_cuda_devices") or [{}])[0].get("name")
            if env.get("torch_cuda_devices")
            else None,
            "nvidia_smi": env.get("nvidia_smi"),
        },
        "artifact_locators": {},
        "durable_store": {
            "kind": args.durable_store_kind or None,
            "locator": _public_locator(args),
            "required": bool(args.durable_required),
        },
        "continuation_required": bool(resume),
        "entrypoint": Path(sys.argv[0]).name,
        "started_at_utc": utc_now(),
        "notebook_session_clock": SessionBudget.from_environment(require_global_clock=True).snapshot(),
    }
    write_run_record(record_path, record)
    append_segment_event(
        segment_ledger,
        {
            "segment_id": segment_id,
            "parent_run_id": args.run_id,
            "state": "START",
            "timestamp_utc": utc_now(),
            "source_git_sha": args.source_git_commit,
            "host_identity": host_identity(),
            "gpu_identity": record["hardware_identity"],
            "input_artifact_hashes": {
                "manifest": run_identity["manifest_sha256"],
                "class_map": run_identity["class_map_sha256"],
                "pretrained": run_identity["pretrained_sha256"],
                "teacher": run_identity["teacher_sha256"],
                "student_init": run_identity["student_init_sha256"],
            },
            "starting_checkpoint_present": bool(resume),
        },
    )

    budget = SessionBudget.from_environment(
        require_global_clock=True,
        hard_limit_seconds=float(args.session_hard_limit_seconds),
        finalization_margin_seconds=float(args.finalization_margin_seconds),
    )
    try:
        result = run_training(
            student=student,
            teacher=teacher,
            projection=projection,
            train_dataset=train_ds,
            val_dataset=val_ds,
            ctc=ctc,
            objective=config["objective"],
            run_identity=run_identity,
            output_dir=checkpoint_root,
            num_workers=args.num_workers,
            resume=resume,
            max_optimizer_steps=max_optimizer_steps,
            validation_enabled=(mode == "scientific"),
            checkpoint_every_steps=args.checkpoint_every_steps,
            session_budget=budget,
            min_free_bytes=int(float(args.min_free_gb) * 1024**3),
        )

        segment_result_path = output / f"segment_{segment_id}.json"
        atomic_write_json(segment_result_path, result)

        persistence = None
        durable_sync_seconds = 0.0
        if store is not None:
            sync_started = time.perf_counter()
            persistence = store.sync(
                checkpoint_root,
                run_id=args.run_id,
                segment_id=segment_id,
            ).to_dict()
            durable_sync_seconds = time.perf_counter() - sync_started
        result["durable_sync_seconds"] = durable_sync_seconds
        elif args.durable_required:
            raise RuntimeError("durable persistence is required but no durable store is configured")

        if result.get("planned_rollover"):
            record.update(
                {
                    "status": "LAUNCHED",
                    "continuation_required": True,
                    "last_segment_result_sha256": sha256_file(segment_result_path),
                    "persistence_status": persistence or {"status": "NOT_CONFIGURED"},
                    "last_optimizer_step": result.get("optimizer_step_total"),
                    "last_checkpoint_sha256": result.get("latest_checkpoint_sha256"),
                }
            )
            write_run_record(record_path, record)
            append_segment_event(
                segment_ledger,
                {
                    "segment_id": segment_id,
                    "parent_run_id": args.run_id,
                    "state": "END",
                    "timestamp_utc": utc_now(),
                    "termination_reason": result.get("rollover_reason") or "PLANNED_SESSION_ROLLOVER",
                    "ending_checkpoint_sha": result.get("latest_checkpoint_sha256"),
                    "ending_optimizer_step": result.get("optimizer_step_total"),
                    "persistence_status": persistence,
                },
            )
            return record

        if mode == "calibration":
            record.update(
                {
                    "status": "PASS",
                    "continuation_required": False,
                    "result_summary": result,
                    "last_segment_result_sha256": sha256_file(segment_result_path),
                    "persistence_status": persistence or {"status": "NOT_CONFIGURED"},
                    "finished_at_utc": utc_now(),
                }
            )
        else:
            metrics_path = output / "metrics.json"
            atomic_write_json(metrics_path, result)
            locator = _public_locator(args)
            record.update(
                {
                    "status": "PASS",
                    "continuation_required": False,
                    "result_summary": {
                        k: result.get(k)
                        for k in (
                            "mode",
                            "optimizer_steps_segment",
                            "optimizer_step_total",
                            "examples_segment",
                            "examples_total",
                            "wall_seconds_segment",
                            "sec_per_optimizer_step",
                            "examples_per_second",
                            "dataloader_wait_seconds",
                            "dataloader_examples_per_wait_second",
                            "peak_gpu_memory_bytes",
                            "checkpoint_save_seconds",
                            "checkpoint_load_seconds",
                            "durable_sync_seconds",
                            "selected_epoch",
                            "selected_metrics",
                            "selected_checkpoint_sha256",
                            "latest_checkpoint_sha256",
                            "validation_forward_benchmark",
                        )
                    },
                    "artifact_locators": {
                        "metrics": {"basename": metrics_path.name, "sha256": sha256_file(metrics_path)},
                        "selected_checkpoint": {
                            "sha256": result["selected_checkpoint_sha256"],
                            "public_git": False,
                            "durable_locator": locator,
                        },
                        "latest_checkpoint": {
                            "sha256": result["latest_checkpoint_sha256"],
                            "public_git": False,
                            "durable_locator": locator,
                        },
                    },
                    "persistence_status": persistence or {"status": "NOT_CONFIGURED"},
                    "finished_at_utc": utc_now(),
                }
            )

        write_run_record(record_path, record)
        append_segment_event(
            segment_ledger,
            {
                "segment_id": segment_id,
                "parent_run_id": args.run_id,
                "state": "END",
                "timestamp_utc": utc_now(),
                "termination_reason": "COMPLETE",
                "ending_checkpoint_sha": result.get("latest_checkpoint_sha256"),
                "ending_optimizer_step": result.get("optimizer_step_total"),
                "persistence_status": persistence,
            },
        )
        return record

    except BaseException as exc:
        is_interrupt = isinstance(exc, (KeyboardInterrupt, SystemExit))
        status = "INTERRUPTED" if is_interrupt else "FAIL"
        failure = {
            "type": type(exc).__name__,
            "reason": str(exc),
            "technical_retry_allowed_only_if_science_unchanged": True,
            "traceback_tail": traceback.format_exc()[-4000:],
        }
        record.update(
            {
                "status": status,
                "continuation_required": False,
                "failure": failure,
                "finished_at_utc": utc_now(),
            }
        )
        try:
            write_run_record(record_path, record)
            append_segment_event(
                segment_ledger,
                {
                    "segment_id": segment_id,
                    "parent_run_id": args.run_id,
                    "state": "END",
                    "timestamp_utc": utc_now(),
                    "termination_reason": status,
                    "error": failure,
                    "persistence_status": "NOT_CLAIMED_SAFE",
                },
            )
        finally:
            raise


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--class-map", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--pair-init", required=True)
    ap.add_argument("--pair-init-evidence", required=True)
    ap.add_argument("--teacher-checkpoint", default="")
    ap.add_argument("--teacher-evidence", default="")
    ap.add_argument("--teacher-factory", default="")
    ap.add_argument("--teacher-factory-manifest", default="")
    ap.add_argument("--teacher-factory-root", default="")
    ap.add_argument("--teacher-class-order-evidence", default="")
    ap.add_argument("--g1-seal", required=True)
    ap.add_argument("--g2-barrier", default="")
    ap.add_argument("--dependency-lock", default="journal_extension/locks/execution_dependency_lock.json")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--lane-id", choices=["K1", "K2", "K3"], required=True)
    ap.add_argument("--source-git-commit", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--row-id-column", required=True)
    ap.add_argument("--path-column", required=True)
    ap.add_argument("--split-column", required=True)
    ap.add_argument("--class-index-column", required=True)
    ap.add_argument("--train-split-value", default="train")
    ap.add_argument("--val-split-value", default="val")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--checkpoint-every-steps", type=int, default=250)
    ap.add_argument("--resume-mode", choices=["auto", "required", "never"], default="auto")
    ap.add_argument("--max-optimizer-steps", type=int, default=0)
    ap.add_argument("--mode", choices=["scientific", "calibration"], default="scientific")
    ap.add_argument("--session-hard-limit-seconds", type=float, default=12 * 3600)
    ap.add_argument("--finalization-margin-seconds", type=float, default=3600)
    ap.add_argument("--min-free-gb", type=float, default=5.0)
    ap.add_argument("--durable-store-kind", choices=["filesystem", "kaggle-dataset"], default="")
    ap.add_argument("--durable-store-locator", default="")
    ap.add_argument("--durable-required", action="store_true")
    return ap


def main() -> int:
    args = parser().parse_args()
    if args.mode == "scientific" and args.max_optimizer_steps:
        raise SystemExit("scientific mode must run the full locked 30 epochs")
    if args.mode == "calibration" and args.max_optimizer_steps <= 0:
        raise SystemExit("calibration mode requires --max-optimizer-steps")
    record = execute(
        args,
        max_optimizer_steps=(args.max_optimizer_steps or None),
        resume_mode=args.resume_mode,
        mode=args.mode,
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
