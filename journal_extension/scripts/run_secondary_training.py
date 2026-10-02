from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.data import CropCopManifestDataset
from cropcop_je.environment import capture_environment, software_stack_identity, validate_locked_core
from cropcop_je.frozen_v1_manifest import load_frozen_v1_rows
from cropcop_je.g1 import (
    validate_dependency_environment,
    validate_dependency_lock_object,
    validate_teacher_class_order_evidence,
)
from cropcop_je.secondary_g2 import validate_secondary_g2_barrier_object
from cropcop_je.hashing import sha256_file, sha256_json
from cropcop_je.models import (
    build_projection_without_state_drift,
    load_exact_teacher,
    load_pair_initialization,
)
from cropcop_je.persistence import build_store
from cropcop_je.runlog import claim_run_directory, write_run_record
from cropcop_je.secondary import (
    CLASS_MAP_SHA256,
    MANIFEST_SHA256,
    S1_PAIR_ID,
    S1_SEED,
    experiment_config_path,
    load_baseline_initialization,
    load_json,
    validate_secondary_config,
    validate_secondary_g1_bundle,
)
from cropcop_je.segments import append_segment_event, host_identity, new_segment_id, utc_now
from cropcop_je.session import SessionBudget
from cropcop_je.terminal_recovery import completed_scientific_checkpoint_result
from cropcop_je.train import _identity as checkpoint_identity
from cropcop_je.train import run_training

TRAIN_COUNT = 76376
VAL_COUNT = 16368


def software_identity() -> tuple[dict, str]:
    env = capture_environment()
    drift = validate_locked_core(env)
    if drift:
        raise RuntimeError(f"locked software identity mismatch: {json.dumps(drift, sort_keys=True)}")
    if env.get("cuda_available") is not True:
        raise RuntimeError("secondary scientific execution requires CUDA")
    return env, sha256_json(software_stack_identity(env))


def durable_store(args):
    if not args.durable_store_kind:
        return None
    if not args.durable_store_locator:
        raise ValueError("durable store kind requires locator")
    return build_store(args.durable_store_kind, args.durable_store_locator)


def public_locator(args) -> str | None:
    if not args.durable_store_locator:
        return None
    return args.durable_store_locator if args.durable_store_kind == "kaggle-dataset" else "restricted-filesystem-locator"


def prepare(args, *, mode: str):
    repo = Path(args.repo_root).resolve()
    config_path = Path(args.config) if args.config else repo / experiment_config_path(args.experiment_id)
    if not config_path.is_absolute():
        config_path = repo / config_path
    config = load_json(config_path)
    errors = validate_secondary_config(config)
    if errors:
        raise RuntimeError("secondary config invalid: " + "; ".join(errors))
    if config["experiment_id"] != args.experiment_id:
        raise RuntimeError("requested experiment/config mismatch")

    env, stack_sha = software_identity()
    dep_path = Path(args.dependency_lock)
    if not dep_path.is_absolute():
        dep_path = repo / dep_path
    dep = load_json(dep_path)
    dep_errors = validate_dependency_lock_object(dep) + validate_dependency_environment(dep)
    if dep_errors:
        raise RuntimeError("dependency lock mismatch: " + "; ".join(dep_errors))

    seal, seal_errors = validate_secondary_g1_bundle(args.secondary_g1_bundle)
    if seal_errors:
        raise RuntimeError("secondary G1 invalid: " + "; ".join(seal_errors))
    if seal.get("source_git_sha") != args.source_git_commit:
        raise RuntimeError("secondary G1 source differs from requested execution source")
    if seal.get("dependency_lock_sha256") != dep.get("dependency_lock_sha256"):
        raise RuntimeError("secondary G1 dependency lock differs from execution lock")

    g2_sha = None
    if mode == "scientific":
        if not args.g2_barrier:
            raise RuntimeError("secondary scientific execution requires terminal G2 barrier")
        g2 = load_json(args.g2_barrier)
        g2_errors = validate_secondary_g2_barrier_object(
            g2,
            expected_source_sha=args.source_git_commit,
            expected_g1_seal_sha256=seal["secondary_g1_seal_sha256"],
        )
        if g2_errors:
            raise RuntimeError("secondary G2 invalid: " + "; ".join(g2_errors))
        if g2.get("dependency_lock_sha256") != dep.get("dependency_lock_sha256"):
            raise RuntimeError("secondary G2 dependency lock mismatch")
        g2_sha = g2["barrier_sha256"]

    manifest_sha = sha256_file(args.manifest)
    class_map_sha = sha256_file(args.class_map)
    if manifest_sha != MANIFEST_SHA256 or class_map_sha != CLASS_MAP_SHA256:
        raise RuntimeError("frozen dataset identity mismatch")

    train_rows = load_frozen_v1_rows(
        args.manifest,
        args.class_map,
        expected_manifest_sha256=MANIFEST_SHA256,
        expected_class_map_sha256=CLASS_MAP_SHA256,
        surface="DS-V1-TRAIN",
        row_id_column=args.row_id_column,
        path_column=args.path_column,
        split_column=args.split_column,
        label_column=args.label_column,
        class_index_column=args.class_index_column,
        train_split_value=args.train_split_value,
        val_split_value=args.val_split_value,
        expected_count=TRAIN_COUNT,
    )
    val_rows = load_frozen_v1_rows(
        args.manifest,
        args.class_map,
        expected_manifest_sha256=MANIFEST_SHA256,
        expected_class_map_sha256=CLASS_MAP_SHA256,
        surface="DS-V1-VAL",
        row_id_column=args.row_id_column,
        path_column=args.path_column,
        split_column=args.split_column,
        label_column=args.label_column,
        class_index_column=args.class_index_column,
        train_split_value=args.train_split_value,
        val_split_value=args.val_split_value,
        expected_count=VAL_COUNT,
    )

    bundle = Path(args.secondary_g1_bundle)
    private = bundle / "private"
    evidence = bundle / "evidence"
    family = config["model_family"]
    teacher = None
    projection = None
    teacher_identity = None
    teacher_sha = None
    teacher_factory_bundle_sha = None

    if family == "mnv4":
        pair = seal["mnv4_s1"]
        student, payload = load_pair_initialization(
            private / pair["student_init_basename"],
            expected_sha256=pair["student_init_sha256"],
            pair_id=S1_PAIR_ID,
            seed=S1_SEED,
        )
        init_sha = pair["student_init_sha256"]
        pretrained_sha = pair["pretrained_sha256"]
        if payload.get("pretrained_sha256") != pretrained_sha:
            raise RuntimeError("secondary MNV4 initialization/pretrained identity mismatch")
        if config["condition"] == "teacher":
            teacher_row = seal["teacher"]
            class_order = load_json(evidence / "TEACHER_CLASS_ORDER_EVIDENCE.json")
            order_errors = validate_teacher_class_order_evidence(
                class_order,
                factory_bundle_sha256=teacher_row["factory_bundle_sha256"],
            )
            if order_errors:
                raise RuntimeError("teacher class-order evidence invalid: " + "; ".join(order_errors))
            teacher, teacher_identity = load_exact_teacher(
                private / teacher_row["artifact_basename"],
                factory_spec=teacher_row["factory_entrypoint"],
                factory_bundle_manifest=evidence / "TEACHER_FACTORY_BUNDLE.json",
                repo_root=repo,
                factory_source_root=repo / "journal_extension/teacher_factory",
            )
            teacher_sha = teacher_row["checkpoint_sha256"]
            teacher_factory_bundle_sha = teacher_row["factory_bundle_sha256"]
            if float(config["objective"].get("feature", 0.0)) > 0:
                projection = build_projection_without_state_drift(student, teacher, seed=S1_SEED)
    else:
        row = seal["baselines"][family]
        student, payload = load_baseline_initialization(
            private / row["init_basename"],
            expected_sha256=row["init_sha256"],
            model_key=family,
            seed=S1_SEED,
            experiment_id=config["experiment_id"],
        )
        init_sha = row["init_sha256"]
        pretrained_sha = row["pretrained_sha256"]
        if payload.get("pretrained_sha256") != pretrained_sha:
            raise RuntimeError("baseline initialization/pretrained identity mismatch")

    ctc = load_json(repo / config["ctc_config"])
    train_ds = CropCopManifestDataset(train_rows, args.image_root, training_seed=S1_SEED, train=True)
    val_ds = CropCopManifestDataset(val_rows, args.image_root, training_seed=S1_SEED, train=False)
    run_identity = {
        "run_id": args.run_id,
        "experiment_id": config["experiment_id"],
        "authority_id": config["authority_id"],
        "source_git_commit": args.source_git_commit,
        "lane_id": args.lane_id,
        "config_sha256": sha256_json(config),
        "ctc_v2_sha256": sha256_json(ctc),
        "manifest_sha256": manifest_sha,
        "class_map_sha256": class_map_sha,
        "seed": S1_SEED,
        "student_init_sha256": init_sha,
        "pretrained_sha256": pretrained_sha,
        "teacher_sha256": teacher_sha,
        "teacher_factory_sha256": sha256_json(teacher_identity) if teacher_identity else None,
        "teacher_factory_bundle_sha256": teacher_factory_bundle_sha,
        "software_stack_sha256": stack_sha,
        "dependency_lock_sha256": dep["dependency_lock_sha256"],
        "g1_seal_sha256": seal["secondary_g1_seal_sha256"],
        "g2_barrier_sha256": g2_sha,
        "lane_id": args.lane_id,
        "allowed_surfaces": ["DS-V1-TRAIN", "DS-V1-VAL"],
    }
    return config, ctc, student, teacher, projection, train_ds, val_ds, run_identity, env


def execute(args, *, mode: str, max_optimizer_steps: int | None = None, resume_mode: str = "auto") -> dict:
    config, ctc, student, teacher, projection, train_ds, val_ds, identity, env = prepare(args, mode=mode)
    output = Path(args.output_dir)
    claim_run_directory(output, run_id=args.run_id, experiment_id=config["experiment_id"], lane_id=args.lane_id)
    checkpoints = output / "private_checkpoints"
    segments = output / "segments.jsonl"
    segment_id = new_segment_id(args.run_id)
    store = durable_store(args)

    resume = False
    if resume_mode in {"auto", "required"}:
        if not (checkpoints / "checkpoint_index.json").exists() and store is not None:
            try:
                store.restore(checkpoints, run_id=args.run_id)
            except Exception as exc:
                append_segment_event(
                    output / "segments.jsonl",
                    {
                        "segment_id": segment_id,
                        "parent_run_id": args.run_id,
                        "state": "RESTORE_FAILED",
                        "timestamp_utc": utc_now(),
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                raise RuntimeError(
                    "durable checkpoint restore failed; fresh scientific restart is forbidden"
                ) from exc
        resume = (checkpoints / "checkpoint_index.json").exists()
        if resume_mode == "required" and not resume:
            raise RuntimeError("resume required but no verified durable checkpoint is available")
    elif resume_mode == "never":
        if (checkpoints / "checkpoint_index.json").exists():
            raise RuntimeError("fresh execution refused because checkpoint state exists")
    else:
        raise ValueError(f"unsupported resume mode: {resume_mode}")

    record_path = output / "run_record.json"
    record = {
        **identity,
        "status": "LAUNCHED",
        "mode": mode,
        "segment_id": segment_id,
        "environment": env,
        "hardware_identity": {
            "accelerator": (env.get("torch_cuda_devices") or [{}])[0].get("name"),
            "nvidia_smi": env.get("nvidia_smi"),
        },
        "artifact_locators": {},
        "durable_store": {
            "kind": args.durable_store_kind or None,
            "locator": public_locator(args),
            "required": bool(args.durable_required),
        },
        "continuation_required": bool(resume),
        "entrypoint": Path(sys.argv[0]).name,
        "started_at_utc": utc_now(),
        "notebook_session_clock": SessionBudget.from_environment(require_global_clock=True).snapshot(),
        "secondary_track": True,
    }
    write_run_record(record_path, record)
    append_segment_event(segments, {
        "segment_id": segment_id,
        "parent_run_id": args.run_id,
        "state": "START",
        "timestamp_utc": utc_now(),
        "source_git_sha": args.source_git_commit,
        "host_identity": host_identity(),
        "gpu_identity": record["hardware_identity"],
        "starting_checkpoint_present": resume,
    })

    budget = SessionBudget.from_environment(
        require_global_clock=True,
        hard_limit_seconds=float(args.session_hard_limit_seconds),
        finalization_margin_seconds=float(args.finalization_margin_seconds),
    )
    try:
        result = None
        if resume and mode == "scientific":
            result = completed_scientific_checkpoint_result(
                checkpoints,
                expected_identity=checkpoint_identity(identity),
                locked_epochs=int(ctc["schedule"]["epochs"]),
            )
        if result is None:
            result = run_training(
                student=student,
                teacher=teacher,
                projection=projection,
                train_dataset=train_ds,
                val_dataset=val_ds,
                ctc=ctc,
                objective=config["objective"],
                run_identity=identity,
                output_dir=checkpoints,
                num_workers=args.num_workers,
                resume=resume,
                max_optimizer_steps=max_optimizer_steps,
                validation_enabled=(mode == "scientific"),
                checkpoint_every_steps=args.checkpoint_every_steps,
                session_budget=budget,
                min_free_bytes=int(float(args.min_free_gb) * 1024**3),
            )

        segment_result = output / f"segment_{segment_id}.json"
        atomic_write_json(segment_result, result)
        persistence = None
        durable_sync_seconds = 0.0
        if store is not None:
            started = time.perf_counter()
            persistence = store.sync(checkpoints, run_id=args.run_id, segment_id=segment_id).to_dict()
            durable_sync_seconds = time.perf_counter() - started
        elif args.durable_required:
            raise RuntimeError("durable persistence is required but no durable store is configured")
        result["durable_sync_seconds"] = durable_sync_seconds

        if result.get("planned_rollover"):
            record.update({
                "status": "LAUNCHED",
                "continuation_required": True,
                "last_segment_result_sha256": sha256_file(segment_result),
                "persistence_status": persistence or {"status": "NOT_CONFIGURED"},
                "last_optimizer_step": result.get("optimizer_step_total"),
                "last_checkpoint_sha256": result.get("latest_checkpoint_sha256"),
            })
        elif mode == "calibration":
            record.update({
                "status": "PASS",
                "continuation_required": False,
                "result_summary": result,
                "last_segment_result_sha256": sha256_file(segment_result),
                "persistence_status": persistence or {"status": "NOT_CONFIGURED"},
                "finished_at_utc": utc_now(),
            })
        else:
            metrics = output / "metrics.json"
            atomic_write_json(metrics, result)
            locator = public_locator(args)
            summary_keys = (
                "mode", "optimizer_steps_segment", "optimizer_step_total", "examples_segment",
                "examples_total", "wall_seconds_segment", "sec_per_optimizer_step",
                "examples_per_second", "dataloader_wait_seconds", "dataloader_examples_per_wait_second",
                "peak_gpu_memory_bytes", "checkpoint_save_seconds", "checkpoint_load_seconds",
                "durable_sync_seconds", "selected_epoch", "selected_metrics",
                "selected_checkpoint_sha256", "latest_checkpoint_sha256", "validation_forward_benchmark",
            )
            record.update({
                "status": "PASS",
                "continuation_required": False,
                "result_summary": {k: result.get(k) for k in summary_keys},
                "artifact_locators": {
                    "metrics": {"basename": metrics.name, "sha256": sha256_file(metrics)},
                    "selected_checkpoint": {"sha256": result["selected_checkpoint_sha256"], "public_git": False, "durable_locator": locator},
                    "latest_checkpoint": {"sha256": result["latest_checkpoint_sha256"], "public_git": False, "durable_locator": locator},
                },
                "persistence_status": persistence or {"status": "NOT_CONFIGURED"},
                "finished_at_utc": utc_now(),
            })

        write_run_record(record_path, record)
        append_segment_event(segments, {
            "segment_id": segment_id,
            "parent_run_id": args.run_id,
            "state": "END",
            "timestamp_utc": utc_now(),
            "termination_reason": "ROLLOVER" if record.get("continuation_required") else "COMPLETE",
            "ending_checkpoint_sha": result.get("latest_checkpoint_sha256"),
            "ending_optimizer_step": result.get("optimizer_step_total"),
            "persistence_status": persistence,
        })
        return record
    except BaseException as exc:
        status = "INTERRUPTED" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "FAIL"
        record.update({
            "status": status,
            "continuation_required": False,
            "failure": {
                "type": type(exc).__name__,
                "reason": str(exc),
                "technical_retry_allowed_only_if_science_unchanged": True,
                "traceback_tail": traceback.format_exc()[-4000:],
            },
            "finished_at_utc": utc_now(),
        })
        write_run_record(record_path, record)
        raise


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--config", default="")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--class-map", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--secondary-g1-bundle", required=True)
    ap.add_argument("--g2-barrier", default="")
    ap.add_argument("--dependency-lock", default="journal_extension/locks/execution_dependency_lock.json")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--lane-id", choices=["K1", "K2", "K3"], required=True)
    ap.add_argument("--source-git-commit", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--row-id-column", required=True)
    ap.add_argument("--path-column", required=True)
    ap.add_argument("--split-column", required=True)
    ap.add_argument("--label-column", required=True)
    ap.add_argument("--class-index-column", default="")
    ap.add_argument("--train-split-value", default="train")
    ap.add_argument("--val-split-value", default="val")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--checkpoint-every-steps", type=int, default=250)
    ap.add_argument("--resume-mode", choices=["auto", "required", "never"], default="auto")
    ap.add_argument("--mode", choices=["scientific", "calibration"], default="scientific")
    ap.add_argument("--max-optimizer-steps", type=int, default=0)
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
        raise SystemExit("secondary scientific mode must run the full frozen 30 epochs")
    if args.mode == "calibration" and args.max_optimizer_steps <= 0:
        raise SystemExit("calibration mode requires --max-optimizer-steps")
    record = execute(
        args,
        mode=args.mode,
        max_optimizer_steps=(args.max_optimizer_steps or None),
        resume_mode=args.resume_mode,
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
