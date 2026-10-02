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
from cropcop_je.g1 import validate_dependency_environment, validate_dependency_lock_object
from cropcop_je.hashing import sha256_file, sha256_json
from cropcop_je.persistence import build_store
from cropcop_je.runlog import claim_run_directory, write_run_record
from cropcop_je.segments import append_segment_event, host_identity, new_segment_id, utc_now
from cropcop_je.session import SessionBudget
from cropcop_je.source_state import verify_clean_source
from cropcop_je.terminal_recovery import completed_scientific_checkpoint_result
from cropcop_je.tracka_v12 import (
    CLASS_MAP_SHA256,
    EXPERIMENT_SPECS,
    MANIFEST_SHA256,
    experiment_config_path,
    validate_tracka_v12_config,
)
from cropcop_je.tracka_v12_authorization import validate_science_authorization
from cropcop_je.tracka_v12_g2a import validate_g2a_barrier_object, validate_scheduler_freeze
from cropcop_je.tracka_v12_runtime import load_and_validate_g1a_bundle, load_student_and_teacher
from cropcop_je.train import _identity as checkpoint_identity
from cropcop_je.train import run_training

TRAIN_COUNT = 76376
VAL_COUNT = 16368
SLOT_IDS = ("K1/GPU0", "K1/GPU1", "K2/GPU0", "K2/GPU1", "K3/GPU0", "K3/GPU1")


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def software_identity() -> tuple[dict, str]:
    env = capture_environment()
    drift = validate_locked_core(env)
    if drift:
        raise RuntimeError(f"locked software identity mismatch: {json.dumps(drift, sort_keys=True)}")
    if env.get("cuda_available") is not True:
        raise RuntimeError("Track-A v1.2 execution requires CUDA")
    devices = env.get("torch_cuda_devices") or []
    if len(devices) != 1:
        raise RuntimeError(f"Track-A child must observe exactly one CUDA device, got {len(devices)}")
    name = str(devices[0].get("name", ""))
    if name not in {"Tesla T4", "NVIDIA T4"}:
        raise RuntimeError(f"Track-A child requires qualified T4, got {name!r}")
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
    output = Path(args.output_dir).resolve()
    verify_clean_source(repo, authorized_source_sha=args.source_git_commit, output_roots=[output])

    config_path = Path(args.config) if args.config else repo / experiment_config_path(args.experiment_id)
    if not config_path.is_absolute():
        config_path = repo / config_path
    config = load_json(config_path)
    errors = validate_tracka_v12_config(config)
    if errors:
        raise RuntimeError("Track-A v1.2 config invalid: " + "; ".join(errors))
    if config["experiment_id"] != args.experiment_id:
        raise RuntimeError("requested experiment/config mismatch")
    spec = EXPERIMENT_SPECS[args.experiment_id]

    env, stack_sha = software_identity()
    dep_path = Path(args.dependency_lock)
    if not dep_path.is_absolute():
        dep_path = repo / dep_path
    dep = load_json(dep_path)
    dep_errors = validate_dependency_lock_object(dep) + validate_dependency_environment(dep)
    if dep_errors:
        raise RuntimeError("dependency lock mismatch: " + "; ".join(dep_errors))

    g1a_seal, g1a_errors = load_and_validate_g1a_bundle(
        args.g1a_bundle,
        expected_source_sha=args.source_git_commit,
    )
    if g1a_errors:
        raise RuntimeError("Track-A G1A invalid: " + "; ".join(g1a_errors))
    if g1a_seal.get("dependency_lock_sha256") != dep.get("dependency_lock_sha256"):
        raise RuntimeError("G1A dependency lock differs from execution dependency lock")

    g2a_sha = None
    scheduler_sha = None
    authorization_sha = None
    if mode == "scientific":
        if not args.g2a_barrier or not args.scheduler_freeze or not args.science_authorization:
            raise RuntimeError("scientific mode requires G2A barrier, scheduler freeze and final science authorization")
        g2a = load_json(args.g2a_barrier)
        g2_errors = validate_g2a_barrier_object(
            g2a,
            expected_source_sha=args.source_git_commit,
            expected_g1a_seal_sha256=g1a_seal["g1a_seal_sha256"],
        )
        if g2_errors:
            raise RuntimeError("Track-A G2A invalid: " + "; ".join(g2_errors))
        if g2a.get("dependency_lock_sha256") != dep.get("dependency_lock_sha256"):
            raise RuntimeError("G2A dependency lock differs from execution dependency lock")
        g2a_sha = g2a["barrier_sha256"]

        scheduler = load_json(args.scheduler_freeze)
        scheduler_errors = validate_scheduler_freeze(scheduler, expected_g2a_barrier_sha256=g2a_sha)
        if scheduler_errors:
            raise RuntimeError("Track-A scheduler freeze invalid: " + "; ".join(scheduler_errors))
        if scheduler.get("source_git_commit") != args.source_git_commit:
            raise RuntimeError("scheduler freeze source SHA mismatch")
        if scheduler.get("g1a_seal_sha256") != g1a_seal["g1a_seal_sha256"]:
            raise RuntimeError("scheduler freeze G1A mismatch")
        if args.experiment_id not in set(scheduler.get("priority", [])):
            raise RuntimeError("experiment absent from frozen scheduler inventory")
        scheduler_sha = scheduler["scheduler_freeze_sha256"]

        authorization = load_json(args.science_authorization)
        auth_errors = validate_science_authorization(
            authorization,
            expected_source_sha=args.source_git_commit,
            expected_g1a_seal_sha256=g1a_seal["g1a_seal_sha256"],
            expected_g2a_barrier_sha256=g2a_sha,
            expected_scheduler_freeze_sha256=scheduler_sha,
        )
        if auth_errors:
            raise RuntimeError("Track-A science authorization invalid: " + "; ".join(auth_errors))
        if args.experiment_id not in set(authorization["authorized_experiment_ids"]):
            raise RuntimeError("experiment is not authorized by final science GO")
        authorization_sha = authorization["authorization_sha256"]

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

    loaded = load_student_and_teacher(
        repo_root=repo,
        config=config,
        g1a_bundle=args.g1a_bundle,
        expected_source_sha=args.source_git_commit,
    )
    ctc = load_json(repo / config["ctc_config"])
    seed = int(spec["seed"])
    train_ds = CropCopManifestDataset(train_rows, args.image_root, training_seed=seed, train=True)
    val_ds = CropCopManifestDataset(val_rows, args.image_root, training_seed=seed, train=False)

    identity = {
        "run_id": args.run_id,
        "experiment_id": config["experiment_id"],
        "authority_id": config["authority_id"],
        "source_git_commit": args.source_git_commit,
        "lane_id": args.slot_id,
        "config_sha256": sha256_json(config),
        "ctc_v2_sha256": sha256_json(ctc),
        "manifest_sha256": manifest_sha,
        "class_map_sha256": class_map_sha,
        "seed": seed,
        "student_init_sha256": loaded["student_init_sha256"],
        "pretrained_sha256": loaded["pretrained_sha256"],
        "teacher_sha256": loaded["teacher_sha256"],
        "teacher_factory_sha256": loaded["teacher_factory_sha256"],
        "teacher_factory_bundle_sha256": loaded["teacher_factory_bundle_sha256"],
        "software_stack_sha256": stack_sha,
        "dependency_lock_sha256": dep["dependency_lock_sha256"],
        "g1_seal_sha256": loaded["g1a_seal_sha256"],
        "g2_barrier_sha256": g2a_sha,
        "tracka_g1a_seal_sha256": loaded["g1a_seal_sha256"],
        "tracka_g2a_barrier_sha256": g2a_sha,
        "scheduler_freeze_sha256": scheduler_sha,
        "science_authorization_sha256": authorization_sha,
        "allowed_surfaces": ["DS-V1-TRAIN", "DS-V1-VAL"],
        "v1_test_accessed": False,
        "external_protected_surface_accessed": False,
    }
    return (
        config,
        ctc,
        loaded["student"],
        loaded["teacher"],
        loaded["projection"],
        train_ds,
        val_ds,
        identity,
        env,
    )


def execute(args, *, mode: str, max_optimizer_steps: int | None = None, resume_mode: str = "auto") -> dict:
    config, ctc, student, teacher, projection, train_ds, val_ds, identity, env = prepare(args, mode=mode)
    output = Path(args.output_dir)
    claim_run_directory(output, run_id=args.run_id, experiment_id=config["experiment_id"], lane_id=args.slot_id)
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
                    segments,
                    {
                        "segment_id": segment_id,
                        "parent_run_id": args.run_id,
                        "state": "RESTORE_FAILED",
                        "timestamp_utc": utc_now(),
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                if resume_mode == "required" or mode == "scientific":
                    raise RuntimeError("durable checkpoint restore failed; silent fresh scientific restart is forbidden") from exc
        resume = (checkpoints / "checkpoint_index.json").exists()
        if resume_mode == "required" and not resume:
            raise RuntimeError("resume required but no verified checkpoint is available")
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
            "slot_id": args.slot_id,
            "accelerator": (env.get("torch_cuda_devices") or [{}])[0].get("name"),
            "nvidia_smi": env.get("nvidia_smi"),
            "visible_cuda_device_count": len(env.get("torch_cuda_devices") or []),
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
        "track_a_v12_continuation": True,
        "calibration_weights_scientific": False if mode == "calibration" else None,
    }
    write_run_record(record_path, record)
    append_segment_event(
        segments,
        {
            "segment_id": segment_id,
            "parent_run_id": args.run_id,
            "state": "START",
            "timestamp_utc": utc_now(),
            "source_git_sha": args.source_git_commit,
            "host_identity": host_identity(),
            "gpu_identity": record["hardware_identity"],
            "starting_checkpoint_present": resume,
        },
    )

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
            record.update(
                {
                    "status": "LAUNCHED",
                    "continuation_required": True,
                    "last_segment_result_sha256": sha256_file(segment_result),
                    "persistence_status": persistence or {"status": "NOT_CONFIGURED"},
                    "last_optimizer_step": result.get("optimizer_step_total"),
                    "last_checkpoint_sha256": result.get("latest_checkpoint_sha256"),
                }
            )
        elif mode == "calibration":
            record.update(
                {
                    "status": "PASS",
                    "continuation_required": False,
                    "result_summary": result,
                    "last_segment_result_sha256": sha256_file(segment_result),
                    "persistence_status": persistence or {"status": "NOT_CONFIGURED"},
                    "finished_at_utc": utc_now(),
                    "validation_enabled": False,
                    "scientific_metric_computed": False,
                }
            )
        else:
            metrics = output / "metrics.json"
            atomic_write_json(metrics, result)
            locator = public_locator(args)
            summary_keys = (
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
            record.update(
                {
                    "status": "PASS",
                    "continuation_required": False,
                    "result_summary": {key: result.get(key) for key in summary_keys},
                    "artifact_locators": {
                        "metrics": {"basename": metrics.name, "sha256": sha256_file(metrics)},
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
            segments,
            {
                "segment_id": segment_id,
                "parent_run_id": args.run_id,
                "state": "END",
                "timestamp_utc": utc_now(),
                "termination_reason": "ROLLOVER" if record.get("continuation_required") else "COMPLETE",
                "ending_checkpoint_sha": result.get("latest_checkpoint_sha256"),
                "ending_optimizer_step": result.get("optimizer_step_total"),
                "persistence_status": persistence,
            },
        )
        return record
    except BaseException as exc:
        status = "INTERRUPTED" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "FAIL"
        record.update(
            {
                "status": status,
                "continuation_required": False,
                "failure": {
                    "type": type(exc).__name__,
                    "reason": str(exc),
                    "technical_retry_allowed_only_if_science_unchanged": True,
                    "traceback_tail": traceback.format_exc()[-4000:],
                },
                "finished_at_utc": utc_now(),
            }
        )
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
    ap.add_argument("--g1a-bundle", required=True)
    ap.add_argument("--g2a-barrier", default="")
    ap.add_argument("--scheduler-freeze", default="")
    ap.add_argument("--science-authorization", default="")
    ap.add_argument("--dependency-lock", default="journal_extension/locks/execution_dependency_lock.json")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--slot-id", choices=SLOT_IDS, required=True)
    ap.add_argument("--source-git-commit", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--row-id-column", default="record_key")
    ap.add_argument("--path-column", default="portable_relpath")
    ap.add_argument("--split-column", default="split")
    ap.add_argument("--label-column", default="label")
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
        raise SystemExit("scientific mode must run the full frozen 30 epochs")
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
