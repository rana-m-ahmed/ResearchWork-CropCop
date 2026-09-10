from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .hashing import sha256_json
from .secondary import BASELINE_SPECS, SECONDARY_CONFIG_SPECS, S1_SEED, TEACHER_SHA256

REQUIRED_SECONDARY_CALIBRATIONS = (
    "CAL-MNV4-DIRECT",
    "CAL-MNV4-TEACHER",
    "CAL-EFFB0",
    "CAL-CNXTT",
)
EXPECTED_COVERAGE = {
    "CAL-MNV4-DIRECT": set(),
    "CAL-MNV4-TEACHER": {"R12-MNV4-LOGITS-S1", "R12-MNV4-FEATURE-S1"},
    "CAL-EFFB0": {"R06-EFFB0-CONTEXT-S1"},
    "CAL-CNXTT": {"R07-CNXTT-CONTEXT-S1"},
}
TELEMETRY_FIELDS = (
    "sec_per_optimizer_step",
    "examples_per_second",
    "dataloader_wait_seconds",
    "peak_gpu_memory_bytes",
    "checkpoint_save_seconds",
    "checkpoint_load_seconds",
    "validation_forward_benchmark",
)


def secondary_barrier_hash(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("barrier_sha256", None)
    return sha256_json(clean)


def validate_secondary_calibration_summary(summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    cid = str(summary.get("calibration_id", ""))
    if cid not in REQUIRED_SECONDARY_CALIBRATIONS:
        return [f"unexpected secondary calibration ID: {cid}"]
    if summary.get("status") != "PASS":
        errors.append("secondary calibration is not PASS")
    if summary.get("calibration_weights_scientific") is not False:
        errors.append("secondary calibration weights must be explicitly non-scientific")
    if summary.get("resume_success") is not True:
        errors.append("secondary save/resume qualification failed")
    if summary.get("durable_roundtrip_success") is not True:
        errors.append("secondary durable checkpoint roundtrip failed")
    if summary.get("visible_cuda_device_count") != 1:
        errors.append("secondary calibration child did not observe exactly one visible CUDA device")
    if summary.get("visible_gpu_name") not in {"Tesla T4", "NVIDIA T4"}:
        errors.append("secondary calibration child did not observe a T4")
    if summary.get("git_credentials_present") is not False:
        errors.append("secondary calibration child inherited Git publication credentials")
    if summary.get("allowed_surfaces") != ["DS-V1-TRAIN", "DS-V1-VAL"]:
        errors.append("secondary calibration surface set drift")
    if summary.get("v1_test_accessed") is not False:
        errors.append("secondary calibration unexpectedly accessed V1 test")
    if summary.get("external_protected_surface_accessed") is not False:
        errors.append("secondary calibration unexpectedly accessed protected external data")
    for field in (
        "source_git_commit",
        "software_stack_sha256",
        "g1_seal_sha256",
        "dependency_lock_sha256",
        "mnv4_pretrained_sha256",
        "effb0_pretrained_sha256",
        "cnxtt_pretrained_sha256",
    ):
        value = str(summary.get(field, ""))
        if len(value) not in {40, 64}:
            errors.append(f"secondary calibration identity missing/invalid: {field}")
    if not str(summary.get("effb0_pretrained_sha256", "")).startswith(
        BASELINE_SPECS["effb0"]["official_sha256_prefix"]
    ):
        errors.append("secondary G2 EfficientNet-B0 pretrained identity mismatch")
    if not str(summary.get("cnxtt_pretrained_sha256", "")).startswith(
        BASELINE_SPECS["cnxtt"]["official_sha256_prefix"]
    ):
        errors.append("secondary G2 ConvNeXt-Tiny pretrained identity mismatch")
    if cid == "CAL-MNV4-TEACHER":
        if summary.get("teacher_checkpoint_sha256") != TEACHER_SHA256:
            errors.append("secondary teacher calibration checkpoint identity mismatch")
        if len(str(summary.get("teacher_factory_bundle_sha256", ""))) != 64:
            errors.append("secondary teacher calibration factory identity missing")
    expected_coverage = EXPECTED_COVERAGE[cid]
    if set(summary.get("coverage", [])) != expected_coverage:
        errors.append(f"secondary calibration coverage mismatch for {cid}")
    measured = summary.get("measured", {})
    for field in TELEMETRY_FIELDS:
        if measured.get(field) is None:
            errors.append(f"secondary calibration telemetry missing: {field}")
    durability = summary.get("durability", {})
    if durability.get("sync_status") != "PASS":
        errors.append("secondary calibration durable sync status is not PASS")
    if float(durability.get("sync_seconds", 0.0) or 0.0) <= 0:
        errors.append("secondary calibration durable sync duration missing")
    if float(durability.get("restore_seconds", 0.0) or 0.0) <= 0:
        errors.append("secondary calibration durable restore duration missing")
    return errors


def build_secondary_g2_barrier(
    summaries: list[dict[str, Any]],
    *,
    checkpoint_contract_probe: dict[str, Any],
    publication_idempotency: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_id = {str(s.get("calibration_id")): s for s in summaries}
    errors: list[str] = []
    missing = [cid for cid in REQUIRED_SECONDARY_CALIBRATIONS if cid not in by_id]
    errors.extend(f"missing secondary calibration: {cid}" for cid in missing)
    for cid in REQUIRED_SECONDARY_CALIBRATIONS:
        if cid in by_id:
            errors.extend(f"{cid}: {e}" for e in validate_secondary_calibration_summary(by_id[cid]))

    def common(field: str) -> set[Any]:
        return {by_id[cid].get(field) for cid in REQUIRED_SECONDARY_CALIBRATIONS if cid in by_id}

    common_fields = (
        "source_git_commit",
        "software_stack_sha256",
        "g1_seal_sha256",
        "dependency_lock_sha256",
        "mnv4_pretrained_sha256",
        "effb0_pretrained_sha256",
        "cnxtt_pretrained_sha256",
    )
    common_values: dict[str, Any] = {}
    for field in common_fields:
        values = common(field)
        if len(values) != 1:
            errors.append(f"secondary calibrations disagree on {field}: {sorted(str(v) for v in values)}")
            common_values[field] = None
        else:
            common_values[field] = next(iter(values))

    coverage = set()
    for summary in summaries:
        coverage.update(summary.get("coverage", []))
    if coverage != set(SECONDARY_CONFIG_SPECS):
        errors.append("secondary G2 calibration coverage does not cover exactly all four secondary states")

    if checkpoint_contract_probe.get("status") != "PASS":
        errors.append("secondary checkpoint contract probe is not PASS")
    if checkpoint_contract_probe.get("selected_checkpoint_verified") is not True:
        errors.append("secondary selected-checkpoint verification was not demonstrated")
    if checkpoint_contract_probe.get("identity_mismatch_rejected") is not True:
        errors.append("secondary checkpoint identity mismatch rejection was not demonstrated")

    forecast: dict[str, Any] = {}
    if not errors:
        optimizer_steps = 30 * math.ceil(math.ceil(76376 / 16) / 4)
        val_examples = 30 * 16368
        calibration_map = {
            "R12-MNV4-LOGITS-S1": "CAL-MNV4-TEACHER",
            "R12-MNV4-FEATURE-S1": "CAL-MNV4-TEACHER",
            "R06-EFFB0-CONTEXT-S1": "CAL-EFFB0",
            "R07-CNXTT-CONTEXT-S1": "CAL-CNXTT",
        }
        for eid, cid in calibration_map.items():
            measured = by_id[cid]["measured"]
            train_seconds = optimizer_steps * float(measured["sec_per_optimizer_step"])
            bench = measured["validation_forward_benchmark"]
            val_rate = float(bench.get("end_to_end_examples_per_second") or bench.get("forward_examples_per_second") or 0.0)
            validation_seconds = val_examples / max(val_rate, 1e-12)
            raw = train_seconds + validation_seconds
            forecast[eid] = {
                "calibration_id": cid,
                "optimizer_steps_per_30_epoch_run": optimizer_steps,
                "estimated_train_seconds": train_seconds,
                "estimated_validation_seconds": validation_seconds,
                "estimated_run_seconds_raw": raw,
                "estimated_run_seconds_with_20pct_operational_reserve": raw * 1.20,
                "forecast_is_scheduling_only": True,
            }

    teacher = by_id.get("CAL-MNV4-TEACHER", {})
    barrier = {
        "schema_version": "1.0",
        "barrier_kind": "secondary_track_a_g2",
        "status": "PASS" if not errors else "FAIL",
        "required_calibrations": list(REQUIRED_SECONDARY_CALIBRATIONS),
        **common_values,
        "teacher_checkpoint_sha256": teacher.get("teacher_checkpoint_sha256"),
        "teacher_factory_bundle_sha256": teacher.get("teacher_factory_bundle_sha256"),
        "s1_seed": S1_SEED,
        "covered_secondary_experiment_ids": sorted(coverage),
        "input_summary_sha256": {cid: sha256_json(by_id[cid]) for cid in REQUIRED_SECONDARY_CALIBRATIONS if cid in by_id},
        "checkpoint_contract_probe": checkpoint_contract_probe,
        "publication_idempotency": publication_idempotency or {"status": "NOT_RUN_YET"},
        "forecast": forecast,
        "errors": errors,
    }
    barrier["barrier_sha256"] = secondary_barrier_hash(barrier)
    return barrier


def validate_secondary_g2_barrier_object(
    barrier: dict[str, Any],
    *,
    expected_source_sha: str | None = None,
    expected_g1_seal_sha256: str | None = None,
) -> list[str]:
    errors: list[str] = []
    if barrier.get("schema_version") != "1.0" or barrier.get("barrier_kind") != "secondary_track_a_g2":
        errors.append("unsupported secondary G2 barrier schema/kind")
    if barrier.get("status") != "PASS" or barrier.get("errors"):
        errors.append("secondary G2 barrier is not terminal PASS")
    if barrier.get("barrier_sha256") != secondary_barrier_hash(barrier):
        errors.append("secondary G2 barrier self-hash mismatch")
    if expected_source_sha and barrier.get("source_git_commit") != expected_source_sha:
        errors.append("secondary G2 source SHA mismatch")
    if expected_g1_seal_sha256 and barrier.get("g1_seal_sha256") != expected_g1_seal_sha256:
        errors.append("secondary G2 G1-seal binding mismatch")
    if tuple(barrier.get("required_calibrations", [])) != REQUIRED_SECONDARY_CALIBRATIONS:
        errors.append("secondary G2 required calibration set/order mismatch")
    if set(barrier.get("covered_secondary_experiment_ids", [])) != set(SECONDARY_CONFIG_SPECS):
        errors.append("secondary G2 experiment coverage mismatch")
    if barrier.get("checkpoint_contract_probe", {}).get("status") != "PASS":
        errors.append("secondary G2 checkpoint contract probe missing/failing")
    for field in (
        "dependency_lock_sha256",
        "software_stack_sha256",
        "mnv4_pretrained_sha256",
        "effb0_pretrained_sha256",
        "cnxtt_pretrained_sha256",
    ):
        if len(str(barrier.get(field, ""))) != 64:
            errors.append(f"secondary G2 {field} missing/invalid")
    if not str(barrier.get("effb0_pretrained_sha256", "")).startswith(BASELINE_SPECS["effb0"]["official_sha256_prefix"]):
        errors.append("secondary G2 EfficientNet-B0 pretrained identity invalid")
    if not str(barrier.get("cnxtt_pretrained_sha256", "")).startswith(BASELINE_SPECS["cnxtt"]["official_sha256_prefix"]):
        errors.append("secondary G2 ConvNeXt-Tiny pretrained identity invalid")
    if barrier.get("teacher_checkpoint_sha256") != TEACHER_SHA256:
        errors.append("secondary G2 teacher checkpoint identity mismatch")
    if len(str(barrier.get("teacher_factory_bundle_sha256", ""))) != 64:
        errors.append("secondary G2 teacher factory identity missing")
    return errors


def write_secondary_g2_barrier(path: str | Path, summaries: list[dict[str, Any]], *, checkpoint_contract_probe: dict[str, Any], publication_idempotency: dict[str, Any] | None = None) -> dict[str, Any]:
    barrier = build_secondary_g2_barrier(summaries, checkpoint_contract_probe=checkpoint_contract_probe, publication_idempotency=publication_idempotency)
    Path(path).write_text(__import__("json").dumps(barrier, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return barrier


def run_checkpoint_contract_probe(root: str | Path) -> dict[str, Any]:
    import shutil
    from datetime import datetime, timezone
    import torch
    from .checkpointing import save_torch_checkpoint, verify_selected
    from .checkpointing import CheckpointCorruptionError

    probe_root = Path(root) / "checkpoint_contract_probe"
    if probe_root.exists():
        shutil.rmtree(probe_root)
    identity = {
        "experiment_id": "SEC-G2-CHECKPOINT-CONTRACT", "authority_id": "EAAI-JE-SDL-v2.1-QA",
        "source_git_commit": "0" * 40, "config_sha256": "1" * 64, "ctc_v2_sha256": "2" * 64,
        "manifest_sha256": "3" * 64, "class_map_sha256": "4" * 64, "seed": S1_SEED,
        "student_init_sha256": "5" * 64, "pretrained_sha256": "6" * 64, "teacher_sha256": None,
        "teacher_factory_sha256": None, "teacher_factory_bundle_sha256": None,
        "software_stack_sha256": "7" * 64, "dependency_lock_sha256": "8" * 64,
        "g1_seal_sha256": "9" * 64, "g2_barrier_sha256": None, "lane_id": "K1",
    }
    payload = {
        "schema_version": "2.0", "identity": identity, "identity_sha256": sha256_json(identity),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "student": {"probe": torch.arange(4, dtype=torch.float32)}, "projection": None,
        "optimizer": {}, "scheduler": {}, "scaler": {},
        "rng": {"python": None, "numpy": None, "torch": None, "cuda": None},
        "epoch": 1, "batch_in_epoch": 0, "optimizer_step": 1, "examples_seen": 1,
        "data_order_state": {"epoch": 1, "next_batch_in_epoch": 0},
        "selection_state": {"history": [], "best": None},
    }
    latest, _ = save_torch_checkpoint(probe_root, kind="latest", payload=payload, expected_identity=identity)
    selected, _ = save_torch_checkpoint(probe_root, kind="selected", payload=payload, expected_identity=identity)
    verify_selected(probe_root, expected_identity=identity, expected_sha256=selected.sha256)
    mismatch_rejected = False
    wrong = dict(identity)
    wrong["seed"] = S1_SEED + 1
    try:
        verify_selected(probe_root, expected_identity=wrong)
    except CheckpointCorruptionError:
        mismatch_rejected = True
    result = {
        "status": "PASS" if mismatch_rejected else "FAIL",
        "latest_checkpoint_sha256": latest.sha256,
        "selected_checkpoint_sha256": selected.sha256,
        "selected_checkpoint_verified": True,
        "identity_mismatch_rejected": mismatch_rejected,
        "scientific_result_produced": False,
        "protected_data_accessed": False,
    }
    shutil.rmtree(probe_root)
    return result
