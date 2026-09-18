from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .checkpointing import read_index
from .hashing import sha256_file, sha256_json
from .tracka_v12 import (
    AUTHORITY_ID,
    CLASS_MAP_SHA256,
    EXPERIMENT_SPECS,
    MANIFEST_SHA256,
    experiment_config_path,
    validate_tracka_v12_config,
)
from .tracka_v12_placement import checkpoint_identity_projection, logical_lane_id

SCIENCE_SOURCE_SHA = "56023042e57758591df9babb3438f191dbe10312"
SCIENCE_AUTHORIZATION_SHA256 = "58d65a9c9f0c06222c00541feca9b2aaa005ebda850d2261d4df3e0e1fe7fbb7"
SCHEDULER_FREEZE_SHA256 = "7c7eba73096c13eb7fa754143b9fe84548c16ac05ab80f6aa3e1aeeb128c3787"


class TerminalRecoveryError(RuntimeError):
    pass


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TerminalRecoveryError(f"JSON object required: {path}")
    return payload


def account_result(report: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    if report.get("status") != "PASS" or report.get("science_complete") is not True:
        raise TerminalRecoveryError("account report is not terminal PASS/science_complete")
    if report.get("science_source_sha") != SCIENCE_SOURCE_SHA:
        raise TerminalRecoveryError("account report science source mismatch")
    if report.get("science_authorization_sha256") != SCIENCE_AUTHORIZATION_SHA256:
        raise TerminalRecoveryError("account report science authorization mismatch")
    if report.get("scheduler_freeze_sha256") != SCHEDULER_FREEZE_SHA256:
        raise TerminalRecoveryError("account report scheduler freeze mismatch")
    if report.get("cross_gpu_gradient_synchronization") is not False:
        raise TerminalRecoveryError("account report indicates cross-GPU synchronization")
    if report.get("child_git_credentials_removed") is not True:
        raise TerminalRecoveryError("account report does not prove child Git credentials were removed")
    matches: list[dict[str, Any]] = []
    for rows in (report.get("slot_results") or {}).values():
        for row in rows or []:
            if row.get("experiment_id") == experiment_id:
                matches.append(row)
    if len(matches) != 1:
        raise TerminalRecoveryError(f"expected exactly one account result for {experiment_id}, got {len(matches)}")
    row = matches[0]
    if row.get("return_code") != 0 or row.get("run_status") != "PASS":
        raise TerminalRecoveryError(f"account result is not terminal PASS for {experiment_id}")
    if row.get("continuation_required") is not False:
        raise TerminalRecoveryError(f"account result still requires continuation for {experiment_id}")
    selected = str(row.get("selected_checkpoint_sha256", ""))
    if len(selected) != 64:
        raise TerminalRecoveryError(f"account result lacks a selected checkpoint SHA for {experiment_id}")
    run_id = str(row.get("run_id", ""))
    if not run_id:
        raise TerminalRecoveryError(f"account result lacks run_id for {experiment_id}")
    return row


def validate_checkpoint_payload(
    *,
    repo_root: str | Path,
    experiment_id: str,
    run_id: str,
    selected_ref: dict[str, Any],
    payload: dict[str, Any],
    expected_selected_sha256: str,
) -> dict[str, Any]:
    if experiment_id not in EXPERIMENT_SPECS:
        raise TerminalRecoveryError(f"not a frozen v1.2 continuation state: {experiment_id}")
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        raise TerminalRecoveryError("selected checkpoint lacks scientific identity")
    if payload.get("identity_sha256") != sha256_json(identity):
        raise TerminalRecoveryError("selected checkpoint identity digest mismatch")
    if selected_ref.get("sha256") != expected_selected_sha256:
        raise TerminalRecoveryError("checkpoint index selected SHA differs from public terminal account report")
    if int(selected_ref.get("epoch", -1)) != int(payload.get("epoch", -2)):
        raise TerminalRecoveryError("selected checkpoint index/payload epoch mismatch")
    if int(selected_ref.get("optimizer_step", -1)) != int(payload.get("optimizer_step", -2)):
        raise TerminalRecoveryError("selected checkpoint index/payload optimizer-step mismatch")

    repo = Path(repo_root).resolve()
    config_path = repo / experiment_config_path(experiment_id)
    config = _load_json(config_path)
    config_errors = validate_tracka_v12_config(config)
    if config_errors:
        raise TerminalRecoveryError("frozen config invalid: " + "; ".join(config_errors))
    ctc_path = repo / str(config["ctc_config"])
    ctc = _load_json(ctc_path)
    spec = EXPERIMENT_SPECS[experiment_id]
    expected = {
        "experiment_id": experiment_id,
        "authority_id": AUTHORITY_ID,
        "source_git_commit": SCIENCE_SOURCE_SHA,
        "config_sha256": sha256_json(config),
        "ctc_v2_sha256": sha256_json(ctc),
        "manifest_sha256": MANIFEST_SHA256,
        "class_map_sha256": CLASS_MAP_SHA256,
        "seed": int(spec["seed"]),
    }
    expected_identity_keys = set(checkpoint_identity_projection({}))
    if set(identity) != expected_identity_keys:
        missing = sorted(expected_identity_keys - set(identity))
        extra = sorted(set(identity) - expected_identity_keys)
        raise TerminalRecoveryError(
            f"checkpoint identity schema drift: missing={missing}, extra={extra}"
        )
    for field, value in expected.items():
        if identity.get(field) != value:
            raise TerminalRecoveryError(f"checkpoint scientific identity mismatch: {field}")
    if identity.get("lane_id") != logical_lane_id(experiment_id):
        raise TerminalRecoveryError("checkpoint logical lane does not bind experiment identity")

    # Production checkpoints intentionally serialize only checkpoint_identity_projection().
    # Surface-access fields belong to the full run record and are not checkpoint fields.
    # Recover their contract from the frozen config/source instead of treating absence as access.
    if config.get("train_surface") != "DS-V1-TRAIN":
        raise TerminalRecoveryError("frozen config training surface drift")
    if config.get("validation_surface") != "DS-V1-VAL":
        raise TerminalRecoveryError("frozen config validation surface drift")
    required_forbidden = {"DS-V1-TEST-CONSUMED", "DS-EXT-*-SEALED", "DS-HIST-COMPARE"}
    forbidden = set(config.get("forbidden_surfaces") or [])
    if not required_forbidden.issubset(forbidden):
        raise TerminalRecoveryError("frozen config protected-surface policy drift")

    selection = payload.get("selection_state") or {}
    best = selection.get("best") or {}
    metrics = best.get("metrics")
    if not isinstance(metrics, dict):
        raise TerminalRecoveryError("selected checkpoint lacks validation-selection metrics")
    required_metrics = {
        "validation_accuracy",
        "validation_balanced_accuracy",
        "validation_macro_f1",
        "validation_nll",
    }
    if not required_metrics.issubset(metrics):
        raise TerminalRecoveryError("selected checkpoint validation-selection metrics are incomplete")
    best_sha = best.get("checkpoint_sha256")
    if best_sha not in {None, expected_selected_sha256}:
        raise TerminalRecoveryError("selection state binds a different selected checkpoint SHA")
    selected_epoch = int(best.get("epoch", payload.get("epoch", -1)))
    if selected_epoch <= 0:
        raise TerminalRecoveryError("selected epoch is invalid")

    return {
        "identity": identity,
        "identity_sha256": payload["identity_sha256"],
        "selected_epoch": selected_epoch,
        "selected_metrics": {name: float(metrics[name]) for name in sorted(required_metrics)},
        "selected_checkpoint_sha256": expected_selected_sha256,
        "selected_checkpoint_epoch": int(payload["epoch"]),
        "optimizer_step": int(payload["optimizer_step"]),
        "config_sha256": expected["config_sha256"],
        "ctc_v2_sha256": expected["ctc_v2_sha256"],
        "run_id": run_id,
        "surface_contract": {
            "allowed_surfaces": ["DS-V1-TRAIN", "DS-V1-VAL"],
            "forbidden_surfaces": sorted(required_forbidden),
            "checkpoint_identity_contains_surface_flags": False,
            "claim_basis": "frozen_training_runner_and_config_contract",
        },
    }


def load_selected_checkpoint_evidence(
    *,
    repo_root: str | Path,
    checkpoint_root: str | Path,
    experiment_id: str,
    run_id: str,
    expected_selected_sha256: str,
) -> dict[str, Any]:
    root = Path(checkpoint_root).resolve()
    index_path = root / "checkpoint_index.json"
    if not index_path.is_file():
        raise TerminalRecoveryError("checkpoint index is missing")
    index = read_index(root)
    selected = index.get("selected")
    if not isinstance(selected, dict):
        raise TerminalRecoveryError("checkpoint index lacks selected checkpoint")
    rel = str(selected.get("relative_path", ""))
    path = root / rel
    if not rel or not path.is_file():
        raise TerminalRecoveryError("selected checkpoint file is missing")
    if path.stat().st_size != int(selected.get("bytes", -1)):
        raise TerminalRecoveryError("selected checkpoint byte-size mismatch")
    observed_sha = sha256_file(path)
    if observed_sha != expected_selected_sha256 or selected.get("sha256") != expected_selected_sha256:
        raise TerminalRecoveryError("selected checkpoint SHA mismatch")

    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TerminalRecoveryError("selected checkpoint payload is not a mapping")
    evidence = validate_checkpoint_payload(
        repo_root=repo_root,
        experiment_id=experiment_id,
        run_id=run_id,
        selected_ref=selected,
        payload=payload,
        expected_selected_sha256=expected_selected_sha256,
    )
    evidence.update(
        {
            "checkpoint_index_sha256": sha256_file(index_path),
            "selected_checkpoint_file_sha256": observed_sha,
            "selected_checkpoint_relative_path": rel,
        }
    )
    return evidence


def build_recovered_terminal_record(
    *,
    experiment_id: str,
    account_report: dict[str, Any],
    checkpoint_evidence: dict[str, Any],
    account_report_sha256: str,
    durable_locator: str | None,
) -> dict[str, Any]:
    row = account_result(account_report, experiment_id)
    if row["run_id"] != checkpoint_evidence["run_id"]:
        raise TerminalRecoveryError("public account run_id differs from checkpoint recovery run_id")
    if row["selected_checkpoint_sha256"] != checkpoint_evidence["selected_checkpoint_sha256"]:
        raise TerminalRecoveryError("public account selected SHA differs from recovered checkpoint")
    identity = dict(checkpoint_evidence["identity"])
    record: dict[str, Any] = {
        **identity,
        "run_id": row["run_id"],
        "status": "PASS",
        "mode": "scientific",
        "continuation_required": False,
        "allowed_surfaces": list(checkpoint_evidence["surface_contract"]["allowed_surfaces"]),
        "v1_test_accessed": False,
        "external_protected_surface_accessed": False,
        "protected_external_surface_accessed": False,
        "artifact_locators": {
            "selected_checkpoint": {
                "sha256": checkpoint_evidence["selected_checkpoint_sha256"],
                "public_git": False,
                "durable_locator": durable_locator,
            }
        },
        "result_summary": {
            "selected_epoch": checkpoint_evidence["selected_epoch"],
            "selected_metrics": checkpoint_evidence["selected_metrics"],
            "selected_checkpoint_sha256": checkpoint_evidence["selected_checkpoint_sha256"],
        },
        "recovery_provenance": {
            "schema_version": "1.0",
            "kind": "cryptographic_terminal_record_recovery",
            "original_public_run_record_available": False,
            "scientific_training_reperformed": False,
            "optimizer_state_advanced": False,
            "public_account_report_sha256": account_report_sha256,
            "checkpoint_index_sha256": checkpoint_evidence["checkpoint_index_sha256"],
            "selected_checkpoint_file_sha256": checkpoint_evidence["selected_checkpoint_file_sha256"],
            "checkpoint_identity_sha256": checkpoint_evidence["identity_sha256"],
            "science_source_sha": SCIENCE_SOURCE_SHA,
            "science_authorization_sha256": SCIENCE_AUTHORIZATION_SHA256,
            "scheduler_freeze_sha256": SCHEDULER_FREEZE_SHA256,
            "surface_safety": {
                **checkpoint_evidence["surface_contract"],
                "reconstructed_fields": [
                    "allowed_surfaces",
                    "v1_test_accessed",
                    "external_protected_surface_accessed",
                    "protected_external_surface_accessed",
                ],
            },
            "statement": (
                "This terminal record was reconstructed only from the terminal public account report and "
                "the hash/identity-verified selected scientific checkpoint because per-run Git publication "
                "failed after training. No metric was recomputed or inferred from account-level summaries."
            ),
        },
    }
    return record


def validate_recovered_terminal_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    experiment_id = str(record.get("experiment_id", ""))
    if experiment_id not in EXPERIMENT_SPECS:
        errors.append("experiment_id")
    if record.get("status") != "PASS" or record.get("mode") != "scientific":
        errors.append("terminal_status")
    if record.get("continuation_required") is not False:
        errors.append("continuation_required")
    if record.get("source_git_commit") != SCIENCE_SOURCE_SHA:
        errors.append("source_git_commit")
    if record.get("v1_test_accessed") is not False:
        errors.append("v1_test_accessed")
    if record.get("external_protected_surface_accessed") is not False:
        errors.append("external_protected_surface_accessed")
    if record.get("protected_external_surface_accessed") is not False:
        errors.append("protected_external_surface_accessed")
    if record.get("allowed_surfaces") != ["DS-V1-TRAIN", "DS-V1-VAL"]:
        errors.append("allowed_surfaces")
    selected_artifact = (record.get("artifact_locators") or {}).get("selected_checkpoint") or {}
    selected_result = (record.get("result_summary") or {}).get("selected_checkpoint_sha256")
    if not selected_artifact.get("sha256") or selected_artifact.get("sha256") != selected_result:
        errors.append("selected_checkpoint_binding")
    provenance = record.get("recovery_provenance") or {}
    if provenance.get("kind") != "cryptographic_terminal_record_recovery":
        errors.append("recovery_provenance")
    if provenance.get("scientific_training_reperformed") is not False:
        errors.append("training_reperformed")
    if provenance.get("optimizer_state_advanced") is not False:
        errors.append("optimizer_state_advanced")
    surface_safety = provenance.get("surface_safety") or {}
    if surface_safety.get("claim_basis") != "frozen_training_runner_and_config_contract":
        errors.append("surface_safety:claim_basis")
    if surface_safety.get("checkpoint_identity_contains_surface_flags") is not False:
        errors.append("surface_safety:checkpoint_identity_schema")
    if surface_safety.get("allowed_surfaces") != ["DS-V1-TRAIN", "DS-V1-VAL"]:
        errors.append("surface_safety:allowed_surfaces")
    metrics = (record.get("result_summary") or {}).get("selected_metrics") or {}
    for name in ("validation_accuracy", "validation_balanced_accuracy", "validation_macro_f1", "validation_nll"):
        if name not in metrics:
            errors.append(f"selected_metrics:{name}")
    return errors
