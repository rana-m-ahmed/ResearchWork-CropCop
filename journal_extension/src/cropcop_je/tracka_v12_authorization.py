from __future__ import annotations

from typing import Any

from .hashing import sha256_json
from .tracka_v12 import EXPERIMENT_SPECS


class TrackAV12AuthorizationError(RuntimeError):
    pass


REQUIRED_PRE_SCIENCE_GATES = frozenset(
    {
        "exact_head_ci",
        "immutable_v12_lock",
        "v121_runtime_qualification",
        "config_contract_and_failure_injection",
        "candidate_claim_boundary_lock",
        "principal_science_diff",
        "secondary_science_diff",
        "g1a",
        "g2a",
        "scheduler_freeze",
        "training_runner_contract",
        "checkpoint_recovery_contract",
        "cross_slot_recovery_contract",
        "six_gpu_parent_orchestration",
        "selected_checkpoint_replay_implementation",
        "classwise_confusion_implementation",
        "robustness_executor_implementation",
        "efficiency_profiler_implementation",
        "xai_gradcampp_implementation",
        "analysis_selector_implementation",
        "final_selection_closure_implementation",
    }
)


def authorization_hash(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("authorization_sha256", None)
    return sha256_json(clean)


def validate_science_authorization(
    authorization: dict[str, Any],
    *,
    expected_source_sha: str,
    expected_g1a_seal_sha256: str,
    expected_g2a_barrier_sha256: str,
    expected_scheduler_freeze_sha256: str,
) -> list[str]:
    errors: list[str] = []
    if authorization.get("schema_version") != "1.1" or authorization.get("authorization_kind") != "track_a_v12_science_go":
        errors.append("unsupported Track-A science authorization schema/kind")
    if authorization.get("status") != "GO":
        errors.append("Track-A science authorization is not GO")
    if authorization.get("source_git_commit") != expected_source_sha:
        errors.append("Track-A science authorization source SHA mismatch")
    if authorization.get("g1a_seal_sha256") != expected_g1a_seal_sha256:
        errors.append("Track-A science authorization G1A mismatch")
    if authorization.get("g2a_barrier_sha256") != expected_g2a_barrier_sha256:
        errors.append("Track-A science authorization G2A mismatch")
    if authorization.get("scheduler_freeze_sha256") != expected_scheduler_freeze_sha256:
        errors.append("Track-A science authorization scheduler mismatch")
    if set(authorization.get("authorized_experiment_ids", [])) != set(EXPERIMENT_SPECS):
        errors.append("Track-A science authorization experiment inventory mismatch")
    if authorization.get("v1_test_closed") is not True:
        errors.append("Track-A science authorization does not affirm V1-test closure")
    if authorization.get("external_predictions_closed") is not True:
        errors.append("Track-A science authorization does not affirm external-prediction closure")
    if authorization.get("track_c_candidate_runtime_closed") is not True:
        errors.append("Track-A science authorization does not affirm Track-C result closure")

    gates = authorization.get("pre_science_gates", {})
    if set(gates) != REQUIRED_PRE_SCIENCE_GATES:
        errors.append("Track-A science authorization gate inventory mismatch")
    elif any(gates[name] != "PASS" for name in REQUIRED_PRE_SCIENCE_GATES):
        errors.append("Track-A science authorization contains a non-PASS pre-science gate")

    bindings = authorization.get("evidence_bindings", {})
    if set(bindings) != REQUIRED_PRE_SCIENCE_GATES:
        errors.append("Track-A science authorization evidence-binding inventory mismatch")
    else:
        empty = sorted(name for name in REQUIRED_PRE_SCIENCE_GATES if bindings.get(name) in (None, "", {}, []))
        if empty:
            errors.append("Track-A science authorization has empty evidence bindings: " + ", ".join(empty))

    if authorization.get("authorization_sha256") != authorization_hash(authorization):
        errors.append("Track-A science authorization self-hash mismatch")
    return errors


def build_science_authorization(
    *,
    source_git_commit: str,
    g1a_seal_sha256: str,
    g2a_barrier_sha256: str,
    scheduler_freeze_sha256: str,
    pre_science_gates: dict[str, str],
    evidence_bindings: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": "1.1",
        "authorization_kind": "track_a_v12_science_go",
        "status": "GO",
        "source_git_commit": source_git_commit,
        "g1a_seal_sha256": g1a_seal_sha256,
        "g2a_barrier_sha256": g2a_barrier_sha256,
        "scheduler_freeze_sha256": scheduler_freeze_sha256,
        "authorized_experiment_ids": sorted(EXPERIMENT_SPECS),
        "v1_test_closed": True,
        "external_predictions_closed": True,
        "track_c_candidate_runtime_closed": True,
        "pre_science_gates": dict(pre_science_gates),
        "evidence_bindings": dict(evidence_bindings),
        "note": (
            "This GO authorizes only the frozen eleven Track-A v1.2 continuation training states, including "
            "R13 S1/S2/S3. It is valid only after the candidate-comparison claim boundary and the complete "
            "training, recovery, six-GPU orchestration and post-training inference/analysis implementations "
            "are already frozen and qualified. It does not authorize new models, seeds, hyperparameters, "
            "V1-test access, external prediction opening or Track-C candidate result opening."
        ),
    }
    payload["authorization_sha256"] = authorization_hash(payload)
    errors = validate_science_authorization(
        payload,
        expected_source_sha=source_git_commit,
        expected_g1a_seal_sha256=g1a_seal_sha256,
        expected_g2a_barrier_sha256=g2a_barrier_sha256,
        expected_scheduler_freeze_sha256=scheduler_freeze_sha256,
    )
    if errors:
        raise TrackAV12AuthorizationError("invalid science authorization: " + "; ".join(errors))
    return payload
