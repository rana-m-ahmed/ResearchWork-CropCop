from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import master_control_v8 as legacy
from tracka_v12_kaggle_operator_v8 import SCIENCE_SHA, OperatorError

CODE_EXTRA_STATIC_GATES = frozenset(
    {
        "g1a_runtime_global_resolution_contract",
        "kaggle_generation_durability_contract",
        "r13_parity_amendment_contract",
        "teacher_factory_root_binding_contract",
    }
)
LOCK_RUNTIME_QUALIFICATION_FIELD = "r13_v121_runtime_qualification"
AUTH_RUNTIME_QUALIFICATION_GATE = "v121_runtime_qualification"
COMPAT_SCHEMA = "1.1"


def _ensure_science_src(repo: Path) -> None:
    src = repo / "journal_extension" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _valid_actions_run_id(value: Any) -> bool:
    text = str(value or "")
    return text.isdigit() and int(text) > 0


def _verify_embedded_hash(payload: dict[str, Any], field: str) -> bool:
    from cropcop_je.hashing import sha256_json
    clean = dict(payload)
    observed = clean.pop(field, None)
    return isinstance(observed, str) and len(observed) == 64 and observed == sha256_json(clean)


def _validate_attestation_provenance(payload: dict[str, Any], *, expected_kind: str, source_git_commit: str, label: str) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != "1.1": errors.append(f"{label} schema version mismatch")
    if payload.get("attestation_kind") != expected_kind: errors.append(f"{label} kind mismatch")
    if payload.get("github_actions") is not True: errors.append(f"{label} is not marked as GitHub Actions emitted")
    if payload.get("pull_request_head_sha") != source_git_commit: errors.append(f"{label} PR-head SHA mismatch")
    if payload.get("source_git_commit") != source_git_commit: errors.append(f"{label} source SHA mismatch")
    if not _valid_actions_run_id(payload.get("workflow_run_id")): errors.append(f"{label} workflow run ID is missing/invalid")
    if not _valid_actions_run_id(payload.get("workflow_run_attempt")): errors.append(f"{label} workflow run attempt is missing/invalid")
    return errors


def compose_gate_inputs_v12(repo: Path, *, source_git_commit: str, code_attestation: dict[str, Any], lock_runtime_attestation: dict[str, Any], g1a: dict[str, Any], g2a: dict[str, Any], scheduler: dict[str, Any], file_hashes: dict[str, str]) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    _ensure_science_src(repo)
    from cropcop_je.tracka_v12_authorization import REQUIRED_PRE_SCIENCE_GATES
    from cropcop_je.tracka_v12_g1a_v121 import validate_g1a_seal_object
    from cropcop_je.tracka_v12_g2a_durability import validate_g2a_durability_contract
    from cropcop_je.tracka_v12_g2a_v122 import validate_g2a_v122_barrier, validate_scheduler_freeze_v122

    dynamic_gates = {"immutable_v12_lock", AUTH_RUNTIME_QUALIFICATION_GATE, "candidate_claim_boundary_lock", "g1a", "g2a", "scheduler_freeze"}
    static_authorization_gates = set(REQUIRED_PRE_SCIENCE_GATES) - dynamic_gates
    expected_code_inventory = static_authorization_gates | set(CODE_EXTRA_STATIC_GATES)
    errors: list[str] = []

    errors.extend(_validate_attestation_provenance(code_attestation, expected_kind="track_a_v12_pre_science_code", source_git_commit=source_git_commit, label="code attestation"))
    if code_attestation.get("status") != "PASS": errors.append("code attestation is not PASS")
    if not _verify_embedded_hash(code_attestation, "attestation_sha256"): errors.append("code attestation self-hash mismatch")
    static = code_attestation.get("static_pre_science_gates") or {}
    if set(static) != expected_code_inventory:
        missing = sorted(expected_code_inventory - set(static)); unexpected = sorted(set(static) - expected_code_inventory)
        errors.append(f"code attestation static gate inventory mismatch after exact hardening missing={missing} unexpected={unexpected}")
    else:
        nonpass = sorted(name for name in expected_code_inventory if static.get(name) != "PASS")
        if nonpass: errors.append("code attestation contains non-PASS static gates: " + ", ".join(nonpass))
    science_diff_reports = code_attestation.get("science_diff_reports") or {}
    if set(science_diff_reports) != {"principal", "secondary"}: errors.append("code attestation science-diff report inventory mismatch")
    else:
        for name in ("principal", "secondary"):
            if (science_diff_reports.get(name) or {}).get("status") != "PASS": errors.append(f"code attestation {name} science-diff report is not PASS")
    science_diff_sha = code_attestation.get("science_diff_sha256") or {}
    if set(science_diff_sha) != {"principal", "secondary"} or any(len(str(science_diff_sha.get(name, ""))) != 64 for name in ("principal", "secondary")): errors.append("code attestation science-diff SHA inventory invalid")

    errors.extend(_validate_attestation_provenance(lock_runtime_attestation, expected_kind="track_a_v12_exact_head_lock_runtime", source_git_commit=source_git_commit, label="lock/runtime attestation"))
    if lock_runtime_attestation.get("status") != "PASS": errors.append("lock/runtime attestation is not PASS")
    if not _verify_embedded_hash(lock_runtime_attestation, "attestation_sha256"): errors.append("lock/runtime attestation self-hash mismatch")
    if lock_runtime_attestation.get("science_authorized") is not False: errors.append("lock/runtime attestation unexpectedly authorizes science")
    if lock_runtime_attestation.get("immutable_v12_lock") != "PASS": errors.append("lock/runtime attestation gate is not PASS: immutable_v12_lock")
    if lock_runtime_attestation.get(LOCK_RUNTIME_QUALIFICATION_FIELD) != "PASS": errors.append("lock/runtime attestation gate is not PASS: " + LOCK_RUNTIME_QUALIFICATION_FIELD)
    if lock_runtime_attestation.get("candidate_claim_boundary_lock") != "PASS": errors.append("lock/runtime attestation gate is not PASS: candidate_claim_boundary_lock")
    content_report = lock_runtime_attestation.get("content_lock_report") or {}; runtime_report = lock_runtime_attestation.get("runtime_report") or {}
    if content_report.get("overall_status") != "PASS" or (content_report.get("static") or {}).get("status") != "PASS": errors.append("embedded immutable content-lock report is not PASS")
    if runtime_report.get("status") != "PASS" or runtime_report.get("science_authorized") is not False: errors.append("embedded R13 runtime report is not pre-science PASS")
    if runtime_report.get("qualified_target") != "blocks.13.norm1": errors.append("embedded R13 runtime target drift")

    g1_errors = validate_g1a_seal_object(g1a)
    if g1_errors: errors.extend(f"G1A: {error}" for error in g1_errors)
    if g1a.get("status") != "PASS" or g1a.get("science_authorized") is not False: errors.append("G1A is not pre-science PASS")
    if g1a.get("source_git_sha") != source_git_commit: errors.append("G1A source SHA mismatch")
    g2_errors = validate_g2a_v122_barrier(g2a, expected_source_sha=source_git_commit, expected_g1a_seal_sha256=g1a.get("g1a_seal_sha256"))
    if g2_errors: errors.extend(f"G2A: {error}" for error in g2_errors)
    scheduler_errors = validate_scheduler_freeze_v122(scheduler, expected_g2a_barrier_sha256=g2a.get("barrier_sha256"))
    if scheduler_errors: errors.extend(f"scheduler: {error}" for error in scheduler_errors)
    if scheduler.get("source_git_commit") != source_git_commit: errors.append("scheduler source SHA mismatch")
    if scheduler.get("g1a_seal_sha256") != g1a.get("g1a_seal_sha256"): errors.append("scheduler/G1A binding mismatch")
    dependency_values = {str(g1a.get("dependency_lock_sha256", "")), str(g2a.get("dependency_lock_sha256", ""))}
    if len(dependency_values) != 1 or any(len(value) != 64 for value in dependency_values): errors.append("G1A/G2A dependency-lock identity mismatch")
    durability = g2a.get("durability_contract") or {}
    durability_errors = validate_g2a_durability_contract(durability, expected_input_summary_sha256=g2a.get("input_summary_sha256") or {})
    if durability_errors: errors.extend(f"G2A durability: {error}" for error in durability_errors)
    if errors: raise ValueError("pre-science qualification failed: " + "; ".join(errors))

    gates = {name: "PASS" for name in REQUIRED_PRE_SCIENCE_GATES}; bindings: dict[str, Any] = {}
    for name in static_authorization_gates:
        bindings[name] = {"kind": "exact_head_code_attestation", "attestation_sha256": code_attestation["attestation_sha256"], "artifact_file_sha256": file_hashes["code_attestation"], "source_git_commit": source_git_commit, "workflow_run_id": str(code_attestation["workflow_run_id"])}
    bindings["immutable_v12_lock"] = {"kind": "exact_head_lock_runtime_attestation", "attestation_sha256": lock_runtime_attestation["attestation_sha256"], "content_lock_report_sha256": lock_runtime_attestation["content_lock_report_sha256"], "artifact_file_sha256": file_hashes["lock_runtime_attestation"], "source_git_commit": source_git_commit, "workflow_run_id": str(lock_runtime_attestation["workflow_run_id"])}
    bindings[AUTH_RUNTIME_QUALIFICATION_GATE] = {"kind": "exact_head_lock_runtime_attestation", "attestation_sha256": lock_runtime_attestation["attestation_sha256"], "runtime_report_sha256": lock_runtime_attestation["runtime_report_sha256"], "artifact_file_sha256": file_hashes["lock_runtime_attestation"], "source_git_commit": source_git_commit, "workflow_run_id": str(lock_runtime_attestation["workflow_run_id"])}
    bindings["candidate_claim_boundary_lock"] = {"kind": "exact_head_lock_runtime_attestation", "attestation_sha256": lock_runtime_attestation["attestation_sha256"], "candidate_claim_boundary_sha256": lock_runtime_attestation["candidate_claim_boundary_sha256"], "xai_operationalization_sha256": lock_runtime_attestation["xai_operationalization_sha256"], "source_git_commit": source_git_commit, "workflow_run_id": str(lock_runtime_attestation["workflow_run_id"])}
    bindings["g1a"] = {"kind": "track_a_v12_g1a", "g1a_seal_sha256": g1a["g1a_seal_sha256"], "artifact_file_sha256": file_hashes["g1a"], "source_git_commit": source_git_commit}
    bindings["g2a"] = {"kind": "track_a_v12_g2a_full_run_forecast", "barrier_sha256": g2a["barrier_sha256"], "durability_contract_sha256": durability["durability_contract_sha256"], "artifact_file_sha256": file_hashes["g2a"], "source_git_commit": source_git_commit}
    bindings["scheduler_freeze"] = {"kind": "track_a_v12_pre_science_scheduler_full_run", "scheduler_freeze_sha256": scheduler["scheduler_freeze_sha256"], "artifact_file_sha256": file_hashes["scheduler"], "source_git_commit": source_git_commit}
    if set(gates) != set(REQUIRED_PRE_SCIENCE_GATES): raise AssertionError("v12 GO gate inventory construction drift")
    if set(bindings) != set(REQUIRED_PRE_SCIENCE_GATES): raise AssertionError("v12 GO binding inventory construction drift")

    compatibility = {
        "schema_version": COMPAT_SCHEMA,
        "stage": "TRACKA_V12_CONTROL_SCHEMA_COMPATIBILITY",
        "status": "PASS",
        "science_source_sha": source_git_commit,
        "code_attestation_extra_static_gates": {name: static[name] for name in sorted(CODE_EXTRA_STATIC_GATES)},
        "code_attestation_extra_static_gate": "teacher_factory_root_binding_contract",
        "code_attestation_extra_static_gate_status": static["teacher_factory_root_binding_contract"],
        "authorization_runtime_gate": AUTH_RUNTIME_QUALIFICATION_GATE,
        "lock_runtime_attestation_field": LOCK_RUNTIME_QUALIFICATION_FIELD,
        "lock_runtime_attestation_field_status": lock_runtime_attestation[LOCK_RUNTIME_QUALIFICATION_FIELD],
        "note": "Runtime-only compatibility bridge for the exact post-authorization attestation hardening gates and the exact R13-prefixed runtime-qualification field. No scientific model, seed, data, objective, selector, G1A, G2A, or protected-surface contract is changed.",
    }
    return gates, bindings, compatibility


def seal_science_go_v12(repo: Path, *, code_attestation_path: Path, lock_attestation_path: Path, g1a_path: Path, g2a_path: Path, scheduler_path: Path, output_path: Path, compatibility_path: Path) -> dict[str, Any]:
    _ensure_science_src(repo)
    from cropcop_je.atomic_io import atomic_write_json
    from cropcop_je.hashing import sha256_file
    from cropcop_je.tracka_v12_authorization import build_science_authorization, validate_science_authorization
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if source != SCIENCE_SHA: raise OperatorError(f"v12 control bridge requires exact frozen science SHA {SCIENCE_SHA}; observed {source}")
    paths = {"code_attestation": code_attestation_path, "lock_runtime_attestation": lock_attestation_path, "g1a": g1a_path, "g2a": g2a_path, "scheduler": scheduler_path}
    missing = [name for name, path in paths.items() if not Path(path).is_file()]
    if missing: raise OperatorError("required GO artifact missing: " + ", ".join(missing))
    payloads = {name: json.loads(Path(path).read_text(encoding="utf-8")) for name, path in paths.items()}; file_hashes = {name: sha256_file(path) for name, path in paths.items()}
    try:
        gates, bindings, compatibility = compose_gate_inputs_v12(repo, source_git_commit=source, code_attestation=payloads["code_attestation"], lock_runtime_attestation=payloads["lock_runtime_attestation"], g1a=payloads["g1a"], g2a=payloads["g2a"], scheduler=payloads["scheduler"], file_hashes=file_hashes)
    except ValueError as exc: raise OperatorError(str(exc)) from exc
    authorization = build_science_authorization(source_git_commit=source, g1a_seal_sha256=payloads["g1a"]["g1a_seal_sha256"], g2a_barrier_sha256=payloads["g2a"]["barrier_sha256"], scheduler_freeze_sha256=payloads["scheduler"]["scheduler_freeze_sha256"], pre_science_gates=gates, evidence_bindings=bindings)
    validation_errors = validate_science_authorization(authorization, expected_source_sha=source, expected_g1a_seal_sha256=payloads["g1a"]["g1a_seal_sha256"], expected_g2a_barrier_sha256=payloads["g2a"]["barrier_sha256"], expected_scheduler_freeze_sha256=payloads["scheduler"]["scheduler_freeze_sha256"])
    if validation_errors: raise OperatorError("constructed Track-A science GO failed self-validation: " + "; ".join(validation_errors))
    atomic_write_json(output_path, authorization); atomic_write_json(compatibility_path, compatibility)
    print("CONTROL_SCHEMA_COMPATIBILITY_V12_PASS " f"extra_gates={','.join(sorted(CODE_EXTRA_STATIC_GATES))} " f"runtime_field={LOCK_RUNTIME_QUALIFICATION_FIELD}", flush=True)
    return compatibility


CONTROL_FILES = legacy.CONTROL_FILES


def validate_control_bundle(repo: Path, g1a_seal: dict, control_dir: Path) -> dict:
    return legacy.validate_control_bundle(repo, g1a_seal, control_dir)


def build_control_k1(repo: Path, *, g1a_bundle: Path, g1a_seal: dict, summaries: dict[str, Path], master_root: Path, stack: dict) -> tuple[Path, dict]:
    control_dir = master_root / "control"
    fetched = legacy.fetch_public_bundle(repo, legacy.control_public_run_id(), CONTROL_FILES, control_dir)
    if fetched is not None:
        control = legacy.validate_control_bundle(repo, g1a_seal, control_dir); print("Reusing already-published canonical Track-A control plane."); return control_dir, control
    shutil.rmtree(control_dir, ignore_errors=True); control_dir.mkdir(parents=True, exist_ok=True)
    barrier = control_dir / "TRACKA_V12_G2A_BARRIER.json"; scheduler = control_dir / "TRACKA_V12_SCHEDULER_FREEZE.json"
    command = [sys.executable, str(repo / "journal_extension/scripts/seal_tracka_v12_g2a.py")]
    for calibration_id in sorted(legacy.REQUIRED_G2A): command += ["--summary", str(summaries[calibration_id])]
    command += ["--barrier-out", str(barrier), "--scheduler-out", str(scheduler)]
    if subprocess.run(command, cwd=repo, text=True).returncode != 0: raise OperatorError("G2A barrier/scheduler sealer failed")
    attestation_dir = master_root / "release-attestations"; code_attestation, lock_attestation = legacy.materialize_verified_attestations(attestation_dir)
    science_go = control_dir / "TRACKA_V12_SCIENCE_GO.json"; compatibility_path = control_dir / "TRACKA_V12_CONTROL_SCHEMA_COMPAT_V12.json"
    compatibility = seal_science_go_v12(repo, code_attestation_path=code_attestation, lock_attestation_path=lock_attestation, g1a_path=g1a_bundle / "TRACKA_V12_G1A_SEAL.json", g2a_path=barrier, scheduler_path=scheduler, output_path=science_go, compatibility_path=compatibility_path)
    owners = {"K1": legacy.owner_from_summary(summaries["CAL-EFFB0"]), "K2": legacy.owner_from_summary(summaries["CAL-MNV4-LOGITS"]), "K3": legacy.owner_from_summary(summaries["CAL-R13"])}
    if legacy.owner_from_summary(summaries["CAL-CNXTT"]) != owners["K1"]: raise OperatorError("K1 G2A summaries disagree on account owner")
    if legacy.owner_from_summary(summaries["CAL-MNV4-FEATURE"]) != owners["K2"]: raise OperatorError("K2 G2A summaries disagree on account owner")
    scheduler_payload = legacy.load_json(scheduler); durable_map = legacy.scientific_durable_map(scheduler_payload, owners)
    legacy.write_json(control_dir / "TRACKA_V12_DURABLE_MAP.json", durable_map); legacy.write_json(control_dir / "TRACKA_V12_ACCOUNT_OWNERS.json", owners)
    control = legacy.validate_control_bundle(repo, g1a_seal, control_dir)
    report = {"schema_version": "1.2.2", "stage": "TRACKA_V12_MASTER_CONTROL", "status": "PASS", "science_source_sha": SCIENCE_SHA, "operator_runtime_sha": legacy.operator_runtime_head(), "dependency_lock_sha256": stack["dependency_lock_sha256"], "g1a_seal_sha256": g1a_seal["g1a_seal_sha256"], "g2a_barrier_sha256": control["barrier"]["barrier_sha256"], "scheduler_freeze_sha256": control["scheduler"]["scheduler_freeze_sha256"], "science_authorization_sha256": control["go"]["authorization_sha256"], "authorized_experiment_count": 11, "account_owners": owners, "control_schema_compatibility": compatibility, "protected_test_accessed": False, "external_surface_accessed": False}
    legacy.write_json(control_dir / "TRACKA_V12_CONTROL_PUBLIC_REPORT.json", report)
    publish_paths = [control_dir / name for name in CONTROL_FILES]; publish_paths.append(compatibility_path); legacy.publish_public_files(repo, legacy.control_public_run_id(), publish_paths)
    if legacy.fetch_public_bundle(repo, legacy.control_public_run_id(), CONTROL_FILES, control_dir) is None: raise OperatorError("control-plane publication did not round-trip through GitHub")
    return control_dir, legacy.validate_control_bundle(repo, g1a_seal, control_dir)
