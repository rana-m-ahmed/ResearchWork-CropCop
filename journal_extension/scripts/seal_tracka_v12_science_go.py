from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.hashing import sha256_file, sha256_json
from cropcop_je.tracka_v12_authorization import (
    REQUIRED_PRE_SCIENCE_GATES,
    build_science_authorization,
    validate_science_authorization,
)
from cropcop_je.tracka_v12_g1a import validate_g1a_seal_object
from cropcop_je.tracka_v12_g2a_v122 import (
    validate_g2a_v122_barrier,
    validate_scheduler_freeze_v122,
)

DYNAMIC_GATES = {
    "immutable_v12_lock",
    "v121_runtime_qualification",
    "candidate_claim_boundary_lock",
    "g1a",
    "g2a",
    "scheduler_freeze",
}
STATIC_GATES = set(REQUIRED_PRE_SCIENCE_GATES) - DYNAMIC_GATES


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def verify_embedded_hash(payload: dict[str, Any], field: str) -> bool:
    clean = dict(payload)
    observed = clean.pop(field, None)
    return isinstance(observed, str) and len(observed) == 64 and observed == sha256_json(clean)


def compose_gate_inputs(
    *,
    source_git_commit: str,
    code_attestation: dict[str, Any],
    lock_runtime_attestation: dict[str, Any],
    g1a: dict[str, Any],
    g2a: dict[str, Any],
    scheduler: dict[str, Any],
    file_hashes: dict[str, str],
) -> tuple[dict[str, str], dict[str, Any]]:
    errors: list[str] = []
    if code_attestation.get("status") != "PASS" or code_attestation.get("source_git_commit") != source_git_commit:
        errors.append("code attestation is not exact-source PASS")
    if not verify_embedded_hash(code_attestation, "attestation_sha256"):
        errors.append("code attestation self-hash mismatch")
    static = code_attestation.get("static_pre_science_gates", {})
    if set(static) != STATIC_GATES or any(static.get(name) != "PASS" for name in STATIC_GATES):
        errors.append("code attestation static gate inventory/PASS state mismatch")

    if lock_runtime_attestation.get("status") != "PASS" or lock_runtime_attestation.get("source_git_commit") != source_git_commit:
        errors.append("lock/runtime attestation is not exact-source PASS")
    if not verify_embedded_hash(lock_runtime_attestation, "attestation_sha256"):
        errors.append("lock/runtime attestation self-hash mismatch")
    if lock_runtime_attestation.get("science_authorized") is not False:
        errors.append("lock/runtime attestation unexpectedly authorizes science")
    for gate in ("immutable_v12_lock", "v121_runtime_qualification", "candidate_claim_boundary_lock"):
        if lock_runtime_attestation.get(gate) != "PASS":
            errors.append(f"lock/runtime attestation gate is not PASS: {gate}")

    g1_errors = validate_g1a_seal_object(g1a)
    if g1_errors:
        errors.extend(f"G1A: {error}" for error in g1_errors)
    if g1a.get("status") != "PASS" or g1a.get("science_authorized") is not False:
        errors.append("G1A is not pre-science PASS")
    if g1a.get("source_git_sha") != source_git_commit:
        errors.append("G1A source SHA mismatch")

    g2_errors = validate_g2a_v122_barrier(
        g2a,
        expected_source_sha=source_git_commit,
        expected_g1a_seal_sha256=g1a.get("g1a_seal_sha256"),
    )
    if g2_errors:
        errors.extend(f"G2A: {error}" for error in g2_errors)
    scheduler_errors = validate_scheduler_freeze_v122(
        scheduler,
        expected_g2a_barrier_sha256=g2a.get("barrier_sha256"),
    )
    if scheduler_errors:
        errors.extend(f"scheduler: {error}" for error in scheduler_errors)
    if scheduler.get("source_git_commit") != source_git_commit:
        errors.append("scheduler source SHA mismatch")
    if scheduler.get("g1a_seal_sha256") != g1a.get("g1a_seal_sha256"):
        errors.append("scheduler/G1A binding mismatch")

    dependency_values = {
        str(g1a.get("dependency_lock_sha256", "")),
        str(g2a.get("dependency_lock_sha256", "")),
    }
    if len(dependency_values) != 1 or any(len(value) != 64 for value in dependency_values):
        errors.append("G1A/G2A dependency-lock identity mismatch")

    if errors:
        raise ValueError("pre-science qualification failed: " + "; ".join(errors))

    gates = {name: "PASS" for name in REQUIRED_PRE_SCIENCE_GATES}
    bindings: dict[str, Any] = {}
    for name in STATIC_GATES:
        bindings[name] = {
            "kind": "exact_head_code_attestation",
            "attestation_sha256": code_attestation["attestation_sha256"],
            "artifact_file_sha256": file_hashes["code_attestation"],
            "source_git_commit": source_git_commit,
        }
    bindings["immutable_v12_lock"] = {
        "kind": "exact_head_lock_runtime_attestation",
        "attestation_sha256": lock_runtime_attestation["attestation_sha256"],
        "content_lock_report_sha256": lock_runtime_attestation["content_lock_report_sha256"],
        "artifact_file_sha256": file_hashes["lock_runtime_attestation"],
        "source_git_commit": source_git_commit,
    }
    bindings["v121_runtime_qualification"] = {
        "kind": "exact_head_lock_runtime_attestation",
        "attestation_sha256": lock_runtime_attestation["attestation_sha256"],
        "runtime_report_sha256": lock_runtime_attestation["runtime_report_sha256"],
        "artifact_file_sha256": file_hashes["lock_runtime_attestation"],
        "source_git_commit": source_git_commit,
    }
    bindings["candidate_claim_boundary_lock"] = {
        "kind": "exact_head_lock_runtime_attestation",
        "attestation_sha256": lock_runtime_attestation["attestation_sha256"],
        "candidate_claim_boundary_sha256": lock_runtime_attestation["candidate_claim_boundary_sha256"],
        "xai_operationalization_sha256": lock_runtime_attestation["xai_operationalization_sha256"],
        "source_git_commit": source_git_commit,
    }
    bindings["g1a"] = {
        "kind": "track_a_v12_g1a",
        "g1a_seal_sha256": g1a["g1a_seal_sha256"],
        "artifact_file_sha256": file_hashes["g1a"],
        "source_git_commit": source_git_commit,
    }
    bindings["g2a"] = {
        "kind": "track_a_v12_g2a_full_run_forecast",
        "barrier_sha256": g2a["barrier_sha256"],
        "artifact_file_sha256": file_hashes["g2a"],
        "source_git_commit": source_git_commit,
    }
    bindings["scheduler_freeze"] = {
        "kind": "track_a_v12_pre_science_scheduler_full_run",
        "scheduler_freeze_sha256": scheduler["scheduler_freeze_sha256"],
        "artifact_file_sha256": file_hashes["scheduler"],
        "source_git_commit": source_git_commit,
    }
    if set(gates) != set(REQUIRED_PRE_SCIENCE_GATES) or set(bindings) != set(REQUIRED_PRE_SCIENCE_GATES):
        raise AssertionError("final GO gate/binding inventory construction drift")
    return gates, bindings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--code-attestation", required=True)
    ap.add_argument("--lock-runtime-attestation", required=True)
    ap.add_argument("--g1a-seal", required=True)
    ap.add_argument("--g2a-barrier", required=True)
    ap.add_argument("--scheduler-freeze", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    source = git_head(repo)
    if len(source) != 40:
        raise SystemExit("current checkout is not bound to a full Git SHA")
    paths = {
        "code_attestation": Path(args.code_attestation).resolve(),
        "lock_runtime_attestation": Path(args.lock_runtime_attestation).resolve(),
        "g1a": Path(args.g1a_seal).resolve(),
        "g2a": Path(args.g2a_barrier).resolve(),
        "scheduler": Path(args.scheduler_freeze).resolve(),
    }
    if any(not path.is_file() for path in paths.values()):
        missing = [name for name, path in paths.items() if not path.is_file()]
        raise SystemExit("required GO artifact missing: " + ", ".join(missing))
    payloads = {name: load_json(path) for name, path in paths.items()}
    file_hashes = {name: sha256_file(path) for name, path in paths.items()}

    gates, bindings = compose_gate_inputs(
        source_git_commit=source,
        code_attestation=payloads["code_attestation"],
        lock_runtime_attestation=payloads["lock_runtime_attestation"],
        g1a=payloads["g1a"],
        g2a=payloads["g2a"],
        scheduler=payloads["scheduler"],
        file_hashes=file_hashes,
    )
    authorization = build_science_authorization(
        source_git_commit=source,
        g1a_seal_sha256=payloads["g1a"]["g1a_seal_sha256"],
        g2a_barrier_sha256=payloads["g2a"]["barrier_sha256"],
        scheduler_freeze_sha256=payloads["scheduler"]["scheduler_freeze_sha256"],
        pre_science_gates=gates,
        evidence_bindings=bindings,
    )
    errors = validate_science_authorization(
        authorization,
        expected_source_sha=source,
        expected_g1a_seal_sha256=payloads["g1a"]["g1a_seal_sha256"],
        expected_g2a_barrier_sha256=payloads["g2a"]["barrier_sha256"],
        expected_scheduler_freeze_sha256=payloads["scheduler"]["scheduler_freeze_sha256"],
    )
    if errors:
        raise SystemExit("constructed Track-A science GO failed self-validation: " + "; ".join(errors))
    atomic_write_json(args.output, authorization)
    print(json.dumps(authorization, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
