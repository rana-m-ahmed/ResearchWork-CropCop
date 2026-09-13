from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import seal_tracka_v12_science_go as base
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.hashing import sha256_file
from cropcop_je.tracka_v12_authorization import build_science_authorization, validate_science_authorization
from cropcop_je.tracka_v12_g2a_durability import validate_g2a_durability_contract


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
    source = base.git_head(repo)
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

    payloads = {name: base.load_json(path) for name, path in paths.items()}
    file_hashes = {name: sha256_file(path) for name, path in paths.items()}

    gates, bindings = base.compose_gate_inputs(
        source_git_commit=source,
        code_attestation=payloads["code_attestation"],
        lock_runtime_attestation=payloads["lock_runtime_attestation"],
        g1a=payloads["g1a"],
        g2a=payloads["g2a"],
        scheduler=payloads["scheduler"],
        file_hashes=file_hashes,
    )

    durability = payloads["g2a"].get("durability_contract") or {}
    durability_errors = validate_g2a_durability_contract(
        durability,
        expected_input_summary_sha256=payloads["g2a"].get("input_summary_sha256") or {},
    )
    if durability_errors:
        raise SystemExit("G2A durability qualification failed: " + "; ".join(durability_errors))
    bindings["g2a"]["durability_contract_sha256"] = durability["durability_contract_sha256"]

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
