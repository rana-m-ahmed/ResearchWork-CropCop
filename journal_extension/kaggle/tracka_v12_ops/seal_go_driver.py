from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from tracka_v12_kaggle_operator_v2 import (
    SCIENCE_SHA,
    assert_clean_science_checkout,
    assert_kaggle_paths,
    assert_python_version,
    discover_g1a_bundle,
    discover_json_by_key,
    ensure_science_checkout,
    install_locked_stack,
    load_json,
    scientific_durable_map,
    sha256_file,
    verify_locked_stack,
    write_json,
)

REQUIRED_PROFILES = {
    "CAL-EFFB0",
    "CAL-CNXTT",
    "CAL-MNV4-LOGITS",
    "CAL-MNV4-FEATURE",
    "CAL-R13",
}


def unique_named(input_root: Path, name: str, override_env: str) -> Path:
    override = os.environ.get(override_env, "").strip()
    if override:
        path = Path(override).resolve()
        if not path.is_file():
            raise RuntimeError(f"{override_env} does not point to a file: {path}")
        return path
    matches = sorted(path.resolve() for path in input_root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"{name} resolution must be unique, found {len(matches)}: {matches}")
    return matches[0]


def owner_from_summary(summary: dict, calibration_id: str) -> str:
    locator = str(summary.get("durability", {}).get("locator", ""))
    owner, sep, dataset = locator.partition("/")
    if not sep or not owner or not dataset:
        raise RuntimeError(f"{calibration_id} has invalid durability locator: {locator}")
    return owner


def main() -> int:
    assert_kaggle_paths()
    assert_python_version()

    output_root = Path("/kaggle/working/TRACKA_V12_EXECUTION_CONTROL_BUNDLE")
    if output_root.exists():
        raise RuntimeError(
            f"{output_root} already exists. G2A/scheduler/GO artifacts are immutable; use a fresh session."
        )
    output_root.mkdir(parents=True)

    repo = ensure_science_checkout()
    install_locked_stack(repo)
    stack = verify_locked_stack(repo)
    assert_clean_science_checkout(repo)

    input_root = Path("/kaggle/input")
    g1a_bundle = discover_g1a_bundle(
        override=os.environ.get("CROPCOP_G1A_BUNDLE", ""),
    )
    g1a_seal = g1a_bundle / "TRACKA_V12_G1A_SEAL.json"
    summaries = discover_json_by_key(
        input_root,
        key="calibration_id",
        values=REQUIRED_PROFILES,
    )
    code_attestation = unique_named(
        input_root,
        "tracka-v12-pre-science-code-attestation.json",
        "CROPCOP_CODE_ATTESTATION",
    )
    lock_attestation = unique_named(
        input_root,
        "tracka-v12-exact-head-lock-runtime-attestation.json",
        "CROPCOP_LOCK_RUNTIME_ATTESTATION",
    )

    barrier = output_root / "TRACKA_V12_G2A_BARRIER.json"
    scheduler = output_root / "TRACKA_V12_SCHEDULER_FREEZE.json"
    command = [
        sys.executable,
        str(repo / "journal_extension/scripts/seal_tracka_v12_g2a.py"),
    ]
    for calibration_id in sorted(REQUIRED_PROFILES):
        command += ["--summary", str(summaries[calibration_id])]
    command += ["--barrier-out", str(barrier), "--scheduler-out", str(scheduler)]
    cp = subprocess.run(command, cwd=repo, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"G2A barrier sealer failed with exit code {cp.returncode}")

    science_go = output_root / "TRACKA_V12_SCIENCE_GO.json"
    command = [
        sys.executable,
        str(repo / "journal_extension/scripts/seal_tracka_v12_science_go_v123.py"),
        "--repo-root", str(repo),
        "--code-attestation", str(code_attestation),
        "--lock-runtime-attestation", str(lock_attestation),
        "--g1a-seal", str(g1a_seal),
        "--g2a-barrier", str(barrier),
        "--scheduler-freeze", str(scheduler),
        "--output", str(science_go),
    ]
    cp = subprocess.run(command, cwd=repo, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"final science-GO sealer failed with exit code {cp.returncode}")

    summary_payloads = {cid: load_json(path) for cid, path in summaries.items()}
    account_owners = {
        "K1": owner_from_summary(summary_payloads["CAL-EFFB0"], "CAL-EFFB0"),
        "K2": owner_from_summary(summary_payloads["CAL-MNV4-LOGITS"], "CAL-MNV4-LOGITS"),
        "K3": owner_from_summary(summary_payloads["CAL-R13"], "CAL-R13"),
    }
    if owner_from_summary(summary_payloads["CAL-CNXTT"], "CAL-CNXTT") != account_owners["K1"]:
        raise RuntimeError("K1 G2A profile durability owners disagree")
    if owner_from_summary(summary_payloads["CAL-MNV4-FEATURE"], "CAL-MNV4-FEATURE") != account_owners["K2"]:
        raise RuntimeError("K2 G2A profile durability owners disagree")

    scheduler_payload = load_json(scheduler)
    durable_map = scientific_durable_map(scheduler_payload, account_owners)
    durable_map_path = output_root / "TRACKA_V12_DURABLE_MAP.json"
    write_json(durable_map_path, durable_map)

    sys.path.insert(0, str(repo / "journal_extension/src"))
    from cropcop_je.tracka_v12_orchestration import validate_durable_map

    errors = validate_durable_map(durable_map)
    if errors:
        raise RuntimeError("scientific durable map invalid: " + "; ".join(errors))

    go = load_json(science_go)
    if go.get("status") != "PASS" or go.get("science_authorized") is not True:
        raise RuntimeError("final science authorization is not PASS/authorized")
    if go.get("source_git_commit") != SCIENCE_SHA:
        raise RuntimeError("final science GO source mismatch")
    if len(go.get("authorized_experiment_ids", [])) != 11:
        raise RuntimeError("final science GO must authorize exactly 11 continuation states")

    write_json(output_root / "TRACKA_V12_ACCOUNT_OWNERS.json", account_owners)
    shutil.copy2(g1a_seal, output_root / "TRACKA_V12_G1A_SEAL.json")
    shutil.copy2(code_attestation, output_root / code_attestation.name)
    shutil.copy2(lock_attestation, output_root / lock_attestation.name)
    for calibration_id, path in summaries.items():
        shutil.copy2(path, output_root / f"{calibration_id}_SUMMARY.json")

    report = {
        "schema_version": "1.0",
        "stage": "TRACKA_V12_FINAL_PRE_SCIENCE_CONTROL_PLANE",
        "status": "PASS",
        "science_source_sha": SCIENCE_SHA,
        "dependency_lock_sha256": stack["dependency_lock_sha256"],
        "g1a_seal_sha256": load_json(g1a_seal)["g1a_seal_sha256"],
        "g2a_barrier_sha256": load_json(barrier)["barrier_sha256"],
        "scheduler_freeze_sha256": scheduler_payload["scheduler_freeze_sha256"],
        "science_authorization_sha256": go["authorization_sha256"],
        "code_attestation_file_sha256": sha256_file(code_attestation),
        "lock_runtime_attestation_file_sha256": sha256_file(lock_attestation),
        "account_owners": account_owners,
        "durable_map_file_sha256": sha256_file(durable_map_path),
        "authorized_experiment_count": 11,
    }
    write_json(output_root / "TRACKA_V12_PRE_SCIENCE_OPERATOR_REPORT.json", report)
    archive = shutil.make_archive(
        "/kaggle/working/TRACKA_V12_EXECUTION_CONTROL_BUNDLE",
        "zip",
        root_dir=output_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Execution-control bundle: {output_root}")
    print(f"Transfer archive: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
