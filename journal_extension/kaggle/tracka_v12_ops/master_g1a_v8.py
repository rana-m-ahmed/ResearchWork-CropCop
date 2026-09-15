from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from tracka_v12_kaggle_operator_v8 import (
    SCIENCE_SHA,
    OperatorError,
    adopt_g1a_if_present,
    download_and_validate_g1a_dataset,
    ensure_private_dataset,
    fetch_public_file,
    g1a_handoff_run_id,
    load_json,
    operator_runtime_head,
    prepare_r13,
    publish_public_files,
    resolve_principal_g1_bundle,
    shared_g1a_locator,
    validate_g1a_bundle_with_science,
    version_private_dataset,
    wait_kaggle_dataset_ready,
    write_json,
)
from master_wait_v8 import POLL_SECONDS, dependency_wait_expired
from master_verified_pretrained_v5 import prepare_verified_torchvision

G1A_HANDOFF_FILE = "TRACKA_V12_G1A_HANDOFF.json"
HANDOFF_SCHEMA_VERSION = "1.2.1"
G1A_FAILURE_CODE = "K1_G1A_PIPELINE_FAILED"


def _retry_pre_science_io(label: str, fn, *, attempts: int = 3):
    if attempts < 1:
        raise OperatorError("retry attempts must be positive")
    last_error: Exception | None = None
    delays = (10, 30, 60)
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except (OperatorError, subprocess.SubprocessError, OSError, RuntimeError) as exc:
            last_error = exc
            if attempt == attempts:
                break
            delay = delays[min(attempt - 1, len(delays) - 1)]
            print(f"{label} attempt {attempt}/{attempts} failed: {type(exc).__name__}; retrying in {delay}s")
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def publish_json(repo: Path, run_id: str, path: Path) -> str:
    return publish_public_files(repo, run_id, [path])


def write_handoff(path: Path, *, status: str, locator: str, seal: dict | None = None, failure_code: str | None = None) -> dict:
    if status not in {"PREPARING", "READY", "FAILED"}:
        raise OperatorError(f"invalid G1A handoff status: {status}")
    if status == "READY" and (seal is None or len(str(seal.get("g1a_seal_sha256", ""))) != 64):
        raise OperatorError("READY G1A handoff requires one validated seal hash")
    if status != "READY" and seal is not None:
        raise OperatorError(f"{status} G1A handoff must not carry a seal")
    if status == "FAILED" and failure_code != G1A_FAILURE_CODE:
        raise OperatorError("FAILED G1A handoff requires the frozen non-sensitive failure code")
    if status != "FAILED" and failure_code is not None:
        raise OperatorError(f"{status} G1A handoff must not carry a failure code")
    payload = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "stage": "TRACKA_V12_G1A_SHARED_HANDOFF",
        "status": status,
        "science_source_sha": SCIENCE_SHA,
        "operator_runtime_sha": operator_runtime_head(),
        "private_kaggle_dataset_locator": locator,
        "g1a_seal_sha256": None if seal is None else seal.get("g1a_seal_sha256"),
        "failure_code": failure_code,
        "science_authorized": False,
        "protected_test_accessed": False,
        "external_surface_accessed": False,
    }
    write_json(path, payload)
    return payload


def g1a_public_report(path: Path, seal: dict, locator: str, stack: dict) -> dict:
    payload = {
        "schema_version": "1.2.1",
        "stage": "TRACKA_V12_G1A_MASTER",
        "status": "PASS",
        "science_source_sha": SCIENCE_SHA,
        "operator_runtime_sha": operator_runtime_head(),
        "g1a_seal_sha256": seal["g1a_seal_sha256"],
        "r13_parity_contract_id": (seal.get("r13") or {}).get("parity_contract_id"),
        "r13_parity_required_max_abs_difference": (seal.get("r13") or {}).get("required_max_abs_difference"),
        "dependency_lock_sha256": stack["dependency_lock_sha256"],
        "private_handoff_locator": locator,
        "science_authorized": False,
        "protected_test_accessed": False,
        "external_surface_accessed": False,
    }
    write_json(path, payload)
    return payload


def build_g1a_once(repo: Path, *, manifest: Path, class_map: Path, image_root: Path, output_bundle: Path) -> dict:
    principal = resolve_principal_g1_bundle(override=os.environ.get("CROPCOP_PRINCIPAL_G1", ""))
    upstream_root = output_bundle.parent / "upstream"
    if upstream_root.exists():
        shutil.rmtree(upstream_root)
    upstream_root.mkdir(parents=True)
    baselines = _retry_pre_science_io(
        "Official TorchVision download + exact tensor provenance",
        lambda: prepare_verified_torchvision(repo, upstream_root / "torchvision"),
    )
    r13 = _retry_pre_science_io(
        "Pinned R13 Hugging Face artifact",
        lambda: prepare_r13(upstream_root / "r13"),
    )
    command = [
        sys.executable, str(repo / "journal_extension/scripts/seal_tracka_v12_g1a_v121.py"),
        "--repo-root", str(repo), "--authorized-source-sha", SCIENCE_SHA,
        "--manifest", str(manifest), "--class-map", str(class_map), "--image-root", str(image_root),
        "--principal-g1-bundle", str(principal),
        "--effb0-pretrained", str(baselines["effb0"]["artifact"]),
        "--effb0-provenance", str(baselines["effb0"]["provenance"]),
        "--cnxtt-pretrained", str(baselines["cnxtt"]["artifact"]),
        "--cnxtt-provenance", str(baselines["cnxtt"]["provenance"]),
        "--r13-pretrained", str(r13), "--bundle-dir", str(output_bundle),
    ]
    cp = subprocess.run(command, cwd=repo, text=True)
    if cp.returncode != 0:
        raise OperatorError(f"G1A v1.2.1 sealer failed with rc={cp.returncode}")
    return validate_g1a_bundle_with_science(repo, output_bundle)


def _publish_failed_handoff_best_effort(repo: Path, handoff: Path, locator: str) -> None:
    write_handoff(handoff, status="FAILED", locator=locator, failure_code=G1A_FAILURE_CODE)
    try:
        publish_json(repo, g1a_handoff_run_id(), handoff)
    except Exception as publish_exc:
        print(f"G1A_FAIL_HANDOFF_PUBLICATION_FAILED {type(publish_exc).__name__}; preserving original K1 failure.", flush=True)


def ensure_canonical_g1a_k1(
    repo: Path, *, manifest: Path, class_map: Path, image_root: Path, username: str,
    kaggle_env: dict[str, str], stack: dict, master_root: Path,
) -> tuple[Path, dict, str]:
    locator = os.environ.get("CROPCOP_G1A_SHARED_DATASET", "").strip() or shared_g1a_locator(username)
    if locator.split("/", 1)[0].casefold() != username.casefold():
        raise OperatorError("K1 canonical G1A shared dataset must be owned by authenticated K1")
    ensure_private_dataset(locator, title="CropCop Track-A v1.2.1 canonical G1A", env=kaggle_env)
    handoff = master_root / G1A_HANDOFF_FILE
    print(f"Canonical private G1A dataset: {locator}")
    print("Before K2/K3 launch, grant both worker accounts Can view access to this private Kaggle dataset.")
    adopted = adopt_g1a_if_present(repo, locator, master_root / "g1a-shared-download", env=kaggle_env)
    if adopted is not None:
        bundle, seal = adopted
        print("Reusing already-valid canonical v1.2.1 G1A bundle.")
    else:
        write_handoff(handoff, status="PREPARING", locator=locator)
        publish_json(repo, g1a_handoff_run_id(), handoff)
        bundle = master_root / "TRACKA_V12_G1A_BUNDLE"
        shutil.rmtree(bundle, ignore_errors=True)
        try:
            seal = build_g1a_once(repo, manifest=manifest, class_map=class_map, image_root=image_root, output_bundle=bundle)
            version_private_dataset(locator, bundle, message=f"Canonical Track-A v1.2.1 G1A {SCIENCE_SHA[:12]}", env=kaggle_env)
            wait_kaggle_dataset_ready(locator, env=kaggle_env)
            roundtrip_bundle, roundtrip_seal = download_and_validate_g1a_dataset(repo, locator, master_root / "g1a-roundtrip", env=kaggle_env)
            if roundtrip_seal["g1a_seal_sha256"] != seal["g1a_seal_sha256"]:
                raise OperatorError("private G1A round-trip changed canonical seal")
            bundle, seal = roundtrip_bundle, roundtrip_seal
        except BaseException:
            shutil.rmtree(bundle, ignore_errors=True)
            _publish_failed_handoff_best_effort(repo, handoff, locator)
            raise
    report = master_root / "TRACKA_V12_G1A_PUBLIC_REPORT.json"
    g1a_public_report(report, seal, locator, stack)
    public_seal = master_root / "TRACKA_V12_G1A_SEAL.json"
    shutil.copy2(bundle / "TRACKA_V12_G1A_SEAL.json", public_seal)
    publish_public_files(repo, f"TRACKA-V12-G1A-{SCIENCE_SHA[:12]}", [public_seal, report])
    write_handoff(handoff, status="READY", locator=locator, seal=seal)
    publish_json(repo, g1a_handoff_run_id(), handoff)
    return bundle, seal, locator


def _wait_for_current_g1a_handoff(repo: Path, destination: Path) -> dict:
    runtime_sha = operator_runtime_head()
    while True:
        path = fetch_public_file(repo, g1a_handoff_run_id(), G1A_HANDOFF_FILE, destination)
        if path is not None:
            payload = load_json(path)
            if payload.get("science_source_sha") == SCIENCE_SHA and payload.get("operator_runtime_sha") == runtime_sha:
                status = payload.get("status")
                if status == "FAILED":
                    if payload.get("failure_code") != G1A_FAILURE_CODE:
                        raise OperatorError("current-runtime G1A FAILED handoff has invalid failure code")
                    raise OperatorError("K1 reported canonical G1A failure for this exact science/runtime identity")
                if status == "READY" and len(str(payload.get("g1a_seal_sha256", ""))) == 64:
                    return payload
        if dependency_wait_expired():
            raise TimeoutError("session dependency budget exhausted waiting for current-runtime canonical G1A handoff")
        time.sleep(POLL_SECONDS)


def acquire_canonical_g1a_worker(repo: Path, *, kaggle_env: dict[str, str], master_root: Path) -> tuple[Path, dict, str]:
    handoff = _wait_for_current_g1a_handoff(repo, master_root / "handoff" / G1A_HANDOFF_FILE)
    locator = str(handoff["private_kaggle_dataset_locator"])
    print(f"Validating canonical G1A private access: {locator}")
    wait_kaggle_dataset_ready(locator, env=kaggle_env)
    bundle, seal = download_and_validate_g1a_dataset(repo, locator, master_root / "g1a-shared-download", env=kaggle_env)
    if seal["g1a_seal_sha256"] != handoff["g1a_seal_sha256"]:
        raise OperatorError("shared G1A seal differs from published handoff")
    return bundle, seal, locator
