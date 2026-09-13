from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from tracka_v12_kaggle_operator_v3 import (
    SCIENCE_SHA, OperatorError, adopt_g1a_if_present, download_and_validate_g1a_dataset,
    ensure_private_dataset, g1a_handoff_run_id, load_json, operator_runtime_head,
    prepare_official_torchvision, prepare_r13, publish_public_files,
    resolve_principal_g1_bundle, shared_g1a_locator, validate_g1a_bundle_with_science,
    version_private_dataset, wait_for_public_file, wait_kaggle_dataset_ready, write_json,
)

G1A_HANDOFF_FILE = "TRACKA_V12_G1A_HANDOFF.json"


def _retry_pre_science_io(label: str, fn, *, attempts: int = 3):
    """Retry deterministic upstream downloads before any G1A result exists."""
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


def write_handoff(path: Path, *, status: str, locator: str, seal: dict | None = None) -> dict:
    payload = {
        "schema_version": "1.0", "stage": "TRACKA_V12_G1A_SHARED_HANDOFF",
        "status": status, "science_source_sha": SCIENCE_SHA,
        "operator_runtime_sha": operator_runtime_head(),
        "private_kaggle_dataset_locator": locator,
        "g1a_seal_sha256": None if seal is None else seal.get("g1a_seal_sha256"),
        "science_authorized": False, "protected_test_accessed": False,
        "external_surface_accessed": False,
    }
    write_json(path, payload)
    return payload


def g1a_public_report(path: Path, seal: dict, locator: str, stack: dict) -> dict:
    payload = {
        "schema_version": "1.0", "stage": "TRACKA_V12_G1A_MASTER", "status": "PASS",
        "science_source_sha": SCIENCE_SHA, "operator_runtime_sha": operator_runtime_head(),
        "g1a_seal_sha256": seal["g1a_seal_sha256"],
        "dependency_lock_sha256": stack["dependency_lock_sha256"],
        "private_handoff_locator": locator, "science_authorized": False,
        "protected_test_accessed": False, "external_surface_accessed": False,
    }
    write_json(path, payload)
    return payload


def build_g1a_once(repo: Path, *, manifest: Path, class_map: Path, image_root: Path, output_bundle: Path) -> dict:
    # Preflight binds CROPCOP_PRINCIPAL_G1 to one exact complete historical package.
    principal = resolve_principal_g1_bundle(override=os.environ.get("CROPCOP_PRINCIPAL_G1", ""))
    upstream_root = output_bundle.parent / "upstream"
    if upstream_root.exists():
        shutil.rmtree(upstream_root)
    upstream_root.mkdir(parents=True)

    baselines = _retry_pre_science_io(
        "Official TorchVision pretrained provenance",
        lambda: prepare_official_torchvision(repo, upstream_root / "torchvision"),
    )
    r13 = _retry_pre_science_io(
        "Pinned R13 Hugging Face artifact",
        lambda: prepare_r13(upstream_root / "r13"),
    )

    command = [
        sys.executable, str(repo / "journal_extension/scripts/seal_tracka_v12_g1a.py"),
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
        raise OperatorError(f"G1A sealer failed with rc={cp.returncode}")
    return validate_g1a_bundle_with_science(repo, output_bundle)


def ensure_canonical_g1a_k1(
    repo: Path, *, manifest: Path, class_map: Path, image_root: Path, username: str,
    kaggle_env: dict[str, str], stack: dict, master_root: Path,
) -> tuple[Path, dict, str]:
    locator = os.environ.get("CROPCOP_G1A_SHARED_DATASET", "").strip() or shared_g1a_locator(username)
    if locator.split("/", 1)[0].casefold() != username.casefold():
        raise OperatorError("K1 canonical G1A shared dataset must be owned by authenticated K1")
    ensure_private_dataset(locator, title="CropCop Track-A v1.2 canonical G1A", env=kaggle_env)
    handoff = master_root / G1A_HANDOFF_FILE
    print(f"Canonical private G1A dataset: {locator}")
    print("One-time setup: give K2 and K3 Can view access in Kaggle Dataset Settings > Sharing.")

    # An empty placeholder dataset is safe to resume; a present invalid/stale G1A seal remains a hard stop.
    adopted = adopt_g1a_if_present(repo, locator, master_root / "g1a-shared-download", env=kaggle_env)
    if adopted is not None:
        bundle, seal = adopted
        print("Reusing already-valid canonical G1A bundle.")
    else:
        write_handoff(handoff, status="PREPARING", locator=locator)
        publish_json(repo, g1a_handoff_run_id(), handoff)
        bundle = master_root / "TRACKA_V12_G1A_BUNDLE"
        shutil.rmtree(bundle, ignore_errors=True)
        seal = build_g1a_once(repo, manifest=manifest, class_map=class_map, image_root=image_root, output_bundle=bundle)
        version_private_dataset(locator, bundle, message=f"Canonical Track-A v1.2 G1A {SCIENCE_SHA[:12]}", env=kaggle_env)
        wait_kaggle_dataset_ready(locator, env=kaggle_env)
        roundtrip_bundle, roundtrip_seal = download_and_validate_g1a_dataset(
            repo, locator, master_root / "g1a-roundtrip", env=kaggle_env,
        )
        if roundtrip_seal["g1a_seal_sha256"] != seal["g1a_seal_sha256"]:
            raise OperatorError("private G1A round-trip changed canonical seal")
        bundle, seal = roundtrip_bundle, roundtrip_seal

    report = master_root / "TRACKA_V12_G1A_PUBLIC_REPORT.json"
    g1a_public_report(report, seal, locator, stack)
    public_seal = master_root / "TRACKA_V12_G1A_SEAL.json"
    shutil.copy2(bundle / "TRACKA_V12_G1A_SEAL.json", public_seal)
    publish_public_files(repo, f"TRACKA-V12-G1A-{SCIENCE_SHA[:12]}", [public_seal, report])
    write_handoff(handoff, status="READY", locator=locator, seal=seal)
    publish_json(repo, g1a_handoff_run_id(), handoff)
    return bundle, seal, locator


def acquire_canonical_g1a_worker(repo: Path, *, kaggle_env: dict[str, str], master_root: Path) -> tuple[Path, dict, str]:
    handoff_path = wait_for_public_file(
        repo, g1a_handoff_run_id(), G1A_HANDOFF_FILE, master_root / "handoff" / G1A_HANDOFF_FILE,
        predicate=lambda p: p.get("status") == "READY" and p.get("science_source_sha") == SCIENCE_SHA
        and len(str(p.get("g1a_seal_sha256", ""))) == 64,
    )
    handoff = load_json(handoff_path)
    locator = str(handoff["private_kaggle_dataset_locator"])
    print(f"Waiting for canonical G1A private access: {locator}")
    wait_kaggle_dataset_ready(locator, env=kaggle_env)
    bundle, seal = download_and_validate_g1a_dataset(repo, locator, master_root / "g1a-shared-download", env=kaggle_env)
    if seal["g1a_seal_sha256"] != handoff["g1a_seal_sha256"]:
        raise OperatorError("shared G1A seal differs from published handoff")
    return bundle, seal, locator
