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
    ensure_science_checkout,
    install_locked_stack,
    load_json,
    prepare_official_torchvision,
    prepare_r13,
    resolve_frozen_dataset,
    resolve_image_root,
    resolve_principal_g1_bundle,
    sha256_file,
    verify_locked_stack,
    write_json,
)


def main() -> int:
    assert_kaggle_paths()
    assert_python_version()

    output_bundle = Path("/kaggle/working/TRACKA_V12_G1A_BUNDLE")
    if output_bundle.exists():
        raise RuntimeError(
            f"{output_bundle} already exists. G1A is immutable; use a fresh Kaggle session "
            "instead of overwriting or partially reusing a previous attempt."
        )

    repo = ensure_science_checkout()
    install_locked_stack(repo)
    stack = verify_locked_stack(repo)
    assert_clean_science_checkout(repo)

    manifest, class_map = resolve_frozen_dataset(
        manifest_override=os.environ.get("CROPCOP_MANIFEST", ""),
        class_map_override=os.environ.get("CROPCOP_CLASS_MAP", ""),
    )
    image_root = resolve_image_root(
        manifest,
        override=os.environ.get("CROPCOP_IMAGE_ROOT", ""),
    )
    principal_g1 = resolve_principal_g1_bundle(
        override=os.environ.get("CROPCOP_PRINCIPAL_G1", ""),
    )

    upstream_root = Path("/kaggle/working/tracka_v12_upstream")
    if upstream_root.exists():
        shutil.rmtree(upstream_root)
    upstream_root.mkdir(parents=True)

    baselines = prepare_official_torchvision(repo, upstream_root / "torchvision")
    r13 = prepare_r13(upstream_root / "r13")
    assert_clean_science_checkout(repo)

    cmd = [
        sys.executable,
        str(repo / "journal_extension/scripts/seal_tracka_v12_g1a.py"),
        "--repo-root", str(repo),
        "--authorized-source-sha", SCIENCE_SHA,
        "--manifest", str(manifest),
        "--class-map", str(class_map),
        "--image-root", str(image_root),
        "--principal-g1-bundle", str(principal_g1),
        "--effb0-pretrained", str(baselines["effb0"]["artifact"]),
        "--effb0-provenance", str(baselines["effb0"]["provenance"]),
        "--cnxtt-pretrained", str(baselines["cnxtt"]["artifact"]),
        "--cnxtt-provenance", str(baselines["cnxtt"]["provenance"]),
        "--r13-pretrained", str(r13),
        "--bundle-dir", str(output_bundle),
    ]
    cp = subprocess.run(cmd, cwd=repo, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"G1A sealer failed with exit code {cp.returncode}")

    seal_path = output_bundle / "TRACKA_V12_G1A_SEAL.json"
    seal = load_json(seal_path)
    if seal.get("status") != "PASS":
        raise RuntimeError("G1A did not terminate PASS")
    if seal.get("science_authorized") is not False:
        raise RuntimeError("G1A must not authorize science")
    if seal.get("source_git_sha") != SCIENCE_SHA:
        raise RuntimeError("G1A source binding mismatch")

    report = {
        "schema_version": "1.1",
        "stage": "TRACKA_V12_G1A",
        "status": "PASS",
        "science_source_sha": SCIENCE_SHA,
        "dependency_lock_sha256": stack["dependency_lock_sha256"],
        "g1a_seal_sha256": seal["g1a_seal_sha256"],
        "g1a_seal_file_sha256": sha256_file(seal_path),
        "manifest_sha256": sha256_file(manifest),
        "class_map_sha256": sha256_file(class_map),
        "r13_sha256": sha256_file(r13),
        "protected_test_accessed": False,
        "external_surface_accessed": False,
    }
    write_json("/kaggle/working/TRACKA_V12_G1A_OPERATOR_REPORT.json", report)
    archive = shutil.make_archive(
        "/kaggle/working/TRACKA_V12_G1A_BUNDLE",
        "zip",
        root_dir=output_bundle,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"G1A bundle: {output_bundle}")
    print(f"Transfer archive: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
