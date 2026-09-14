from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import master_g1a as _base
from tracka_v12_kaggle_operator_v3 import SCIENCE_SHA, OperatorError, prepare_r13, resolve_principal_g1_bundle, validate_g1a_bundle_with_science
from master_verified_pretrained_v5 import prepare_verified_torchvision


def build_g1a_once_v6(repo: Path, *, manifest: Path, class_map: Path, image_root: Path, output_bundle: Path) -> dict:
    principal = resolve_principal_g1_bundle(override=os.environ.get("CROPCOP_PRINCIPAL_G1", ""))
    upstream_root = output_bundle.parent / "upstream"
    if upstream_root.exists():
        shutil.rmtree(upstream_root)
    upstream_root.mkdir(parents=True)

    baselines = _base._retry_pre_science_io(
        "Official TorchVision download + exact tensor provenance",
        lambda: prepare_verified_torchvision(repo, upstream_root / "torchvision"),
    )
    r13 = _base._retry_pre_science_io(
        "Pinned R13 Hugging Face artifact",
        lambda: prepare_r13(upstream_root / "r13"),
    )

    compat = Path(__file__).resolve().parent / "master_g1a_sealer_compat_v6.py"
    sealer_args = [
        "--repo-root", str(repo), "--authorized-source-sha", SCIENCE_SHA,
        "--manifest", str(manifest), "--class-map", str(class_map), "--image-root", str(image_root),
        "--principal-g1-bundle", str(principal),
        "--effb0-pretrained", str(baselines["effb0"]["artifact"]),
        "--effb0-provenance", str(baselines["effb0"]["provenance"]),
        "--cnxtt-pretrained", str(baselines["cnxtt"]["artifact"]),
        "--cnxtt-provenance", str(baselines["cnxtt"]["provenance"]),
        "--r13-pretrained", str(r13), "--bundle-dir", str(output_bundle),
    ]
    command = [sys.executable, "-u", str(compat), str(repo), "--", *sealer_args]
    cp = subprocess.run(command, cwd=repo, text=True)
    if cp.returncode != 0:
        raise OperatorError(f"G1A sealer compatibility runner failed with rc={cp.returncode}")
    return validate_g1a_bundle_with_science(repo, output_bundle)


# Deliberately patch only the G1A build seam. All handoff, private round-trip,
# publication, adoption, and worker-acquisition behavior remains the previously
# QA-passed implementation from master_g1a.py.
_base.build_g1a_once = build_g1a_once_v6

ensure_canonical_g1a_k1 = _base.ensure_canonical_g1a_k1
acquire_canonical_g1a_worker = _base.acquire_canonical_g1a_worker
G1A_HANDOFF_FILE = _base.G1A_HANDOFF_FILE
