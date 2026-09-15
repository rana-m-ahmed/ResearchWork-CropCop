from __future__ import annotations

import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import master_g1a as _v5
from master_verified_pretrained_v5 import prepare_verified_torchvision
from tracka_v12_kaggle_operator_v3 import (
    SCIENCE_SHA,
    OperatorError,
    prepare_r13,
    resolve_principal_g1_bundle,
    validate_g1a_bundle_with_science,
)


def _compat_launcher_source() -> str:
    """Return the tiny launcher used to execute the unchanged frozen sealer.

    The frozen Track-A sealer at SCIENCE_SHA uses TORCHVISION_VERSION inside main()
    but does not import that symbol. The canonical value already lives in the frozen
    cropcop_je.secondary module. We inject exactly that frozen constant into the
    script globals without editing the science checkout.
    """
    return (
        "import runpy, sys\n"
        "from pathlib import Path\n"
        "script = Path(sys.argv[1]).resolve()\n"
        "forward = sys.argv[2:]\n"
        "sys.path.insert(0, str(script.parent))\n"
        "import _bootstrap  # noqa: F401\n"
        "from cropcop_je.secondary import TORCHVISION_VERSION\n"
        "sys.argv = [str(script), *forward]\n"
        "runpy.run_path(str(script), run_name='__main__', init_globals={'TORCHVISION_VERSION': TORCHVISION_VERSION})\n"
    )


def frozen_sealer_command(repo: Path, forwarded_args: list[str]) -> list[str]:
    script = repo / "journal_extension/scripts/seal_tracka_v12_g1a.py"
    if not script.is_file():
        raise OperatorError(f"frozen G1A sealer missing: {script}")
    return [sys.executable, "-c", _compat_launcher_source(), str(script), *forwarded_args]


def build_g1a_once(
    repo: Path,
    *,
    manifest: Path,
    class_map: Path,
    image_root: Path,
    output_bundle: Path,
) -> dict:
    principal = resolve_principal_g1_bundle(override=os.environ.get("CROPCOP_PRINCIPAL_G1", ""))
    upstream_root = output_bundle.parent / "upstream"
    if upstream_root.exists():
        shutil.rmtree(upstream_root)
    upstream_root.mkdir(parents=True)

    baselines = _v5._retry_pre_science_io(
        "Official TorchVision download + exact tensor provenance",
        lambda: prepare_verified_torchvision(repo, upstream_root / "torchvision"),
    )
    r13 = _v5._retry_pre_science_io(
        "Pinned R13 Hugging Face artifact",
        lambda: prepare_r13(upstream_root / "r13"),
    )

    forwarded = [
        "--repo-root", str(repo),
        "--authorized-source-sha", SCIENCE_SHA,
        "--manifest", str(manifest),
        "--class-map", str(class_map),
        "--image-root", str(image_root),
        "--principal-g1-bundle", str(principal),
        "--effb0-pretrained", str(baselines["effb0"]["artifact"]),
        "--effb0-provenance", str(baselines["effb0"]["provenance"]),
        "--cnxtt-pretrained", str(baselines["cnxtt"]["artifact"]),
        "--cnxtt-provenance", str(baselines["cnxtt"]["provenance"]),
        "--r13-pretrained", str(r13),
        "--bundle-dir", str(output_bundle),
    ]

    print(
        "FROZEN_G1A_SEALER_COMPAT: injecting TORCHVISION_VERSION from "
        "frozen cropcop_je.secondary; science checkout remains byte-clean.",
        flush=True,
    )
    cp = subprocess.run(frozen_sealer_command(repo, forwarded), cwd=repo, text=True)
    if cp.returncode != 0:
        raise OperatorError(f"G1A sealer failed with rc={cp.returncode}")
    return validate_g1a_bundle_with_science(repo, output_bundle)


def ensure_canonical_g1a_k1(*args, **kwargs):
    """Use the v5 recovery/publication logic with only the sealer launch replaced."""
    original = _v5.build_g1a_once
    _v5.build_g1a_once = build_g1a_once
    try:
        return _v5.ensure_canonical_g1a_k1(*args, **kwargs)
    finally:
        _v5.build_g1a_once = original


acquire_canonical_g1a_worker = _v5.acquire_canonical_g1a_worker
