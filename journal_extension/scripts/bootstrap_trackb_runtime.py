from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

EXPECTED = {
    "torch": "2.12.1",
    "torchvision": "0.27.1",
    "timm": "1.0.26",
    "numpy": "2.5.2",
    "Pillow": "12.3.0",
    "transformers": "5.0.0",
    "huggingface-hub": "1.30.0",
    "safetensors": "0.8.0",
    "opencv-python-headless": "4.13.0.92",
    "kaggle": "2.2.4",
}
OPENCV_VARIANTS = (
    "opencv-python",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
)
EXPECTED_PYTHON = "3.12.13"


class BootstrapError(RuntimeError):
    pass


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _metadata_snapshot() -> dict[str, str | None]:
    return {name: _version(name) for name in EXPECTED}


def _normalize(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    if name in {"torch", "torchvision"}:
        return value.split("+", 1)[0]
    return value


def _drift(snapshot: dict[str, str | None]) -> dict[str, dict[str, str | None]]:
    bad = {}
    for name, expected in EXPECTED.items():
        observed = _normalize(name, snapshot.get(name))
        if observed != expected:
            bad[name] = {"expected": expected, "observed": observed}
    return bad


def _run(
    args: list[str],
    *,
    timeout: int = 7200,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        args,
        env=dict(os.environ),
        capture_output=capture,
        text=True,
        check=False,
        timeout=timeout,
    )
    if proc.returncode != 0:
        detail = ""
        if capture:
            detail = (proc.stderr or proc.stdout or "").strip()
            if len(detail) > 8000:
                detail = detail[-8000:]
        raise BootstrapError(
            f"command failed rc={proc.returncode}: {' '.join(args)}"
            + (f"\n{detail}" if detail else "")
        )
    return proc


def _disk(path: Path) -> dict[str, float]:
    use = shutil.disk_usage(path)
    return {
        "total_gib": round(use.total / 1024**3, 3),
        "used_gib": round(use.used / 1024**3, 3),
        "free_gib": round(use.free / 1024**3, 3),
    }


def _remove_conflicting_opencv_variants() -> list[str]:
    installed = [name for name in OPENCV_VARIANTS if _version(name) is not None]
    if installed:
        print(
            "Removing conflicting OpenCV wheel variants before installing the "
            f"frozen headless wheel: {installed}",
            flush=True,
        )
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "uninstall",
                "-y",
                *installed,
            ],
            timeout=1200,
            capture=False,
        )
    return installed


def _fresh_runtime_probe() -> dict[str, object]:
    code = r"""
import importlib.metadata as metadata
import json, platform
import cv2
import numpy as np
import PIL
import safetensors
import timm
import torch
import torchvision
import transformers
import huggingface_hub

names = [
    "torch", "torchvision", "timm", "numpy", "Pillow", "transformers",
    "huggingface-hub", "safetensors", "opencv-python-headless", "kaggle",
]
payload = {
    "python": platform.python_version(),
    "versions": {name: metadata.version(name) for name in names},
    "torch_runtime_version": torch.__version__,
    "torchvision_runtime_version": torchvision.__version__,
    "timm_runtime_version": timm.__version__,
    "numpy_runtime_version": np.__version__,
    "pillow_runtime_version": PIL.__version__,
    "opencv_runtime_version": cv2.__version__,
    "transformers_runtime_version": transformers.__version__,
    "huggingface_hub_runtime_version": huggingface_hub.__version__,
    "safetensors_runtime_version": safetensors.__version__,
    "torch_cuda_version": torch.version.cuda,
    "cuda_available": bool(torch.cuda.is_available()),
    "cuda_device_count": int(torch.cuda.device_count()),
    "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
}
print(json.dumps(payload, sort_keys=True))
"""
    result = _run([sys.executable, "-c", code], timeout=900, capture=True)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise BootstrapError("fresh scientific subprocess returned no runtime probe")
    try:
        return json.loads(lines[-1])
    except Exception as exc:
        raise BootstrapError(
            f"fresh scientific subprocess returned invalid JSON: {lines[-1]!r}"
        ) from exc


def _validate_probe(probe: dict[str, object]) -> None:
    if probe.get("python") != EXPECTED_PYTHON:
        raise BootstrapError(
            f"Python drift after lock repair: expected {EXPECTED_PYTHON}, got {probe.get('python')}"
        )
    versions = probe.get("versions")
    if not isinstance(versions, dict):
        raise BootstrapError("fresh runtime probe lacks package-version mapping")
    bad = _drift({str(k): None if v is None else str(v) for k, v in versions.items()})
    if bad:
        raise BootstrapError(f"fresh runtime package verification failed: {json.dumps(bad, sort_keys=True)}")
    if probe.get("cuda_available") is not True:
        raise BootstrapError("frozen Track-B runtime has no CUDA access")
    if int(probe.get("cuda_device_count", 0) or 0) < 1:
        raise BootstrapError("frozen Track-B runtime sees zero CUDA devices")
    if not str(probe.get("torch_cuda_version") or "").strip():
        raise BootstrapError("frozen Track-B torch build exposes no CUDA runtime")
    if str(probe.get("opencv_runtime_version")) != "4.13.0":
        raise BootstrapError(
            f"OpenCV runtime drift: expected cv2 4.13.0, got {probe.get('opencv_runtime_version')}"
        )


def bootstrap(requirements: Path, receipt_path: Path) -> dict[str, object]:
    if platform.python_version() != EXPECTED_PYTHON:
        raise BootstrapError(
            f"Track-B requires Kaggle Python {EXPECTED_PYTHON}; got {platform.python_version()}"
        )
    if not requirements.is_file():
        raise BootstrapError(f"Track-B requirements lock missing: {requirements}")

    working = Path("/kaggle/working")
    pre = _metadata_snapshot()
    pre_drift = _drift(pre)
    removed_opencv_variants: list[str] = []
    repaired = bool(pre_drift)

    if pre_drift:
        print(
            "Frozen stack verification requires repair: "
            + json.dumps(pre_drift, sort_keys=True),
            flush=True,
        )
        print(
            "Installing exact Track-B lock once with the active Kaggle interpreter; "
            "scientific execution will occur only in a fresh subprocess afterward.",
            flush=True,
        )
        removed_opencv_variants = _remove_conflicting_opencv_variants()
        _run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "--no-cache-dir",
                "-r",
                str(requirements),
            ],
            timeout=7200,
            capture=False,
        )

    # Never trust modules already present in the notebook process after pip repair.
    # The scientific stack is imported and verified only in a fresh child interpreter.
    probe = _fresh_runtime_probe()
    _validate_probe(probe)

    receipt = {
        "schema_version": "2.0",
        "status": "PASS",
        "mode": (
            "ACTIVE_INTERPRETER_LOCK_REPAIRED_FRESH_SUBPROCESS"
            if repaired
            else "ACTIVE_INTERPRETER_ALREADY_EXACT_FRESH_SUBPROCESS"
        ),
        "python_executable": sys.executable,
        "requirements_lock": str(requirements),
        "python_expected": EXPECTED_PYTHON,
        "pre_install_versions": pre,
        "pre_install_drift": pre_drift,
        "removed_conflicting_opencv_variants": removed_opencv_variants,
        "pip_cache_disabled": True,
        "venv_used": False,
        "ensurepip_used": False,
        "scientific_execution_requires_fresh_subprocess": True,
        "disk_before_or_after_repair": _disk(working),
        "probe": probe,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Repair Kaggle's active interpreter to the frozen CropCop Track-B lock "
            "and verify it from a fresh scientific subprocess."
        )
    )
    ap.add_argument(
        "--requirements",
        required=True,
        help="Exact Track-B requirements lock.",
    )
    ap.add_argument(
        "--receipt",
        default="/kaggle/working/TRACKB_RUNTIME_BOOTSTRAP.json",
    )
    args = ap.parse_args()

    receipt = bootstrap(
        Path(args.requirements).resolve(),
        Path(args.receipt).resolve(),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
