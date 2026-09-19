from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TORCH_INDEX = "https://download.pytorch.org/whl/cu126"
EXPECTED = {
    "torch": "2.12.1",
    "torchvision": "0.27.1",
    "timm": "1.0.26",
    "numpy": "2.5.2",
    "Pillow": "12.3.0",
    "transformers": "5.0.0",
    "huggingface-hub": "1.30.0",
    "safetensors": "0.8.0",
    "opencv-python-headless": "4.12.0.88",
}
NON_TORCH_REQUIREMENTS = [
    "timm==1.0.26",
    "numpy==2.5.2",
    "Pillow==12.3.0",
    "transformers==5.0.0",
    "huggingface-hub==1.30.0",
    "safetensors==0.8.0",
    "opencv-python-headless==4.12.0.88",
]


class BootstrapError(RuntimeError):
    pass


def _run(args: list[str], *, timeout: int = 7200) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        args,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        if len(detail) > 6000:
            detail = detail[-6000:]
        raise BootstrapError(f"command failed rc={proc.returncode}: {' '.join(args)}\n{detail}")
    return proc


def _venv_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _metadata_probe(python: Path) -> dict[str, object]:
    code = r'''
import importlib.metadata as metadata
import json, platform
import torch, torchvision, timm
expected_names = [
    "torch", "torchvision", "timm", "numpy", "Pillow",
    "transformers", "huggingface-hub", "safetensors", "opencv-python-headless",
]
print(json.dumps({
    "python": platform.python_version(),
    "versions": {name: metadata.version(name) for name in expected_names},
    "torch_runtime_version": torch.__version__,
    "torchvision_runtime_version": torchvision.__version__,
    "torch_cuda_version": torch.version.cuda,
    "cuda_available": bool(torch.cuda.is_available()),
    "cuda_device_count": int(torch.cuda.device_count()),
    "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "timm_runtime_version": timm.__version__,
}, sort_keys=True))
'''
    result = _run([str(python), "-c", code], timeout=600)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise BootstrapError("isolated runtime metadata probe returned no output")
    try:
        return json.loads(lines[-1])
    except Exception as exc:
        raise BootstrapError(f"invalid isolated runtime probe output: {lines[-1]!r}") from exc


def _matches(probe: dict[str, object]) -> bool:
    versions = probe.get("versions") or {}
    if not isinstance(versions, dict):
        return False
    for name, expected in EXPECTED.items():
        observed = str(versions.get(name, ""))
        if name in {"torch", "torchvision"}:
            observed = observed.split("+", 1)[0]
        if observed != expected:
            return False
    return (
        str(probe.get("timm_runtime_version", "")) == EXPECTED["timm"]
        and probe.get("cuda_available") is True
        and int(probe.get("cuda_device_count", 0) or 0) >= 1
    )


def bootstrap(venv_root: Path, receipt_path: Path, *, force: bool = False) -> dict[str, object]:
    if sys.version_info[:2] != (3, 12):
        raise BootstrapError(
            f"Track-B runtime bootstrap requires Python 3.12.x host; got {sys.version.split()[0]}"
        )

    python = _venv_python(venv_root)
    if not force and python.is_file():
        try:
            probe = _metadata_probe(python)
            if _matches(probe):
                receipt = {
                    "schema_version": "1.0",
                    "status": "PASS",
                    "mode": "REUSED_EXISTING_ISOLATED_ENVIRONMENT",
                    "venv_python": str(python),
                    "torch_index": TORCH_INDEX,
                    "probe": probe,
                }
                receipt_path.parent.mkdir(parents=True, exist_ok=True)
                receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
                return receipt
        except Exception:
            pass

    if venv_root.exists():
        shutil.rmtree(venv_root)
    venv_root.parent.mkdir(parents=True, exist_ok=True)

    _run([sys.executable, "-m", "venv", str(venv_root)], timeout=600)
    python = _venv_python(venv_root)
    if not python.is_file():
        raise BootstrapError(f"venv Python missing after creation: {python}")

    _run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir", "--upgrade", "pip"], timeout=1200)
    _run(
        [
            str(python), "-m", "pip", "install",
            "--disable-pip-version-check", "--no-cache-dir",
            "--index-url", TORCH_INDEX,
            "torch==2.12.1", "torchvision==0.27.1",
        ],
        timeout=7200,
    )
    _run(
        [
            str(python), "-m", "pip", "install",
            "--disable-pip-version-check", "--no-cache-dir",
            *NON_TORCH_REQUIREMENTS,
        ],
        timeout=7200,
    )

    probe = _metadata_probe(python)
    if not _matches(probe):
        raise BootstrapError(f"isolated runtime verification failed: {json.dumps(probe, sort_keys=True)}")

    receipt = {
        "schema_version": "1.0",
        "status": "PASS",
        "mode": "CREATED_ISOLATED_ENVIRONMENT",
        "venv_python": str(python),
        "torch_index": TORCH_INDEX,
        "pip_cache_disabled": True,
        "live_kernel_packages_modified": False,
        "probe": probe,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    ap = argparse.ArgumentParser(description="Create and verify the isolated CropCop Track-B Kaggle runtime.")
    ap.add_argument("--venv-root", default="/kaggle/working/trackb_runtime_env")
    ap.add_argument("--receipt", default="/kaggle/working/TRACKB_RUNTIME_BOOTSTRAP.json")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    receipt = bootstrap(Path(args.venv_root).resolve(), Path(args.receipt).resolve(), force=args.force)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
