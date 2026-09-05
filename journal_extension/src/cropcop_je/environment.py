from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

SAFE_KAGGLE_ENV_KEYS = (
    "KAGGLE_KERNEL_RUN_TYPE",
    "KAGGLE_KERNEL_ID",
    "KAGGLE_DOCKER_IMAGE",
    "KAGGLE_CONTAINER_NAME",
)

LOCKED_CORE = {
    "python": "3.12.13",
    "torch": "2.12.1",
    "torchvision": "0.27.1",
    "timm": "1.0.26",
}


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nvidia_smi() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        cp = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10)
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    rows = []
    for line in cp.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3:
            rows.append({"name": parts[0], "uuid": parts[1], "driver_version": parts[2]})
    return {"available": True, "gpus": rows}


def capture_environment() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": platform.python_version(),
        "python_executable": Path(sys.executable).name,
        "platform": platform.platform(),
        "packages": {
            "torch": _version("torch"),
            "torchvision": _version("torchvision"),
            "timm": _version("timm"),
            "numpy": _version("numpy"),
            "Pillow": _version("Pillow"),
            "safetensors": _version("safetensors"),
            "kaggle": _version("kaggle"),
        },
        "kaggle": {k: os.environ[k] for k in SAFE_KAGGLE_ENV_KEYS if k in os.environ},
        "nvidia_smi": _nvidia_smi(),
    }
    try:
        import torch
        payload["cuda_runtime"] = torch.version.cuda
        payload["cudnn"] = torch.backends.cudnn.version()
        payload["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            payload["cuda_device_count"] = torch.cuda.device_count()
            payload["torch_cuda_devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": list(torch.cuda.get_device_capability(i)),
                    "total_memory": torch.cuda.get_device_properties(i).total_memory,
                }
                for i in range(torch.cuda.device_count())
            ]
    except Exception as exc:
        payload["torch_runtime_error"] = f"{type(exc).__name__}: {exc}"
    return payload


def validate_locked_core(env: dict[str, Any]) -> dict[str, dict[str, str | None]]:
    observed = {
        "python": env.get("python"),
        "torch": env.get("packages", {}).get("torch"),
        "torchvision": env.get("packages", {}).get("torchvision"),
        "timm": env.get("packages", {}).get("timm"),
    }
    return {
        k: {"required": v, "observed": observed.get(k)}
        for k, v in LOCKED_CORE.items()
        if observed.get(k) != v
    }


def software_stack_identity(env: dict[str, Any]) -> dict[str, Any]:
    return {
        "python": env.get("python"),
        "packages": dict(env.get("packages", {})),
        "cuda_runtime": env.get("cuda_runtime"),
        "cudnn": env.get("cudnn"),
    }
