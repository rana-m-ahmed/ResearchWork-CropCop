"""Fail-closed validation of the final public-safe Track-C evidence package."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from analyze_trackc import build_summary


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"TRACKC_EVIDENCE_INVALID: {message}")


def compare_values(actual, expected, path: str = "root") -> None:
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f"type mismatch {path}")
        require(set(actual) == set(expected), f"key mismatch {path}")
        for key in expected:
            compare_values(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, bool):
        require(actual is expected, f"value mismatch {path}")
        return
    if isinstance(expected, (int, float)):
        require(isinstance(actual, (int, float)), f"numeric type mismatch {path}")
        require(
            math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-9),
            f"numeric mismatch {path}: {actual} != {expected}",
        )
        return
    require(actual == expected, f"value mismatch {path}: {actual!r} != {expected!r}")


def main() -> None:
    trackc_root = Path(__file__).resolve().parents[1]
    root = trackc_root / "results" / "public"

    manifest = json.loads((root / "EVIDENCE_MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = root / name
        require(path.is_file(), f"missing {name}")
        actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        require(actual == expected["sha256"], f"hash mismatch {name}")
        require(path.stat().st_size == expected["bytes"], f"byte count mismatch {name}")

    expected_counts = {
        "model_latency_int8.csv": 1000,
        "model_latency_fp32.csv": 1000,
        "end_to_end_int8.csv": 768,
        "warmup_transition_int8.csv": 10,
        "load_trials_int8.csv": 10,
        "int8_tensor_fidelity_device.csv": 256,
        "int8_host_device_fidelity.csv": 256,
        "fp32_host_device_fidelity.csv": 256,
        "raw_pipeline_fidelity.csv": 256,
    }
    for name, count in expected_counts.items():
        require(len(rows(root / name)) == count, f"row count {name}")

    int8_fidelity = rows(root / "int8_host_device_fidelity.csv")
    fp32_fidelity = rows(root / "fp32_host_device_fidelity.csv")
    require(
        sum(row["match"] == "true" for row in int8_fidelity) == 256,
        "final INT8 canonical host-device result is not 256/256",
    )
    require(
        sum(row["match"] == "true" for row in fp32_fidelity) == 256,
        "final FP32 canonical host-device result is not 256/256",
    )

    tensor_device = rows(root / "int8_tensor_fidelity_device.csv")
    for i, (device_row, fidelity_row) in enumerate(zip(tensor_device, int8_fidelity)):
        require(
            device_row["sample_index"] == fidelity_row["sample_index"],
            f"INT8 tensor sample-index mismatch at row {i}",
        )
        require(
            device_row["top1"] == fidelity_row["device_top1"],
            f"INT8 tensor/device top1 mismatch at row {i}",
        )

    raw = rows(root / "raw_pipeline_fidelity.csv")
    raw_matches = sum(row["match"] == "true" for row in raw)
    require(raw_matches == 254, f"raw pipeline final result changed: {raw_matches}/256")

    build = json.loads((root / "BUILD_MANIFEST.json").read_text(encoding="utf-8"))
    require(build["device"]["model"] == "Pixel 7", "device identity changed")
    require(build["device"]["abi"] == "arm64-v8a", "device ABI changed")
    require(build["runtime"]["version"] == "1.3.1", "ExecuTorch version changed")
    require(build["runtime"]["backend"] == "XNNPACK CPU", "runtime backend changed")
    require(build["runtime"]["requested_threads"] == 4, "requested-thread policy changed")
    require(
        build["artifacts"]["int8"]["sha256"]
        == "2e0c54a1b5bb7c0018d0159a642e49e4f4bd79fcc1f940e215a40734a2695bb8",
        "INT8 artifact identity changed",
    )
    require(
        build["artifacts"]["fp32"]["sha256"]
        == "61556330cd9fc4725b4ff4759aef3835696b81865b2cbe030f815a4c0e8d5aec",
        "FP32 artifact identity changed",
    )

    committed_summary = json.loads(
        (root / "analysis_summary.json").read_text(encoding="utf-8")
    )
    recomputed_summary = build_summary(root)
    compare_values(recomputed_summary, committed_summary, "analysis_summary")

    forbidden = {".pte", ".zip", ".npy", ".npz", ".pt", ".ckpt"}
    require(
        not [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in forbidden
        ],
        "restricted payload in public package",
    )
    print("PASS_TRACKC_PUBLIC_EVIDENCE_VALIDATION")


if __name__ == "__main__":
    main()
