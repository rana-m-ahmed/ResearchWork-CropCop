"""Fail-closed validation of the separately versioned POCO M3 compatibility campaign."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def require(value: bool, message: str) -> None:
    if not value:
        raise SystemExit(f"TRACKC_POCO_COMPATIBILITY_INVALID: {message}")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "results" / "public" / "device2_poco_m3_compat"
    manifest = json.loads((root / "POCO_COMPATIBILITY_MANIFEST.json").read_text(encoding="utf-8"))
    expected_artifact = "2e0c54a1b5bb7c0018d0159a642e49e4f4bd79fcc1f940e215a40734a2695bb8"
    require(manifest["status"] == "PASS_SEPARATE_RUNTIME_COMPATIBILITY_CAMPAIGN", "campaign status")
    require(manifest["device"]["abi"] == "arm64-v8a", "device ABI")
    require(manifest["artifact"]["sha256"] == expected_artifact, "frozen artifact identity")
    require(manifest["runtime"]["label"] == "1.3.1-poco-generic-dotprod-off", "runtime identity")
    require("not a direct Pixel 7 performance comparison" in manifest["scope"], "comparison boundary")
    expected_files = {
        "tensor_fidelity.csv": (4295, "f1454e247916a6d3e04c67e196b0782e5bda4a76c3235ae70cb7bb805cc8d758", 256),
        "raw_fidelity.csv": (3822, "2c3947cd4a0c86a38510347219036be598c0a1f243d4688880408078f673cc43", 256),
        "warmup_transition.csv": (169, "885829647cd6cd01de7fab412c120df0152da7886d8237c9830af7e3c925b61f", 10),
        "model_latency.csv": (15589, "ff71227941a44ae47fed79e57c8bb3a9e3c34681828f130ef9211757bc51ec9b", 1000),
        "end_to_end.csv": (13189, "e83d858f5a499329a809917e928f0c04c58561b78e6fa49af3a21537c6dbb756", 768),
        "load_readiness_trials.csv": (835, "06ae81ea3bbc6c97e35a04e9b0b75dca0fe617307a1f1362061e60290c1b70d6", 10),
    }
    for name, (byte_count, digest, count) in expected_files.items():
        path = root / name
        contents = path.read_bytes()
        require(len(contents) == byte_count, f"byte count {name}")
        require(hashlib.sha256(contents).hexdigest() == digest, f"hash {name}")
        require(len(rows(path)) == count, f"row count {name}")
    tensor = rows(root / "tensor_fidelity.csv")
    raw = rows(root / "raw_fidelity.csv")
    loads = rows(root / "load_readiness_trials.csv")
    primary = rows(root.parent / "int8_host_device_fidelity.csv")
    expected_top1 = {int(row["sample_index"]): int(row["expected_top1"]) for row in primary}
    mismatches = [
        int(row["sample_index"])
        for row in tensor
        if int(row["top1"]) != expected_top1[int(row["sample_index"])]
    ]
    require(mismatches == [155], "tensor fidelity reconciliation")
    require(sum(row["match"] == "true" for row in raw) == 253, "raw compatibility matches")
    require(all(row["ready_nonclaim"] == "true" and row["artifact_sha256"] == expected_artifact for row in loads), "fresh load readiness")
    print("PASS_TRACKC_POCO_COMPATIBILITY_VALIDATION")


if __name__ == "__main__":
    main()
