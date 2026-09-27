"""Fail-closed validation of the public-safe Track-C evidence package."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"TRACKC_EVIDENCE_INVALID: {message}")


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "results" / "public"
    manifest = json.loads((root / "EVIDENCE_MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = root / name
        require(path.is_file(), f"missing {name}")
        # Evidence is public text committed through Git; canonicalize Windows
        # line endings before comparing against the Git-canonical manifest.
        actual = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        require(actual == expected["sha256"], f"hash mismatch {name}")

    for name, count in {
        "model_latency_int8.csv": 1000,
        "model_latency_fp32.csv": 1000,
        "end_to_end_int8.csv": 768,
        "warmup_transition_int8.csv": 10,
        "load_trials_int8.csv": 10,
    }.items():
        require(len(rows(root / name)) == count, f"row count {name}")

    for name in ("int8_host_device_fidelity.csv", "fp32_host_device_fidelity.csv"):
        fidelity = rows(root / name)
        require(len(fidelity) == 256, f"row count {name}")
        require(sum(row["match"] == "true" for row in fidelity) >= 255, f"fidelity acceptance {name}")

    raw = rows(root / "raw_pipeline_fidelity.csv")
    require(len(raw) == 256, "raw pipeline row count")
    require(sum(row["match"] == "true" for row in raw) == 254, "raw pipeline bounded-result mismatch")

    forbidden = {".pte", ".zip", ".npy", ".npz", ".pt", ".ckpt"}
    require(not [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in forbidden], "restricted payload in public package")
    print("PASS_TRACKC_PUBLIC_EVIDENCE_VALIDATION")


if __name__ == "__main__":
    main()
