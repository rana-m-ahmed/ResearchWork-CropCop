"""Independent summaries for raw Track-C device evidence CSVs.

This module intentionally consumes raw samples; it never accepts precomputed
device summaries as numerical input.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median, stdev


def summarize_ns(path: Path) -> dict[str, float | int]:
    with path.open(newline="", encoding="utf-8") as handle:
        values = [int(row["elapsed_ns"]) for row in csv.DictReader(handle)]
    if not values:
        raise ValueError(f"no samples: {path}")
    ordered = sorted(values)
    def quantile(p: float) -> float:
        index = (len(ordered) - 1) * p
        low, high = math.floor(index), math.ceil(index)
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)
    return {
        "samples": len(values),
        "median_ms": median(values) / 1e6,
        "p95_ms": quantile(0.95) / 1e6,
        "mean_ms": (sum(values) / len(values)) / 1e6,
        "sample_sd_ms": (stdev(values) / 1e6) if len(values) > 1 else 0.0,
        "throughput_per_s": 1e9 / (sum(values) / len(values)),
    }


def summarize_ms(path: Path, column: str) -> dict[str, float | int]:
    with path.open(newline="", encoding="utf-8") as handle:
        values = [float(row[column]) for row in csv.DictReader(handle)]
    if not values:
        raise ValueError(f"no samples: {path}")
    ordered = sorted(values)
    index = (len(ordered) - 1) * 0.95
    low, high = math.floor(index), math.ceil(index)
    return {
        "samples": len(values),
        "median_ms": median(values),
        "p95_ms": ordered[low] + (ordered[high] - ordered[low]) * (index - low),
        "mean_ms": sum(values) / len(values),
        "sample_sd_ms": stdev(values) if len(values) > 1 else 0.0,
    }


def summarize_raw_fidelity(path: Path) -> dict[str, float | int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 256:
        raise ValueError(f"raw fidelity sample count mismatch: {len(rows)}")
    matches = sum(row["match"].lower() == "true" for row in rows)
    return {"samples": len(rows), "matches": matches, "mismatches": len(rows) - matches, "agreement": matches / len(rows)}


def summarize_tensor_fidelity(path: Path) -> dict[str, float | int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 256:
        raise ValueError(f"tensor fidelity sample count mismatch: {len(rows)}")
    return {"samples": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = {"warmup_transition": 10, "model_latency": 1000, "end_to_end": 768}
    result = {name: summarize_ns(args.evidence_dir / f"{name}.csv") for name in expected if (args.evidence_dir / f"{name}.csv").is_file()}
    for name, count in expected.items():
        if name in result and result[name]["samples"] != count:
            raise ValueError(f"{name} sample count mismatch")
    fp32 = args.evidence_dir / "model_latency_fp32.csv"
    if fp32.is_file():
        result["model_latency_fp32"] = summarize_ns(fp32)
        if result["model_latency_fp32"]["samples"] != 1000:
            raise ValueError("model_latency_fp32 sample count mismatch")
    load = args.evidence_dir / "load_trials_valid.csv"
    if load.is_file():
        result["load"] = summarize_ms(load, "am_total_ms")
        if result["load"]["samples"] != 10:
            raise ValueError("load sample count mismatch")
    raw_fidelity = args.evidence_dir / "raw_fidelity_uint8_quantized.csv"
    if raw_fidelity.is_file():
        result["raw_pipeline_fidelity"] = summarize_raw_fidelity(raw_fidelity)
    tensor_fidelity = args.evidence_dir / "tensor_fidelity_final.csv"
    if tensor_fidelity.is_file():
        result["tensor_fidelity"] = summarize_tensor_fidelity(tensor_fidelity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
