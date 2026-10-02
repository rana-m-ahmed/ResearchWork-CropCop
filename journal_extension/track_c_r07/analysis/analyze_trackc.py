"""Independent recomputation for the committed Track-C public evidence.

This module consumes raw samples from results/public. It never accepts the
committed analysis_summary.json as numerical input.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median, stdev


PUBLIC_FILES = {
    "warmup_transition": "warmup_transition_int8.csv",
    "model_latency": "model_latency_int8.csv",
    "end_to_end": "end_to_end_int8.csv",
    "model_latency_fp32": "model_latency_fp32.csv",
    "load": "load_trials_int8.csv",
    "raw_pipeline_fidelity": "raw_pipeline_fidelity.csv",
    "tensor_fidelity": "int8_tensor_fidelity_device.csv",
}

EXPECTED_COUNTS = {
    "warmup_transition": 10,
    "model_latency": 1000,
    "end_to_end": 768,
    "model_latency_fp32": 1000,
    "load": 10,
    "raw_pipeline_fidelity": 256,
    "tensor_fidelity": 256,
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _quantile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def summarize_ns(path: Path) -> dict[str, float | int]:
    values = [int(row["elapsed_ns"]) for row in _rows(path)]
    if not values:
        raise ValueError(f"no samples: {path}")
    mean_ns = sum(values) / len(values)
    return {
        "samples": len(values),
        "median_ms": median(values) / 1e6,
        "p95_ms": _quantile([float(v) for v in values], 0.95) / 1e6,
        "mean_ms": mean_ns / 1e6,
        "sample_sd_ms": (stdev(values) / 1e6) if len(values) > 1 else 0.0,
        "throughput_per_s": 1e9 / mean_ns,
    }


def summarize_ms(path: Path, column: str) -> dict[str, float | int]:
    values = [float(row[column]) for row in _rows(path)]
    if not values:
        raise ValueError(f"no samples: {path}")
    return {
        "samples": len(values),
        "median_ms": median(values),
        "p95_ms": _quantile(values, 0.95),
        "mean_ms": sum(values) / len(values),
        "sample_sd_ms": stdev(values) if len(values) > 1 else 0.0,
    }


def summarize_raw_fidelity(path: Path) -> dict[str, float | int]:
    rows = _rows(path)
    matches = sum(row["match"].lower() == "true" for row in rows)
    return {
        "samples": len(rows),
        "matches": matches,
        "mismatches": len(rows) - matches,
        "agreement": matches / len(rows),
    }


def summarize_tensor_fidelity(path: Path) -> dict[str, float | int]:
    return {"samples": len(_rows(path))}


def build_summary(evidence_dir: Path) -> dict[str, dict[str, float | int]]:
    result = {
        "warmup_transition": summarize_ns(evidence_dir / PUBLIC_FILES["warmup_transition"]),
        "model_latency": summarize_ns(evidence_dir / PUBLIC_FILES["model_latency"]),
        "end_to_end": summarize_ns(evidence_dir / PUBLIC_FILES["end_to_end"]),
        "model_latency_fp32": summarize_ns(evidence_dir / PUBLIC_FILES["model_latency_fp32"]),
        "load": summarize_ms(evidence_dir / PUBLIC_FILES["load"], "am_total_ms"),
        "raw_pipeline_fidelity": summarize_raw_fidelity(
            evidence_dir / PUBLIC_FILES["raw_pipeline_fidelity"]
        ),
        "tensor_fidelity": summarize_tensor_fidelity(
            evidence_dir / PUBLIC_FILES["tensor_fidelity"]
        ),
    }
    for name, expected in EXPECTED_COUNTS.items():
        observed = int(result[name]["samples"])
        if observed != expected:
            raise ValueError(f"{name} sample count mismatch: {observed} != {expected}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_summary(args.evidence_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
