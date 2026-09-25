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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
