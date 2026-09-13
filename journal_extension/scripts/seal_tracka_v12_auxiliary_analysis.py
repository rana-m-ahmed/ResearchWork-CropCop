from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.hashing import sha256_file, sha256_json
from cropcop_je.tracka_v12_analysis import arithmetic_mean, sample_sd

SEEDS = ("S1", "S2", "S3")
R04 = {seed: f"R04-MNV4-DIRECT-{seed}" for seed in SEEDS}
R05 = {seed: f"R05-MNV4-TEACHER-{seed}" for seed in SEEDS}
R12_LOGITS = {seed: f"R12-MNV4-LOGITS-{seed}" for seed in SEEDS}
R12_FEATURE = {seed: f"R12-MNV4-FEATURE-{seed}" for seed in SEEDS}
REQUIRED = set(R04.values()) | set(R05.values()) | set(R12_LOGITS.values()) | set(R12_FEATURE.values())
METRICS = ("validation_macro_f1", "validation_balanced_accuracy", "validation_accuracy", "validation_nll")


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clean_summary(payload: dict) -> dict:
    if "clean_summary" in payload:
        return payload["clean_summary"]
    clean = payload.get("clean_summary") or payload.get("clean")
    if clean:
        return clean
    raise ValueError("evidence artifact lacks clean summary")


def summarize_condition(rows: dict[str, dict]) -> dict:
    if set(rows) != set(SEEDS):
        raise ValueError("condition evidence must contain exactly S1/S2/S3")
    output = {"seedwise": {seed: {metric: float(rows[seed][metric]) for metric in METRICS} for seed in SEEDS}}
    output["three_seed"] = {}
    for metric in METRICS:
        values = [float(rows[seed][metric]) for seed in SEEDS]
        output["three_seed"][metric] = {
            "mean": arithmetic_mean(values),
            "sample_sd": sample_sd(values),
            "worst_seed": min(values) if metric != "validation_nll" else max(values),
        }
    return output


def paired_summary(left: dict[str, dict], right: dict[str, dict], *, label: str) -> dict:
    result = {"label": label, "hypothesis_test_authorized": False, "metric_deltas": {}}
    for metric in METRICS:
        deltas = {seed: float(left[seed][metric]) - float(right[seed][metric]) for seed in SEEDS}
        values = list(deltas.values())
        result["metric_deltas"][metric] = {
            "seedwise": deltas,
            "mean": arithmetic_mean(values),
            "sample_sd": sample_sd(values),
        }
    class_deltas = []
    for class_index in range(120):
        seedwise = {
            seed: float(left[seed]["class_f1"][class_index]) - float(right[seed]["class_f1"][class_index])
            for seed in SEEDS
        }
        values = list(seedwise.values())
        class_deltas.append(
            {
                "class_index": class_index,
                "seedwise_f1_delta": seedwise,
                "mean_f1_delta": arithmetic_mean(values),
                "sample_sd_f1_delta": sample_sd(values),
            }
        )
    result["per_class_f1_delta"] = class_deltas
    result["class_delta_summary"] = {
        "mean_of_120_class_mean_deltas": arithmetic_mean(row["mean_f1_delta"] for row in class_deltas),
        "min_class_mean_delta": min(row["mean_f1_delta"] for row in class_deltas),
        "max_class_mean_delta": max(row["mean_f1_delta"] for row in class_deltas),
    }
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-index", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    index_path = Path(args.evidence_index).resolve()
    index = load_json(index_path)
    states = index.get("states", {})
    if index.get("schema_version") != "1.0" or set(states) != REQUIRED:
        raise SystemExit("auxiliary-analysis evidence index must contain exact R04/R05/R12 S1/S2/S3 inventory")

    loaded = {}
    hashes = {}
    for experiment_id, raw_path in states.items():
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise SystemExit(f"auxiliary-analysis evidence missing: {experiment_id}")
        payload = load_json(path)
        if payload.get("experiment_id") != experiment_id:
            raise SystemExit(f"auxiliary-analysis experiment identity mismatch: {experiment_id}")
        if payload.get("status") != "PASS":
            raise SystemExit(f"auxiliary-analysis state is not PASS: {experiment_id}")
        if payload.get("v1_test_accessed") is not False or payload.get("external_surface_accessed") is not False:
            raise SystemExit(f"protected surface marker invalid: {experiment_id}")
        if experiment_id.startswith("R04-"):
            summary = payload.get("clean_summary")
            if summary is None:
                robust_path = Path(index.get("r04_robustness_replay", {}).get(experiment_id, "")).resolve()
                if not robust_path.is_file():
                    raise SystemExit(f"R04 clean-summary source missing: {experiment_id}")
                summary = load_json(robust_path).get("clean_summary")
                hashes[f"{experiment_id}:robustness_replay"] = sha256_file(robust_path)
        else:
            if payload.get("replay_gate", {}).get("status") != "PASS" or payload.get("classwise_pass") is not True:
                raise SystemExit(f"auxiliary replay/classwise gate failed: {experiment_id}")
            summary = payload.get("clean_summary")
        if not summary or int(summary.get("row_count", -1)) != 16368 or len(summary.get("class_f1", [])) != 120:
            raise SystemExit(f"clean summary incomplete: {experiment_id}")
        loaded[experiment_id] = summary
        hashes[experiment_id] = sha256_file(path)

    def group(mapping):
        return {seed: loaded[experiment_id] for seed, experiment_id in mapping.items()}

    r04 = group(R04)
    r05 = group(R05)
    logits = group(R12_LOGITS)
    feature = group(R12_FEATURE)
    result = {
        "schema_version": "1.0",
        "status": "PASS",
        "closure_kind": "track_a_auxiliary_three_seed_analysis",
        "state_inventory": sorted(REQUIRED),
        "state_count": len(REQUIRED),
        "conditions": {
            "R04_direct": summarize_condition(r04),
            "R05_teacher": summarize_condition(r05),
            "R12_logits": summarize_condition(logits),
            "R12_feature": summarize_condition(feature),
        },
        "paired_analyses": {
            "R05_teacher_minus_R04_direct": paired_summary(r05, r04, label="R05 teacher minus R04 direct"),
            "R12_logits_minus_feature": paired_summary(logits, feature, label="R12 logits minus R12 feature"),
        },
        "hypothesis_tests_authorized": False,
        "multiple_comparison_p_values_authorized": False,
        "v1_test_accessed": False,
        "external_surface_accessed": False,
        "evidence_index_sha256": sha256_file(index_path),
        "evidence_sha256": hashes,
    }
    result["closure_sha256"] = sha256_json(result)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
