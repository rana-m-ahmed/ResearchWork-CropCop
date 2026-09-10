from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.science_diff import validate_science_diff
from cropcop_je.secondary import (
    PRINCIPAL_SCIENCE_SOURCE_SHA,
    SECONDARY_CONFIG_SPECS,
    experiment_config_path,
    load_json,
    validate_secondary_config,
)
from cropcop_je.secondary_g2 import REQUIRED_SECONDARY_CALIBRATIONS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--output", default="")
    args = ap.parse_args()
    root = Path(args.repo_root).resolve()
    errors: list[str] = []

    principal = validate_science_diff(root)
    errors.extend("principal-science-diff: " + e for e in principal.get("errors", []))

    registry_path = root / "journal_extension/locks/secondary_experiment_registry.json"
    if not registry_path.is_file():
        registry = {"experiments": []}
        errors.append("secondary experiment registry missing")
    else:
        registry = load_json(registry_path)
    rows = registry.get("experiments", [])
    ids = [str(row.get("experiment_id", "")) for row in rows]
    if set(ids) != set(SECONDARY_CONFIG_SPECS) or len(ids) != len(SECONDARY_CONFIG_SPECS):
        errors.append("secondary registry must contain exactly the four frozen experiment IDs")
    if any("*" in eid for eid in ids):
        errors.append("secondary registry contains wildcard experiment authorization")
    by_id = {row["experiment_id"]: row for row in rows if row.get("experiment_id")}

    secondary_config_status = {}
    for eid in SECONDARY_CONFIG_SPECS:
        path = root / experiment_config_path(eid)
        if not path.is_file():
            errors.append(f"secondary config missing: {path.relative_to(root)}")
            continue
        cfg = load_json(path)
        cfg_errors = validate_secondary_config(cfg)
        secondary_config_status[eid] = {"path": str(path.relative_to(root)), "errors": cfg_errors}
        errors.extend(f"{eid}: {e}" for e in cfg_errors)
        row = by_id.get(eid, {})
        if row.get("state") != "active":
            errors.append(f"secondary registry state is not active: {eid}")
        if row.get("seed") != cfg.get("seed"):
            errors.append(f"secondary registry/config seed mismatch: {eid}")
        if row.get("config_path") != experiment_config_path(eid):
            errors.append(f"secondary registry/config path mismatch: {eid}")
        if row.get("selection_surface") != "DS-V1-VAL":
            errors.append(f"secondary selection surface drift: {eid}")
        if row.get("allowed_surfaces") != ["DS-V1-TRAIN", "DS-V1-VAL"]:
            errors.append(f"secondary allowed-surface drift: {eid}")

    calibrations = registry.get("calibrations", {})
    if tuple(calibrations) != REQUIRED_SECONDARY_CALIBRATIONS:
        errors.append("secondary registry G2 calibration set/order mismatch")
    if "CAL-EFFB0" not in calibrations:
        errors.append("secondary G2 does not explicitly qualify EfficientNet-B0")

    notebook = load_json(root / "journal_extension/kaggle/canonical_lane.ipynb")
    principal_code = "".join(c.get("source", []) for c in notebook.get("cells", []) if c.get("cell_type") == "code")
    if f'AUTHORIZED_SOURCE_SHA = "{PRINCIPAL_SCIENCE_SOURCE_SHA}"' not in principal_code:
        errors.append("principal canonical notebook f171 source binding changed")

    runner = (root / "journal_extension/scripts/run_secondary_training.py").read_text(encoding="utf-8")
    for token in (
        "durable checkpoint restore failed; fresh scientific restart is forbidden",
        "validate_secondary_g2_barrier_object",
        "completed_scientific_checkpoint_result",
        "run_training(",
    ):
        if token not in runner:
            errors.append(f"secondary runner missing required continuation/gate token: {token}")
    for forbidden in ("DistributedDataParallel", "nn.DataParallel", "SyncBatchNorm", "FullyShardedDataParallel"):
        if forbidden in runner:
            errors.append(f"secondary runner contains forbidden distributed scientific path: {forbidden}")

    report = {
        "schema_version": "2.0",
        "status": "PASS" if not errors else "FAIL",
        "principal_science_diff": principal,
        "principal_science_source_sha": PRINCIPAL_SCIENCE_SOURCE_SHA,
        "principal_registry_preserved_by_existing_science_sentinel": principal.get("status") == "PASS",
        "secondary_registry_path": str(registry_path.relative_to(root)),
        "secondary_config_status": secondary_config_status,
        "secondary_experiment_ids": sorted(SECONDARY_CONFIG_SPECS),
        "secondary_g2_calibrations": list(REQUIRED_SECONDARY_CALIBRATIONS),
        "errors": errors,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if not errors else 3


if __name__ == "__main__":
    raise SystemExit(main())
