from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.g1_publication import ensure_private_target
from cropcop_je.hashing import sha256_file, sha256_json


def load_json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--account-id", choices=["K1", "K2", "K3"], required=True)
    ap.add_argument("--account-inventory", required=True)
    ap.add_argument("--analysis-source-git-commit", required=True)
    ap.add_argument("--allow-create", choices=["0", "1"], default="0")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    observed = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if observed != args.analysis_source_git_commit:
        raise SystemExit("private evidence target bootstrap requires exact analysis checkout")
    inventory_path = Path(args.account_inventory).resolve()
    inventory = load_json(inventory_path)
    if inventory.get("account_id") != args.account_id or inventory.get("analysis_source_git_commit") != observed:
        raise SystemExit("account inventory account/analysis-source mismatch")

    env = dict(os.environ)
    env["CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET"] = args.allow_create
    states = inventory.get("states") or {}
    results = {}
    locators = set()
    for experiment_id in sorted(states):
        locator = str(states[experiment_id].get("evidence_dataset_locator", "")).strip()
        if not locator:
            raise SystemExit(f"state lacks evidence_dataset_locator: {experiment_id}")
        if locator in locators:
            raise SystemExit(f"evidence dataset locator collision: {locator}")
        locators.add(locator)
        result = ensure_private_target(locator, env=env)
        if result.get("status") != "PASS" or result.get("authoritative_is_private") is not True:
            raise SystemExit(f"private evidence target is not ready/private: {experiment_id}")
        results[experiment_id] = {
            "dataset_slug": locator,
            "authenticated_username": result.get("authenticated_username"),
            "authoritative_is_private": True,
            "current_version_number": result.get("current_version_number"),
            "created_this_run": bool(result.get("created_this_run")),
            "creation_command_nonzero_but_target_settled": bool(
                result.get("creation_command_nonzero_but_target_settled")
            ),
        }

    manifest = {
        "schema_version": "1.0",
        "status": "PASS",
        "gate_kind": "track_a_posttraining_private_evidence_targets",
        "account_id": args.account_id,
        "analysis_source_git_commit": observed,
        "account_inventory_sha256": sha256_file(inventory_path),
        "target_count": len(results),
        "unique_target_count": len(locators),
        "created_target_count": sum(row["created_this_run"] for row in results.values()),
        "states": results,
        "all_targets_private": True,
        "all_targets_owner_bound": True,
        "scientific_metrics_opened": False,
        "v1_test_accessed": False,
        "external_surface_accessed": False,
    }
    manifest["gate_sha256"] = sha256_json(manifest)
    atomic_write_json(args.output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
