from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.hashing import sha256_file, sha256_json


def load_json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--account-id", choices=["K1", "K2", "K3"], required=True)
    ap.add_argument("--materialization-catalog", required=True)
    ap.add_argument("--placement-freeze", required=True)
    ap.add_argument("--campaign-lock", default="journal_extension/locks/track_a_posttraining_campaign_v1.json")
    ap.add_argument("--analysis-source-git-commit", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    observed = git_head(repo)
    if observed != args.analysis_source_git_commit:
        raise SystemExit("account inventory builder requires exact analysis checkout")

    catalog_path = Path(args.materialization_catalog).resolve()
    placement_path = Path(args.placement_freeze).resolve()
    campaign_path = Path(args.campaign_lock)
    if not campaign_path.is_absolute():
        campaign_path = repo / campaign_path
    catalog = load_json(catalog_path)
    placement = load_json(placement_path)
    campaign = load_json(campaign_path)
    if catalog.get("schema_version") != "1.0" or catalog.get("account_id") != args.account_id:
        raise SystemExit("materialization catalog schema/account mismatch")
    if catalog.get("analysis_source_git_commit") != observed:
        raise SystemExit("materialization catalog analysis-source mismatch")
    if placement.get("status") != "PASS" or placement.get("analysis_source_git_commit") != observed:
        raise SystemExit("placement freeze is not PASS for this analysis source")

    assigned = placement.get("account_queues", {}).get(args.account_id)
    if not isinstance(assigned, list):
        raise SystemExit("placement freeze lacks requested account queue")
    catalog_states = catalog.get("states") or {}
    states = {}
    for experiment_id in assigned:
        if experiment_id not in catalog_states:
            raise SystemExit(f"assigned state missing from materialization catalog: {experiment_id}")
        row = dict(catalog_states[experiment_id])
        assignment = placement["assignments"][experiment_id]
        owner = str((campaign.get("account_owners") or {}).get(args.account_id, "")).strip()
        template = str((campaign.get("evidence_durability") or {}).get("locator_template", "")).strip()
        if not owner or not template:
            raise SystemExit("campaign lock lacks deterministic evidence-dataset owner/template")
        experiment_slug = experiment_id.lower().replace("_", "-")
        expected_locator = template.format(
            owner=owner,
            experiment_slug=experiment_slug,
            analysis_sha12=observed[:12],
        )
        supplied_locator = str(row.get("evidence_dataset_locator", "")).strip()
        if supplied_locator and supplied_locator != expected_locator:
            raise SystemExit(
                f"materialization catalog evidence locator differs from frozen campaign rule: {experiment_id}"
            )
        row["evidence_dataset_locator"] = expected_locator
        if assignment["account_id"] != args.account_id:
            raise SystemExit(f"placement/account mismatch: {experiment_id}")
        row["role"] = row.get("role")
        row["analysis_account_id"] = args.account_id
        row["analysis_slot_id"] = assignment["slot_id"]
        states[experiment_id] = row

    common = catalog.get("common") or {}
    required_common = {"manifest", "class_map", "image_root", "g1a_bundle"}
    if not required_common.issubset(common):
        raise SystemExit(f"materialization catalog missing common keys: {sorted(required_common - set(common))}")

    output = {
        "schema_version": "1.0",
        "account_id": args.account_id,
        "analysis_source_git_commit": observed,
        "training_science_source_git_commit": "56023042e57758591df9babb3438f191dbe10312",
        "manifest": common["manifest"],
        "class_map": common["class_map"],
        "image_root": common["image_root"],
        "g1a_bundle": common["g1a_bundle"],
        "states": states,
        "materialization_catalog_sha256": sha256_file(catalog_path),
        "placement_freeze_sha256": sha256_file(placement_path),
        "campaign_lock_sha256": sha256_file(campaign_path),
    }
    output["inventory_sha256"] = sha256_json(output)
    atomic_write_json(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
