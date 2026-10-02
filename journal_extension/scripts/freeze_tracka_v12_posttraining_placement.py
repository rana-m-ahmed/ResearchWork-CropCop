from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.hashing import sha256_file, sha256_json

ACCOUNTS = ("K1", "K2", "K3")
SLOTS = {
    "K1": ("K1/GPU0", "K1/GPU1"),
    "K2": ("K2/GPU0", "K2/GPU1"),
    "K3": ("K3/GPU0", "K3/GPU1"),
}


def load_json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return payload


def git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def verify_hash(payload: dict, field: str) -> bool:
    observed = payload.get(field)
    if not isinstance(observed, str) or len(observed) != 64:
        return False
    clean = dict(payload)
    clean.pop(field, None)
    return observed == sha256_json(clean)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--campaign-lock", default="journal_extension/locks/track_a_posttraining_campaign_v1.json")
    ap.add_argument("--closure-authority", default="journal_extension/locks/track_a_posttraining_closure_authority_v1.json")
    ap.add_argument("--availability-report", action="append", required=True)
    ap.add_argument("--analysis-source-git-commit", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    observed = git_head(repo)
    if observed != args.analysis_source_git_commit:
        raise SystemExit(f"exact checkout mismatch: expected={args.analysis_source_git_commit}, observed={observed}")

    campaign_path = Path(args.campaign_lock)
    if not campaign_path.is_absolute():
        campaign_path = repo / campaign_path
    authority_path = Path(args.closure_authority)
    if not authority_path.is_absolute():
        authority_path = repo / authority_path
    campaign = load_json(campaign_path)
    authority = load_json(authority_path)
    if campaign.get("status") != "PRE_EXECUTION_LOCK":
        raise SystemExit("post-training campaign lock is not frozen PRE_EXECUTION_LOCK")
    if campaign.get("scientific_semantics_change_authorized") is not False:
        raise SystemExit("campaign unexpectedly authorizes scientific semantics change")

    reports = {}
    report_hashes = {}
    for raw in args.availability_report:
        path = Path(raw).resolve()
        report = load_json(path)
        account = str(report.get("account_id", ""))
        if account not in ACCOUNTS or account in reports:
            raise SystemExit(f"availability reports must contain exactly one unique K1/K2/K3 report; got {account!r}")
        if report.get("status") != "PASS" or report.get("gate_kind") != "track_a_v12_historical_artifact_availability":
            raise SystemExit(f"availability report is not PASS for {account}")
        if report.get("analysis_source_git_commit") != observed:
            raise SystemExit(f"availability report analysis-source mismatch for {account}")
        if not verify_hash(report, "gate_sha256"):
            raise SystemExit(f"availability report self-hash mismatch for {account}")
        for marker in ("scientific_metrics_opened", "robustness_results_opened", "xai_results_opened", "selection_outcomes_opened"):
            if report.get(marker) is not False:
                raise SystemExit(f"availability report opened forbidden pre-placement result surface: {account}:{marker}")
        reports[account] = report
        report_hashes[account] = sha256_file(path)
    if set(reports) != set(ACCOUNTS):
        raise SystemExit(f"exact K1/K2/K3 availability reports required, got {sorted(reports)}")

    authority_states = authority.get("state_inventory") or {}
    assignments: dict[str, dict] = {}
    fixed = campaign.get("continuation_account_binding") or {}
    for experiment_id, account in sorted(fixed.items()):
        if experiment_id not in authority_states:
            raise SystemExit(f"campaign continuation state absent from closure authority: {experiment_id}")
        if authority_states[experiment_id].get("terminal_account_id") != account:
            raise SystemExit(f"campaign/authority continuation account mismatch: {experiment_id}")
        assignments[experiment_id] = {
            "account_id": account,
            "placement_basis": "continuation_private_durability_owner",
        }

    preferences = campaign.get("historical_placement_preferences") or {}
    historical = sorted(set(authority_states) - set(fixed))
    if set(preferences) != set(historical):
        raise SystemExit("historical placement preference inventory does not match closure authority")
    for experiment_id in historical:
        chosen = None
        checked = []
        for account in preferences[experiment_id]:
            row = (reports[account].get("states") or {}).get(experiment_id)
            available = bool(row and row.get("available") is True)
            checked.append({"account_id": account, "available": available})
            if available and chosen is None:
                chosen = account
        if chosen is None:
            raise SystemExit(f"no account can verify canonical historical artifact state: {experiment_id}")
        assignments[experiment_id] = {
            "account_id": chosen,
            "placement_basis": "pre_metric_historical_artifact_availability",
            "preference_order": list(preferences[experiment_id]),
            "availability_checked": checked,
        }

    per_account = {account: [] for account in ACCOUNTS}
    for experiment_id in sorted(assignments):
        per_account[assignments[experiment_id]["account_id"]].append(experiment_id)
    for account, states in per_account.items():
        for index, experiment_id in enumerate(states):
            assignments[experiment_id]["slot_id"] = SLOTS[account][index % 2]
            assignments[experiment_id]["queue_index_on_account"] = index

    direct = {experiment_id for experiment_id, row in authority_states.items() if row.get("role") == "direct"}
    auxiliary = set(authority_states) - direct
    if len(assignments) != 21 or len(direct) != 12 or len(auxiliary) != 9:
        raise SystemExit("placement freeze inventory/count mismatch")

    result = {
        "schema_version": "1.0",
        "status": "PASS",
        "freeze_kind": "track_a_v12_posttraining_placement",
        "analysis_source_git_commit": observed,
        "campaign_lock_sha256": sha256_file(campaign_path),
        "closure_authority_sha256": sha256_file(authority_path),
        "availability_report_sha256": report_hashes,
        "assignment_count": len(assignments),
        "direct_state_count": len(direct),
        "auxiliary_state_count": len(auxiliary),
        "assignments": assignments,
        "account_queues": per_account,
        "placement_metric_blind": True,
        "scientific_metrics_opened_before_freeze": False,
        "robustness_results_opened_before_freeze": False,
        "xai_results_opened_before_freeze": False,
        "selection_outcomes_opened_before_freeze": False,
        "cross_gpu_gradient_synchronization": False,
        "placement_is_scientific_identity": False,
    }
    result["placement_freeze_sha256"] = sha256_json(result)
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
