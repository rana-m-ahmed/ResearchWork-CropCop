from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .tracka_v12 import EXPERIMENT_SPECS
from .tracka_v12_g2a import SLOT_ORDER
from .tracka_v12_g2a_v122 import validate_scheduler_freeze_v122

ACCOUNT_SLOTS = {
    "K1": ("K1/GPU0", "K1/GPU1"),
    "K2": ("K2/GPU0", "K2/GPU1"),
    "K3": ("K3/GPU0", "K3/GPU1"),
}
GIT_CREDENTIAL_ENV_NAMES = (
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "CROPCOP_GITHUB_TOKEN",
    "GIT_ASKPASS",
    "GIT_ASKPASS_REQUIRE",
    "SSH_AUTH_SOCK",
)
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


class TrackAV12OrchestrationError(RuntimeError):
    pass


def physical_gpu_index(slot_id: str) -> int:
    if slot_id not in SLOT_ORDER:
        raise TrackAV12OrchestrationError(f"unknown Track-A physical slot: {slot_id}")
    suffix = slot_id.rsplit("GPU", 1)[-1]
    index = int(suffix)
    if index not in {0, 1}:
        raise TrackAV12OrchestrationError(f"slot is not a dual-T4 child slot: {slot_id}")
    return index


def validate_durable_map(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(payload) != set(EXPERIMENT_SPECS):
        errors.append("durable-map experiment inventory must equal the exact 11 frozen states")
        return errors
    locators = []
    for experiment_id in sorted(EXPERIMENT_SPECS):
        value = payload.get(experiment_id)
        if not isinstance(value, str) or value.count("/") != 1:
            errors.append(f"durable locator must be an owner/dataset slug: {experiment_id}")
            continue
        owner, dataset = value.split("/", 1)
        if not owner or not dataset:
            errors.append(f"durable locator owner/dataset is empty: {experiment_id}")
            continue
        locators.append(value)
    if len(locators) != len(set(locators)):
        errors.append("each scientific state must have a distinct private durable dataset locator")
    return errors


def validate_account_queue(scheduler: dict[str, Any], account_id: str) -> list[str]:
    errors = list(validate_scheduler_freeze_v122(scheduler))
    if account_id not in ACCOUNT_SLOTS:
        return errors + [f"unknown Kaggle account lane: {account_id}"]
    queues = scheduler.get("static_slot_queues", {})
    for slot in ACCOUNT_SLOTS[account_id]:
        queue = queues.get(slot)
        if not isinstance(queue, list) or not queue:
            errors.append(f"frozen queue missing/empty for {slot}")
        elif any(experiment_id not in EXPERIMENT_SPECS for experiment_id in queue):
            errors.append(f"frozen queue contains unauthorized experiment for {slot}")
    return errors


def sanitized_child_environment(
    parent_env: dict[str, str],
    *,
    slot_id: str,
) -> dict[str, str]:
    env = dict(parent_env)
    for name in GIT_CREDENTIAL_ENV_NAMES:
        env.pop(name, None)
    env.update(THREAD_ENV)
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_index(slot_id))
    env["CROPCOP_PHYSICAL_SLOT_ID"] = slot_id
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def git_credentials_present(env: dict[str, str]) -> bool:
    return any(bool(env.get(name)) for name in GIT_CREDENTIAL_ENV_NAMES)


def scientific_run_id(experiment_id: str, source_git_commit: str) -> str:
    if experiment_id not in EXPERIMENT_SPECS:
        raise TrackAV12OrchestrationError(f"unauthorized Track-A experiment: {experiment_id}")
    if len(source_git_commit) != 40:
        raise TrackAV12OrchestrationError("source Git commit must be a full 40-character SHA")
    return f"JE-{experiment_id}-{source_git_commit[:12]}-A01"


def account_output_path(output_root: str | Path, *, account_id: str, experiment_id: str) -> Path:
    if account_id not in ACCOUNT_SLOTS or experiment_id not in EXPERIMENT_SPECS:
        raise TrackAV12OrchestrationError("invalid account/experiment output identity")
    return Path(output_root) / account_id / experiment_id


def account_queue_manifest(scheduler: dict[str, Any], account_id: str) -> dict[str, Any]:
    errors = validate_account_queue(scheduler, account_id)
    if errors:
        raise TrackAV12OrchestrationError("invalid account queue: " + "; ".join(errors))
    queues = scheduler["static_slot_queues"]
    return {
        "schema_version": "1.0",
        "account_id": account_id,
        "scheduler_freeze_sha256": scheduler["scheduler_freeze_sha256"],
        "slots": {
            slot: {
                "physical_gpu_index": physical_gpu_index(slot),
                "queue": list(queues[slot]),
            }
            for slot in ACCOUNT_SLOTS[account_id]
        },
        "scientific_results_may_change_queue": False,
    }


def public_evidence_candidates(run_output: str | Path) -> list[Path]:
    root = Path(run_output)
    candidates = [root / "run_record.json", root / "metrics.json", root / "segments.jsonl"]
    candidates.extend(sorted(root.glob("segment_*.json")))
    return [path for path in candidates if path.is_file()]
