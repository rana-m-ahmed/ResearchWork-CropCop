from __future__ import annotations

from .tracka_v12 import EXPERIMENT_SPECS

SLOT_IDS = ("K1/GPU0", "K1/GPU1", "K2/GPU0", "K2/GPU1", "K3/GPU0", "K3/GPU1")
LOGICAL_LANE_PREFIX = "TRACKA-V12"


def logical_lane_id(experiment_id: str) -> str:
    if experiment_id not in EXPERIMENT_SPECS:
        raise ValueError(f"unauthorized Track-A v1.2 experiment: {experiment_id}")
    return f"{LOGICAL_LANE_PREFIX}:{experiment_id}"


def checkpoint_identity_projection(run_identity: dict) -> dict:
    fields = (
        "experiment_id",
        "authority_id",
        "source_git_commit",
        "config_sha256",
        "ctc_v2_sha256",
        "manifest_sha256",
        "class_map_sha256",
        "seed",
        "student_init_sha256",
        "pretrained_sha256",
        "teacher_sha256",
        "teacher_factory_sha256",
        "teacher_factory_bundle_sha256",
        "software_stack_sha256",
        "dependency_lock_sha256",
        "g1_seal_sha256",
        "g2_barrier_sha256",
        "lane_id",
    )
    return {field: run_identity.get(field) for field in fields}
