from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hashing import sha256_json
from .secondary import (
    create_empty_baseline,
    validate_secondary_config,
    validate_secondary_g1_bundle,
)
from .tracka_v12 import AUTHORITY_ID, CLASS_MAP_SHA256, MANIFEST_SHA256

HISTORICAL_PRINCIPAL_SOURCE_SHA = "f171309fc7e9dc22241ecc137ebbb8e4bcdc5433"
HISTORICAL_SECONDARY_SOURCE_SHA = "8904b100d223e4319776199c87ab397db23600ce"
HISTORICAL_SECONDARY_G1_SHA256 = "a723d511e4cb590924942496025962c854baaf3c254c9893187ab845afb30966"
HISTORICAL_SECONDARY_G2_SHA256 = "850d1d3cb0545937446a9853f786c5ecb41cb082facba45cba93e32007ed9e09"
HISTORICAL_SECONDARY_DEPENDENCY_LOCK_SHA256 = "6ea5fb51a0cc39c7940e4aaeb136d0f05214f5c558edd3c47b4cb61c6b516f37"
HISTORICAL_CTC_V2_SHA256 = "ed2331e63455e35ade00b548217fe987a9504f26a41e6c5d8a794e38741550bb"
HISTORICAL_SECONDARY_SOFTWARE_STACK_SHA256 = "d54153fcdb7887941aeaec6978335e737a2cd630326a2604d4c25b7a2ead63b0"

# Canonical selected-checkpoint lineage for every completed Track-A scientific state.
# Values are copied from the immutable Wave-1/Wave-2 validation closure artifacts.
HISTORICAL_TRACKA_CLOSURE_SPECS: dict[str, dict[str, Any]] = {
    "R04-MNV4-DIRECT-S1": {
        "wave": "WAVE1",
        "source_git_commit": HISTORICAL_PRINCIPAL_SOURCE_SHA,
        "run_id": "JE-R04-MNV4-DIRECT-S1-f171309fc7e9-A01",
        "selected_epoch": 28,
        "selected_checkpoint_sha256": "63a5ba04a278fcc5a0a333bfc38a21ab4718df5530c9b4f99bc81ef13011251c",
    },
    "R05-MNV4-TEACHER-S1": {
        "wave": "WAVE1",
        "source_git_commit": HISTORICAL_PRINCIPAL_SOURCE_SHA,
        "run_id": "JE-R05-MNV4-TEACHER-S1-f171309fc7e9-A01",
        "selected_epoch": 29,
        "selected_checkpoint_sha256": "331b4eb79bd7d02fb1f2887918ff5266e3c600814d51533a6730ab8b8ce7f5fa",
    },
    "R04-MNV4-DIRECT-S2": {
        "wave": "WAVE1",
        "source_git_commit": HISTORICAL_PRINCIPAL_SOURCE_SHA,
        "run_id": "JE-R04-MNV4-DIRECT-S2-f171309fc7e9-A01",
        "selected_epoch": 25,
        "selected_checkpoint_sha256": "66a3e5f4a90d363c3f6b4ea342ae804af666414851f874238fd3ad83397cd32a",
    },
    "R05-MNV4-TEACHER-S2": {
        "wave": "WAVE1",
        "source_git_commit": HISTORICAL_PRINCIPAL_SOURCE_SHA,
        "run_id": "JE-R05-MNV4-TEACHER-S2-f171309fc7e9-A01",
        "selected_epoch": 30,
        "selected_checkpoint_sha256": "157375477b04cf36095f8b66dff557760c44839373c77691738bca7e9de8468c",
    },
    "R04-MNV4-DIRECT-S3": {
        "wave": "WAVE1",
        "source_git_commit": HISTORICAL_PRINCIPAL_SOURCE_SHA,
        "run_id": "JE-R04-MNV4-DIRECT-S3-f171309fc7e9-A01",
        "selected_epoch": 28,
        "selected_checkpoint_sha256": "fa200d6add9c25856b1bdaa4d05961a0d1e4050e1204ab2a3103474b9c4a8ed2",
    },
    "R05-MNV4-TEACHER-S3": {
        "wave": "WAVE1",
        "source_git_commit": HISTORICAL_PRINCIPAL_SOURCE_SHA,
        "run_id": "JE-R05-MNV4-TEACHER-S3-f171309fc7e9-A01",
        "selected_epoch": 28,
        "selected_checkpoint_sha256": "0df653f67fe6a0f6a3c4d47fd8746ab563ef92d7182e36598602eba24b518d56",
    },
    "R12-MNV4-LOGITS-S1": {
        "wave": "WAVE2",
        "source_git_commit": HISTORICAL_SECONDARY_SOURCE_SHA,
        "run_id": "JE-R12-MNV4-LOGITS-S1-8904b100d223-A01",
        "selected_epoch": 30,
        "selected_checkpoint_sha256": "2377007e4f9ccc35c31ff9583ac6196e183bb03b1aa45ed0b8f58779abbc1827",
    },
    "R12-MNV4-FEATURE-S1": {
        "wave": "WAVE2",
        "source_git_commit": HISTORICAL_SECONDARY_SOURCE_SHA,
        "run_id": "JE-R12-MNV4-FEATURE-S1-8904b100d223-A01",
        "selected_epoch": 30,
        "selected_checkpoint_sha256": "e5d3b691e9f90602a2785dcdbe4c1c40e482a56c7f72741c708504377eea8950",
    },
    "R06-EFFB0-CONTEXT-S1": {
        "wave": "WAVE2",
        "source_git_commit": HISTORICAL_SECONDARY_SOURCE_SHA,
        "run_id": "JE-R06-EFFB0-CONTEXT-S1-8904b100d223-A01",
        "selected_epoch": 26,
        "selected_checkpoint_sha256": "882e1e45e1d18a8ed8168aab266ce5a66a9b57a74e2a8d98acf692207d2210c7",
    },
    "R07-CNXTT-CONTEXT-S1": {
        "wave": "WAVE2",
        "source_git_commit": HISTORICAL_SECONDARY_SOURCE_SHA,
        "run_id": "JE-R07-CNXTT-CONTEXT-S1-8904b100d223-A01",
        "selected_epoch": 27,
        "selected_checkpoint_sha256": "dc7fea2e8db91bf1fc023cb5e792b23b67659edec22e10a7c1d46b4010db3974",
    },
}

HISTORICAL_SECONDARY_DIRECT_SPECS: dict[str, dict[str, str]] = {
    "R06-EFFB0-CONTEXT-S1": {
        "model_key": "effb0",
        "config_path": "journal_extension/configs/r06_effb0/s1.json",
        "config_sha256": "38b852e816699979df97f4b9d40d5ec8326a31fa14ecb1548388a15fbe297f40",
        "run_id": "JE-R06-EFFB0-CONTEXT-S1-8904b100d223-A01",
        "student_init_sha256": "78896f3e1130358b1c98d90ed1f39dc058bb67261b905f3b64181344a6c6944f",
        "pretrained_sha256": "7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934",
        "selected_checkpoint_sha256": "882e1e45e1d18a8ed8168aab266ce5a66a9b57a74e2a8d98acf692207d2210c7",
    },
    "R07-CNXTT-CONTEXT-S1": {
        "model_key": "cnxtt",
        "config_path": "journal_extension/configs/r07_cnxtt/s1.json",
        "config_sha256": "a70cc70b11074fae07e5a6116bbb80e850a8011cce76bc52fc4406758abcb8d5",
        "run_id": "JE-R07-CNXTT-CONTEXT-S1-8904b100d223-A01",
        "student_init_sha256": "79a8666b099293abe6c4b7116041234d5759f2cb387ef8526adcc16e3c71acbf",
        "pretrained_sha256": "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d",
        "selected_checkpoint_sha256": "dc7fea2e8db91bf1fc023cb5e792b23b67659edec22e10a7c1d46b4010db3974",
    },
}


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_historical_closure_identity(run_record: dict[str, Any]) -> list[str]:
    """Bind a completed historical run record to the immutable Wave-1/Wave-2 selected checkpoint."""
    errors: list[str] = []
    experiment_id = str(run_record.get("experiment_id", ""))
    spec = HISTORICAL_TRACKA_CLOSURE_SPECS.get(experiment_id)
    if spec is None:
        return [f"not a frozen completed historical Track-A state: {experiment_id}"]
    if run_record.get("status") != "PASS":
        errors.append("historical run record is not terminal PASS")
    for field in ("run_id", "source_git_commit"):
        if run_record.get(field) != spec[field]:
            errors.append(f"historical {field} mismatch")
    if run_record.get("authority_id") not in {None, AUTHORITY_ID}:
        errors.append("historical authority mismatch")
    if run_record.get("manifest_sha256") not in {None, MANIFEST_SHA256}:
        errors.append("historical manifest identity mismatch")
    if run_record.get("class_map_sha256") not in {None, CLASS_MAP_SHA256}:
        errors.append("historical class-map identity mismatch")
    if run_record.get("v1_test_accessed") not in {None, False}:
        errors.append("historical run record indicates V1-test access")
    if run_record.get("protected_external_surface_accessed") not in {None, False}:
        errors.append("historical run record indicates protected external-surface access")
    if run_record.get("continuation_required") not in {None, False}:
        errors.append("historical run is not terminal")

    expected_selected = str(spec["selected_checkpoint_sha256"])
    selected_artifact = (run_record.get("artifact_locators") or {}).get("selected_checkpoint") or {}
    selected_result = (run_record.get("result_summary") or {}).get("selected_checkpoint_sha256")
    observed = {str(value) for value in (selected_artifact.get("sha256"), selected_result) if value}
    if not observed or observed != {expected_selected}:
        errors.append("historical selected checkpoint differs from immutable Wave closure")
    selected_epoch = (run_record.get("result_summary") or {}).get("selected_epoch")
    if selected_epoch is not None and int(selected_epoch) != int(spec["selected_epoch"]):
        errors.append("historical selected epoch differs from immutable Wave closure")
    return errors


def validate_historical_secondary_direct_identity(
    run_record: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    errors = validate_historical_closure_identity(run_record)
    experiment_id = str(run_record.get("experiment_id", ""))
    spec = HISTORICAL_SECONDARY_DIRECT_SPECS.get(experiment_id)
    if spec is None:
        return [f"not a frozen historical secondary direct state: {experiment_id}"]

    config_errors = validate_secondary_config(config)
    errors.extend(f"config:{error}" for error in config_errors)
    if config.get("experiment_id") != experiment_id or config.get("condition") != "direct":
        errors.append("config experiment/condition mismatch")
    config_sha = sha256_json(config)
    if config_sha != spec["config_sha256"] or run_record.get("config_sha256") != spec["config_sha256"]:
        errors.append("config SHA mismatch")

    fixed_record = {
        "run_id": spec["run_id"],
        "status": "PASS",
        "mode": "scientific",
        "secondary_track": True,
        "authority_id": AUTHORITY_ID,
        "source_git_commit": HISTORICAL_SECONDARY_SOURCE_SHA,
        "manifest_sha256": MANIFEST_SHA256,
        "class_map_sha256": CLASS_MAP_SHA256,
        "ctc_v2_sha256": HISTORICAL_CTC_V2_SHA256,
        "dependency_lock_sha256": HISTORICAL_SECONDARY_DEPENDENCY_LOCK_SHA256,
        "software_stack_sha256": HISTORICAL_SECONDARY_SOFTWARE_STACK_SHA256,
        "g1_seal_sha256": HISTORICAL_SECONDARY_G1_SHA256,
        "g2_barrier_sha256": HISTORICAL_SECONDARY_G2_SHA256,
        "student_init_sha256": spec["student_init_sha256"],
        "pretrained_sha256": spec["pretrained_sha256"],
        "seed": 21270083,
        "entrypoint": "run_secondary_training.py",
    }
    for field, expected in fixed_record.items():
        if run_record.get(field) != expected:
            errors.append(f"run record {field} mismatch")

    if run_record.get("allowed_surfaces") != ["DS-V1-TRAIN", "DS-V1-VAL"]:
        errors.append("historical allowed-surface inventory mismatch")
    selected_artifact = (run_record.get("artifact_locators") or {}).get("selected_checkpoint") or {}
    if selected_artifact.get("public_git") is not False:
        errors.append("historical checkpoint unexpectedly marked public")
    return errors


def load_historical_secondary_direct_model(
    *,
    repo_root: str | Path,
    run_record: dict[str, Any],
    secondary_g1_bundle: str | Path | None = None,
):
    repo = Path(repo_root).resolve()
    experiment_id = str(run_record.get("experiment_id", ""))
    spec = HISTORICAL_SECONDARY_DIRECT_SPECS.get(experiment_id)
    if spec is None:
        raise RuntimeError(f"unsupported historical secondary direct state: {experiment_id}")
    config = load_json(repo / spec["config_path"])
    errors = validate_historical_secondary_direct_identity(run_record, config)
    if errors:
        raise RuntimeError("historical secondary direct identity invalid: " + "; ".join(errors))

    if secondary_g1_bundle:
        seal, seal_errors = validate_secondary_g1_bundle(secondary_g1_bundle)
        if seal_errors:
            raise RuntimeError("historical secondary G1 bundle invalid: " + "; ".join(seal_errors))
        if seal.get("source_git_sha") != HISTORICAL_SECONDARY_SOURCE_SHA:
            raise RuntimeError("historical secondary G1 source SHA mismatch")
        if seal.get("secondary_g1_seal_sha256") != HISTORICAL_SECONDARY_G1_SHA256:
            raise RuntimeError("historical secondary G1 seal SHA mismatch")
        row = (seal.get("baselines") or {}).get(spec["model_key"], {})
        if row.get("init_sha256") != spec["student_init_sha256"]:
            raise RuntimeError("historical secondary G1 init SHA differs from canonical run")
        if row.get("pretrained_sha256") != spec["pretrained_sha256"]:
            raise RuntimeError("historical secondary G1 pretrained SHA differs from canonical run")
        if row.get("authorized_consumers") != [experiment_id]:
            raise RuntimeError("historical secondary G1 consumer authorization mismatch")

    # Post-training replay needs only the frozen architecture shell. The full
    # scientific student state is loaded from the hash/identity-verified selected
    # checkpoint immediately afterwards. No initialization is regenerated or used.
    model = create_empty_baseline(model_key=spec["model_key"], num_classes=120)
    return model, config
