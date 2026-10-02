from __future__ import annotations

from pathlib import Path
from typing import Any

from .surfaces import validate_training_config

AUTHORITY_ID = "EAAI-JE-SDL-v2.1-QA"
MANIFEST_SHA256 = "bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2"
CLASS_MAP_SHA256 = "46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2"
TEACHER_SHA256 = "74b4701b8931976c9227845ead50788ae47e3596f575f2817b7352a715f53b79"
MNV4_MODEL_NAME = "mobilenetv4_conv_medium.e500_r256_in1k"
R13_MODEL_ID = "vit_dlittle_patch16_reg1_gap_256.sbb_nadamuon_in1k"
TORCHVISION_VERSION = "0.27.1"
TIMM_VERSION = "1.0.26"
SEEDS = {"S1": 21270083, "S2": 606135704, "S3": 1153870846}
PAIR_IDS = {"S1": "MNV4-PAIR-S1", "S2": "MNV4-PAIR-S2", "S3": "MNV4-PAIR-S3"}
FORBIDDEN_SURFACES = {"DS-V1-TEST-CONSUMED", "DS-EXT-*-SEALED", "DS-HIST-COMPARE"}


def _spec(*, family: str, seed_label: str, model_family: str, condition: str, objective: dict[str, float], config_path: str, **extra: Any) -> dict[str, Any]:
    return {
        "family": family,
        "seed_label": seed_label,
        "seed": SEEDS[seed_label],
        "model_family": model_family,
        "condition": condition,
        "objective": objective,
        "config_path": config_path,
        **extra,
    }


EXPERIMENT_SPECS: dict[str, dict[str, Any]] = {
    "R06-EFFB0-CONTEXT-S2": _spec(family="R06", seed_label="S2", model_family="effb0", condition="direct", objective={"ce": 1.0, "kd": 0.0, "feature": 0.0}, config_path="journal_extension/configs/r06_effb0/s2.json", model_name="torchvision.models.efficientnet_b0", weight_enum="EfficientNet_B0_Weights.IMAGENET1K_V1", required_init="EFFB0_INIT_S2.json"),
    "R06-EFFB0-CONTEXT-S3": _spec(family="R06", seed_label="S3", model_family="effb0", condition="direct", objective={"ce": 1.0, "kd": 0.0, "feature": 0.0}, config_path="journal_extension/configs/r06_effb0/s3.json", model_name="torchvision.models.efficientnet_b0", weight_enum="EfficientNet_B0_Weights.IMAGENET1K_V1", required_init="EFFB0_INIT_S3.json"),
    "R07-CNXTT-CONTEXT-S2": _spec(family="R07", seed_label="S2", model_family="cnxtt", condition="direct", objective={"ce": 1.0, "kd": 0.0, "feature": 0.0}, config_path="journal_extension/configs/r07_cnxtt/s2.json", model_name="torchvision.models.convnext_tiny", weight_enum="ConvNeXt_Tiny_Weights.IMAGENET1K_V1", required_init="CNXTT_INIT_S2.json"),
    "R07-CNXTT-CONTEXT-S3": _spec(family="R07", seed_label="S3", model_family="cnxtt", condition="direct", objective={"ce": 1.0, "kd": 0.0, "feature": 0.0}, config_path="journal_extension/configs/r07_cnxtt/s3.json", model_name="torchvision.models.convnext_tiny", weight_enum="ConvNeXt_Tiny_Weights.IMAGENET1K_V1", required_init="CNXTT_INIT_S3.json"),
    "R12-MNV4-LOGITS-S2": _spec(family="R12_LOGITS", seed_label="S2", model_family="mnv4", condition="teacher", objective={"ce": 0.65, "kd": 0.35, "feature": 0.0}, config_path="journal_extension/configs/r12_logits/s2.json", model_name=MNV4_MODEL_NAME, pair_id=PAIR_IDS["S2"], required_init="PAIR_INIT_S2.json"),
    "R12-MNV4-LOGITS-S3": _spec(family="R12_LOGITS", seed_label="S3", model_family="mnv4", condition="teacher", objective={"ce": 0.65, "kd": 0.35, "feature": 0.0}, config_path="journal_extension/configs/r12_logits/s3.json", model_name=MNV4_MODEL_NAME, pair_id=PAIR_IDS["S3"], required_init="PAIR_INIT_S3.json"),
    "R12-MNV4-FEATURE-S2": _spec(family="R12_FEATURE", seed_label="S2", model_family="mnv4", condition="teacher", objective={"ce": 0.85, "kd": 0.0, "feature": 0.15}, config_path="journal_extension/configs/r12_feature/s2.json", model_name=MNV4_MODEL_NAME, pair_id=PAIR_IDS["S2"], required_init="PAIR_INIT_S2.json"),
    "R12-MNV4-FEATURE-S3": _spec(family="R12_FEATURE", seed_label="S3", model_family="mnv4", condition="teacher", objective={"ce": 0.85, "kd": 0.0, "feature": 0.15}, config_path="journal_extension/configs/r12_feature/s3.json", model_name=MNV4_MODEL_NAME, pair_id=PAIR_IDS["S3"], required_init="PAIR_INIT_S3.json"),
    "R13-VIT-DLITTLE-DIFF-CONTEXT-S1": _spec(family="R13", seed_label="S1", model_family="vit_dlittle_diff", condition="direct", objective={"ce": 1.0, "kd": 0.0, "feature": 0.0}, config_path="journal_extension/configs/r13_vit_dlittle/s1.json", model_id=R13_MODEL_ID, required_init="R13_VIT_DLITTLE_INIT_S1.json"),
    "R13-VIT-DLITTLE-DIFF-CONTEXT-S2": _spec(family="R13", seed_label="S2", model_family="vit_dlittle_diff", condition="direct", objective={"ce": 1.0, "kd": 0.0, "feature": 0.0}, config_path="journal_extension/configs/r13_vit_dlittle/s2.json", model_id=R13_MODEL_ID, required_init="R13_VIT_DLITTLE_INIT_S2.json"),
    "R13-VIT-DLITTLE-DIFF-CONTEXT-S3": _spec(family="R13", seed_label="S3", model_family="vit_dlittle_diff", condition="direct", objective={"ce": 1.0, "kd": 0.0, "feature": 0.0}, config_path="journal_extension/configs/r13_vit_dlittle/s3.json", model_id=R13_MODEL_ID, required_init="R13_VIT_DLITTLE_INIT_S3.json"),
}


def experiment_config_path(experiment_id: str) -> str:
    try:
        return str(EXPERIMENT_SPECS[experiment_id]["config_path"])
    except KeyError as exc:
        raise ValueError(f"experiment is not in the frozen Track-A v1.2 continuation set: {experiment_id}") from exc


def validate_tracka_v12_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        validate_training_config(config)
    except Exception as exc:
        return [str(exc)]

    experiment_id = str(config.get("experiment_id", ""))
    spec = EXPERIMENT_SPECS.get(experiment_id)
    if spec is None:
        return [f"experiment is not in the frozen Track-A v1.2 continuation set: {experiment_id}"]

    fixed = {
        "authority_id": AUTHORITY_ID,
        "seed": spec["seed"],
        "num_classes": 120,
        "model_family": spec["model_family"],
        "condition": spec["condition"],
        "objective": spec["objective"],
        "ctc_config": "journal_extension/configs/common/ctc_v2.json",
        "train_surface": "DS-V1-TRAIN",
        "validation_surface": "DS-V1-VAL",
        "manifest_sha256": MANIFEST_SHA256,
        "class_map_sha256": CLASS_MAP_SHA256,
    }
    for field, expected in fixed.items():
        if config.get(field) != expected:
            errors.append(f"Track-A v1.2 config drift for {experiment_id}: {field}")

    forbidden = set(config.get("forbidden_surfaces", []))
    if forbidden != FORBIDDEN_SURFACES:
        errors.append(f"Track-A v1.2 forbidden-surface policy drift for {experiment_id}")

    if spec["model_family"] in {"effb0", "cnxtt"}:
        for field, expected in {
            "model_name": spec["model_name"],
            "torchvision_version": TORCHVISION_VERSION,
            "weight_enum": spec["weight_enum"],
            "required_model_init_evidence": spec["required_init"],
        }.items():
            if config.get(field) != expected:
                errors.append(f"Track-A v1.2 config drift for {experiment_id}: {field}")
    elif spec["model_family"] == "mnv4":
        for field, expected in {
            "model_name": MNV4_MODEL_NAME,
            "pair_id": spec["pair_id"],
            "required_student_init_evidence": spec["required_init"],
            "teacher_checkpoint_sha256": TEACHER_SHA256,
            "required_teacher_evidence": "DINO_TEACHER.json",
        }.items():
            if config.get(field) != expected:
                errors.append(f"Track-A v1.2 config drift for {experiment_id}: {field}")
    elif spec["model_family"] == "vit_dlittle_diff":
        for field, expected in {
            "model_id": R13_MODEL_ID,
            "timm_version": TIMM_VERSION,
            "pretraining_dataset": "ImageNet-1K",
            "required_model_init_evidence": spec["required_init"],
            "required_pretrained_contract": "r13_pretrained_identity_and_normalization_contract_v1_2.json",
            "required_xai_interface_qualification": "r13_xai_interface_qualification_v1_2_1.json",
        }.items():
            if config.get(field) != expected:
                errors.append(f"Track-A v1.2 config drift for {experiment_id}: {field}")

    return errors


def validate_materialized_configs(repo_root: str | Path) -> list[str]:
    import json

    root = Path(repo_root)
    errors: list[str] = []
    for experiment_id, spec in EXPERIMENT_SPECS.items():
        path = root / str(spec["config_path"])
        if not path.is_file():
            errors.append(f"missing Track-A v1.2 config: {path.relative_to(root)}")
            continue
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"invalid Track-A v1.2 config JSON {path.relative_to(root)}: {exc}")
            continue
        errors.extend(validate_tracka_v12_config(config))
    return errors
