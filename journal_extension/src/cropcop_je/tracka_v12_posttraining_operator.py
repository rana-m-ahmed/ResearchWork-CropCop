from __future__ import annotations

from pathlib import Path
from typing import Any

from .tracka_v12 import EXPERIMENT_SPECS

DIRECT_ALLOWED = {
    "manifest", "class_map", "image_root", "g1a_bundle", "secondary_g1_bundle",
    "principal_config", "principal_pair_init", "principal_pair_evidence",
    "row_id_column", "path_column", "split_column", "label_column", "class_index_column",
    "train_split_value", "val_split_value", "batch_size", "num_workers",
}
XAI_ALLOWED = DIRECT_ALLOWED - {"batch_size", "num_workers"}
AUX_ALLOWED = {
    "config", "manifest", "class_map", "image_root", "g1a_bundle",
    "principal_pair_init", "principal_pair_evidence",
    "row_id_column", "path_column", "split_column", "label_column", "class_index_column",
    "train_split_value", "val_split_value", "batch_size", "num_workers",
}
STAGE_ALLOWED = {"direct": DIRECT_ALLOWED, "xai": XAI_ALLOWED, "auxiliary": AUX_ALLOWED}

FILE_ARGS = {
    "manifest", "class_map", "principal_config", "principal_pair_init",
    "principal_pair_evidence", "config",
}
DIR_ARGS = {"image_root", "g1a_bundle", "secondary_g1_bundle"}


def role_for_experiment(experiment_id: str) -> str:
    if experiment_id.startswith(("R04-", "R06-", "R07-", "R13-")):
        return "direct"
    if experiment_id.startswith(("R05-", "R12-")):
        return "auxiliary"
    raise ValueError(f"unsupported Track-A experiment identity: {experiment_id}")


def required_stage_args(experiment_id: str, stage: str) -> set[str]:
    base = {"manifest", "class_map", "image_root"}
    if stage in {"direct", "xai"}:
        if experiment_id.startswith("R04-"):
            return base | {"principal_config", "principal_pair_init", "principal_pair_evidence"}
        if experiment_id in EXPERIMENT_SPECS:
            return base | {"g1a_bundle"}
        if experiment_id.startswith(("R06-", "R07-")):
            return base | {"secondary_g1_bundle"}
        raise ValueError(f"unsupported direct state: {experiment_id}")
    if stage == "auxiliary":
        required = base | {"config"}
        seed = experiment_id.rsplit("-", 1)[-1]
        if experiment_id.startswith("R12-") and seed in {"S2", "S3"}:
            return required | {"g1a_bundle"}
        return required | {"principal_pair_init", "principal_pair_evidence"}
    raise ValueError(f"unsupported post-training stage: {stage}")


def validate_state_operator_spec(
    experiment_id: str,
    spec: dict[str, Any],
    *,
    check_paths: bool = True,
) -> list[str]:
    errors: list[str] = []
    try:
        expected_role = role_for_experiment(experiment_id)
    except ValueError as exc:
        return [str(exc)]
    if spec.get("role") != expected_role:
        errors.append("role_mismatch")
    for field in ("run_record", "checkpoint_root", "evidence_dataset_locator"):
        if not str(spec.get(field, "")).strip():
            errors.append(f"missing:{field}")

    stage_map = spec.get("executor_args")
    if not isinstance(stage_map, dict):
        return errors + ["missing:executor_args"]
    expected_stages = {"direct", "xai"} if expected_role == "direct" else {"auxiliary"}
    if set(stage_map) != expected_stages:
        errors.append(f"executor_stage_inventory:{sorted(stage_map)}")
        return errors

    for stage in sorted(expected_stages):
        args = stage_map.get(stage)
        if not isinstance(args, dict):
            errors.append(f"{stage}:args_not_object")
            continue
        unknown = set(args) - STAGE_ALLOWED[stage]
        if unknown:
            errors.append(f"{stage}:unsupported_args:{sorted(unknown)}")
        missing = {
            name for name in required_stage_args(experiment_id, stage)
            if not str(args.get(name, "")).strip()
        }
        if missing:
            errors.append(f"{stage}:missing_required:{sorted(missing)}")
        if not check_paths:
            continue
        for name in sorted(FILE_ARGS & set(args)):
            value = str(args.get(name, "")).strip()
            if value and not Path(value).expanduser().resolve().is_file():
                errors.append(f"{stage}:file_missing:{name}")
        for name in sorted(DIR_ARGS & set(args)):
            value = str(args.get(name, "")).strip()
            if value and not Path(value).expanduser().resolve().is_dir():
                errors.append(f"{stage}:directory_missing:{name}")
    return errors


def cli_args(mapping: dict[str, Any], allowed: set[str]) -> list[str]:
    unknown = set(mapping) - allowed
    if unknown:
        raise RuntimeError(f"operator executor args contain unsupported keys: {sorted(unknown)}")
    result: list[str] = []
    for key in sorted(mapping):
        value = mapping[key]
        if value in (None, ""):
            continue
        result.extend(["--" + key.replace("_", "-"), str(value)])
    return result
