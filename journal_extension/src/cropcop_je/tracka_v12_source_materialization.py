from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Callable

from .checkpointing import read_index, verify_selected
from .hashing import sha256_file, sha256_json
from .persistence import build_store
from .persistence_v8 import build_store_v8
from .tracka_v12_historical import (
    HISTORICAL_TRACKA_CLOSURE_SPECS,
    validate_historical_closure_identity,
)
from .tracka_v12_recovery import validate_recovered_terminal_record

PROFILE_HISTORICAL_LEGACY = "historical_legacy_kaggle_v1"
PROFILE_CONTINUATION_V8 = "continuation_generation_aware_v8"


class SourceMaterializationError(RuntimeError):
    pass


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SourceMaterializationError(f"JSON object required: {path}")
    return payload


def selected_checkpoint_sha256(record: dict[str, Any]) -> str:
    value = (
        ((record.get("artifact_locators") or {}).get("selected_checkpoint") or {}).get("sha256")
        or (record.get("result_summary") or {}).get("selected_checkpoint_sha256")
    )
    value = str(value or "")
    if len(value) != 64:
        raise SourceMaterializationError("run record does not bind a 64-character selected checkpoint SHA")
    return value


def durable_locator(record: dict[str, Any]) -> str:
    value = (
        ((record.get("artifact_locators") or {}).get("selected_checkpoint") or {}).get("durable_locator")
        or (record.get("durable_store") or {}).get("locator")
        or ""
    )
    value = str(value).strip()
    if "/" not in value:
        raise SourceMaterializationError("run record does not bind a valid Kaggle durable locator")
    return value


def persistence_profile(record: dict[str, Any]) -> str:
    provenance = record.get("recovery_provenance") or {}
    if provenance.get("kind") == "cryptographic_terminal_record_recovery":
        errors = validate_recovered_terminal_record(record)
        if errors:
            raise SourceMaterializationError("recovered continuation record invalid: " + "; ".join(errors))
        return PROFILE_CONTINUATION_V8

    experiment_id = str(record.get("experiment_id", ""))
    if experiment_id in HISTORICAL_TRACKA_CLOSURE_SPECS:
        errors = validate_historical_closure_identity(record)
        if errors:
            raise SourceMaterializationError("historical run record invalid: " + "; ".join(errors))
        persistence = record.get("persistence_status") or {}
        if persistence.get("backend") != "kaggle_private_dataset":
            raise SourceMaterializationError("historical run record does not bind the legacy Kaggle backend")
        files = set(str(x) for x in (persistence.get("files") or []))
        if "checkpoint_index.json" not in files:
            raise SourceMaterializationError("historical persistence inventory lacks checkpoint_index.json")
        selected = selected_checkpoint_sha256(record)
        if not any(path.startswith("objects/selected.") and selected[:16] in path for path in files):
            raise SourceMaterializationError("historical persistence inventory does not bind the canonical selected object")
        return PROFILE_HISTORICAL_LEGACY

    raise SourceMaterializationError("run record is neither a frozen historical state nor a recovered continuation state")


def verify_private_read_access(
    locator: str,
    *,
    api_factory: Callable[[], object] | None = None,
) -> dict[str, Any]:
    if api_factory is None:
        from kaggle.api.kaggle_api_extended import KaggleApi

        def api_factory():
            api = KaggleApi()
            api.authenticate()
            return api

    api = api_factory()
    with tempfile.TemporaryDirectory() as td:
        metadata_path = Path(api.dataset_metadata(locator, td))
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = payload.get("info") or payload
    if metadata.get("isPrivate") is not True:
        raise SourceMaterializationError(f"source checkpoint dataset is not authoritatively private: {locator}")
    observed = str(metadata.get("id") or metadata.get("ref") or "")
    if observed and observed.casefold() != locator.casefold():
        raise SourceMaterializationError(f"source dataset metadata identity mismatch: {observed!r} != {locator!r}")
    return {
        "dataset_slug": locator,
        "authoritative_is_private": True,
        "metadata_visible": True,
    }


def shallow_verify_selected_checkpoint(
    checkpoint_root: str | Path,
    *,
    expected_sha256: str,
    expected_epoch: int | None = None,
) -> dict[str, Any]:
    root = Path(checkpoint_root).resolve()
    index_path = root / "checkpoint_index.json"
    if not index_path.is_file():
        raise SourceMaterializationError("checkpoint index is missing")
    index = read_index(root)
    selected = index.get("selected")
    if not isinstance(selected, dict):
        raise SourceMaterializationError("checkpoint index lacks selected checkpoint")
    if selected.get("sha256") != expected_sha256:
        raise SourceMaterializationError("checkpoint index selected SHA mismatch")
    if expected_epoch is not None and int(selected.get("epoch", -1)) != int(expected_epoch):
        raise SourceMaterializationError("checkpoint index selected epoch mismatch")
    rel = str(selected.get("relative_path", ""))
    path = root / rel
    if not rel or not path.is_file():
        raise SourceMaterializationError("selected checkpoint file is missing")
    if path.stat().st_size != int(selected.get("bytes", -1)):
        raise SourceMaterializationError("selected checkpoint byte-size mismatch")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise SourceMaterializationError("selected checkpoint file SHA mismatch")
    return {
        "checkpoint_index_sha256": sha256_file(index_path),
        "selected_checkpoint_sha256": expected_sha256,
        "selected_checkpoint_file_sha256": observed,
        "selected_checkpoint_epoch": int(selected.get("epoch", -1)),
        "selected_checkpoint_relative_path": rel,
        "payload_deserialized": False,
        "scientific_metrics_opened": False,
    }


def restore_for_historical_availability(
    *,
    record: dict[str, Any],
    checkpoint_root: str | Path,
    verify_private: bool = True,
    legacy_store_factory=build_store,
) -> dict[str, Any]:
    if persistence_profile(record) != PROFILE_HISTORICAL_LEGACY:
        raise SourceMaterializationError("historical availability restore refuses a non-historical persistence profile")
    locator = durable_locator(record)
    if verify_private:
        verify_private_read_access(locator)
    root = Path(checkpoint_root).resolve()
    restored = False
    if not (root / "checkpoint_index.json").is_file():
        store = legacy_store_factory("kaggle-dataset", locator)
        restored = store.restore(root, run_id=str(record["run_id"]))
        if restored is not True:
            raise SourceMaterializationError("legacy historical checkpoint restore returned no material")
    spec = HISTORICAL_TRACKA_CLOSURE_SPECS[str(record["experiment_id"])]
    evidence = shallow_verify_selected_checkpoint(
        root,
        expected_sha256=selected_checkpoint_sha256(record),
        expected_epoch=int(spec["selected_epoch"]),
    )
    return {
        "schema_version": "1.0",
        "status": "PASS",
        "profile": PROFILE_HISTORICAL_LEGACY,
        "experiment_id": record["experiment_id"],
        "run_id": record["run_id"],
        "dataset_locator": locator,
        "restore_performed": bool(restored),
        **evidence,
    }


def materialize_source_checkpoint(
    *,
    record: dict[str, Any],
    checkpoint_root: str | Path,
    verify_private: bool = True,
    legacy_store_factory=build_store,
    v8_store_factory=build_store_v8,
) -> dict[str, Any]:
    profile = persistence_profile(record)
    locator = durable_locator(record)
    if verify_private and profile == PROFILE_HISTORICAL_LEGACY:
        verify_private_read_access(locator)

    root = Path(checkpoint_root).resolve()
    restored = False
    if not (root / "checkpoint_index.json").is_file():
        if profile == PROFILE_HISTORICAL_LEGACY:
            store = legacy_store_factory("kaggle-dataset", locator)
        elif profile == PROFILE_CONTINUATION_V8:
            store = v8_store_factory("kaggle-dataset", locator)
        else:
            raise SourceMaterializationError(f"unsupported source persistence profile: {profile}")
        restored = store.restore(root, run_id=str(record["run_id"]))
        if restored is not True:
            raise SourceMaterializationError("source checkpoint restore returned no material")

    from .train import _identity as checkpoint_identity

    expected = selected_checkpoint_sha256(record)
    checkpoint_path, payload = verify_selected(
        root,
        expected_identity=checkpoint_identity(record),
        expected_sha256=expected,
    )
    result = {
        "schema_version": "1.0",
        "status": "PASS",
        "materialization_kind": "track_a_source_checkpoint",
        "profile": profile,
        "experiment_id": record["experiment_id"],
        "run_id": record["run_id"],
        "scientific_source_git_commit": record.get("source_git_commit"),
        "dataset_locator": locator,
        "restore_performed": bool(restored),
        "checkpoint_index_sha256": sha256_file(root / "checkpoint_index.json"),
        "selected_checkpoint_sha256": expected,
        "selected_checkpoint_file_sha256": sha256_file(checkpoint_path),
        "selected_checkpoint_epoch": int(payload["epoch"]),
        "payload_deserialized": True,
        "training_performed": False,
        "optimizer_state_advanced": False,
        "v1_test_accessed": False,
        "external_surface_accessed": False,
    }
    result["materialization_sha256"] = sha256_json(result)
    return result
