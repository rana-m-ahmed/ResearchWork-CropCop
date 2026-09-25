from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROTOCOL_ID = "TRACKB_EAAI_EXTERNAL_VALIDATION_v2"
REQUIRED_ROLES = {"core", "historical_compare", "gvlid_v5", "irish_potato"}
CLASS_MAP_SHA256 = "46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2"
CHECKPOINTS = {
    "S1": "dc7fea2e8db91bf1fc023cb5e792b23b67659edec22e10a7c1d46b4010db3974",
    "S2": "199afb9f7043e599fbb2239fb3babcfb431324a3c3219250ae6dd0359d8bc310",
    "S3": "621c2e6cfecd23da21b4f17d2244bd068b5a3602ee5240f0dcc95ae4360beb37",
}


class TrackBEAAIError(RuntimeError):
    pass


@dataclass(frozen=True)
class Bundle:
    role: str
    root: Path
    manifest_path: Path
    manifest: dict[str, Any]


def sha256_file(path: str | Path, *, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    data = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TrackBEAAIError(f"JSON object required: {path}")
    return obj


def atomic_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _contains_forbidden_test_surface(obj: Any, key: str = "") -> bool:
    """Reject actual V1-test surfaces while allowing explicit access=false provenance."""
    if isinstance(obj, dict):
        for child_key, value in obj.items():
            child = str(child_key).lower()
            if child in {
                "v1_test_accessed",
                "v1_test_image_bytes_accessed",
                "consumed_v1_test_accessed",
            }:
                if value is not False:
                    return True
                continue
            if _contains_forbidden_test_surface(value, child):
                return True
        return False
    if isinstance(obj, list):
        return any(_contains_forbidden_test_surface(value, key) for value in obj)
    if isinstance(obj, str):
        value = obj.replace("\\", "/").lower()
        if value.strip().upper() == "DS-V1-TEST-CONSUMED":
            return True
        pathish = (
            key == "path"
            or key.endswith(("_path", "_root", "_dir"))
            or key in {"data_root", "image_root", "validation_root", "source_root"}
        )
        normalized = "/" + value.strip("/") + "/"
        if pathish and any(
            marker in normalized
            for marker in (
                "/v1_test/",
                "/v1-test/",
                "/test_consumed/",
                "/ds-v1-test-consumed/",
            )
        ):
            return True
    return False


def discover_bundles(input_root: str | Path) -> dict[str, Bundle]:
    root = Path(input_root).resolve()
    if not root.is_dir():
        raise TrackBEAAIError(f"input root not found: {root}")
    found: dict[str, Bundle] = {}
    for manifest_path in sorted(root.glob("**/TRACKB_INPUT_MANIFEST.json")):
        manifest = load_json(manifest_path)
        role = str(manifest.get("role", "")).strip()
        if not role:
            raise TrackBEAAIError(f"input manifest has no role: {manifest_path}")
        if role in found:
            raise TrackBEAAIError(
                f"duplicate role {role}: "
                f"{found[role].manifest_path} / {manifest_path}"
            )
        if _contains_forbidden_test_surface(manifest):
            raise TrackBEAAIError(
                f"forbidden V1-test surface referenced by role {role}"
            )
        found[role] = Bundle(
            role=role,
            root=manifest_path.parent.resolve(),
            manifest_path=manifest_path.resolve(),
            manifest=manifest,
        )
    missing = REQUIRED_ROLES - set(found)
    extra = set(found) - REQUIRED_ROLES
    if missing or extra:
        raise TrackBEAAIError(
            f"Track-B role mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    hist = found["historical_compare"].manifest
    expected = {
        "coverage_scope": "V1_TRAIN_VAL_ONLY",
        "image_count": 92744,
        "v1_test_image_bytes_accessed": False,
    }
    for key, value in expected.items():
        if hist.get(key) != value:
            raise TrackBEAAIError(
                f"historical comparison contract mismatch: "
                f"{key}={hist.get(key)!r}"
            )
    return found


def validate_materialization_pair(
    input_root: str | Path,
    bundles: dict[str, Bundle],
    expected_materialization_id: str,
) -> dict[str, Any]:
    root = Path(input_root).resolve()
    infra = sorted(root.glob("**/TRACKB_INFRASTRUCTURE_BUNDLE.json"))
    external = sorted(root.glob("**/TRACKB_EXTERNAL_BUNDLE.json"))
    if len(infra) != 1 or len(external) != 1:
        raise TrackBEAAIError(
            "expected exactly one infrastructure and one external bundle receipt"
        )
    infra_obj = load_json(infra[0])
    external_obj = load_json(external[0])
    for obj, role in (
        (infra_obj, "TRACKB_INFRASTRUCTURE"),
        (external_obj, "TRACKB_EXTERNAL"),
    ):
        if obj.get("status") != "PASS_PAIRED_TRACKB_INPUT_BUNDLE":
            raise TrackBEAAIError(f"{role} bundle receipt is not PASS")
        if obj.get("bundle_role") != role:
            raise TrackBEAAIError(f"{role} bundle receipt role mismatch")
        if str(obj.get("materialization_id", "")) != str(
            expected_materialization_id
        ):
            raise TrackBEAAIError(
                f"{role} materialization ID does not match the frozen protocol"
            )

    shared_fields = (
        "materialization_id",
        "repository_source_sha",
        "scientific_execution_lock_sha256",
        "scientific_code_attestation_sha256",
        "external_source_lock_sha256",
        "input_materialization_lock_sha256",
        "role_manifest_sha256",
    )
    for field in shared_fields:
        if infra_obj.get(field) != external_obj.get(field):
            raise TrackBEAAIError(
                f"paired bundle receipts disagree on {field}"
            )

    expected_manifest_hashes = infra_obj.get("role_manifest_sha256") or {}
    observed_manifest_hashes = {
        role: sha256_file(bundle.manifest_path)
        for role, bundle in bundles.items()
    }
    if observed_manifest_hashes != expected_manifest_hashes:
        raise TrackBEAAIError(
            "attached role manifests do not match the paired materialization receipt"
        )
    return {
        "materialization_id": str(infra_obj["materialization_id"]),
        "repository_source_sha": str(infra_obj["repository_source_sha"]),
        "role_manifest_sha256": observed_manifest_hashes,
        "infrastructure_receipt_sha256": sha256_file(infra[0]),
        "external_receipt_sha256": sha256_file(external[0]),
    }


def resolve_file(bundle: Bundle, key: str) -> Path:
    row = (bundle.manifest.get("files") or {}).get(key)
    if not isinstance(row, dict):
        raise TrackBEAAIError(f"{bundle.role} missing file key {key}")
    rel = str(row.get("path", "")).strip()
    expected = str(row.get("sha256", "")).lower()
    if not rel or len(expected) != 64:
        raise TrackBEAAIError(
            f"{bundle.role}/{key} has incomplete path/hash binding"
        )
    path = (bundle.root / rel).resolve()
    if bundle.root not in path.parents and path != bundle.root:
        raise TrackBEAAIError(f"{bundle.role}/{key} escapes bundle root")
    if not path.is_file():
        raise TrackBEAAIError(
            f"missing input file {bundle.role}/{key}: {path}"
        )
    actual = sha256_file(path)
    if actual != expected:
        raise TrackBEAAIError(
            f"input hash mismatch {bundle.role}/{key}: "
            f"expected={expected}, actual={actual}"
        )
    return path


def resolve_dir(bundle: Bundle, key: str) -> Path:
    rel = str(bundle.manifest.get(key, "")).strip()
    if not rel:
        raise TrackBEAAIError(
            f"{bundle.role} missing directory field {key}"
        )
    path = (bundle.root / rel).resolve()
    if bundle.root not in path.parents and path != bundle.root:
        raise TrackBEAAIError(f"{bundle.role}/{key} escapes bundle root")
    if not path.is_dir():
        raise TrackBEAAIError(
            f"missing directory {bundle.role}/{key}: {path}"
        )
    return path


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol = load_json(path)
    if (
        protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("status") != "FROZEN_PRE_RESULTS"
    ):
        raise TrackBEAAIError(
            "simplified EAAI protocol identity/status mismatch"
        )
    if protocol.get("checkpoints") != CHECKPOINTS:
        raise TrackBEAAIError("protocol checkpoint identities drifted")
    if protocol.get("leakage_surface", {}).get("v1_test_accessed") is not False:
        raise TrackBEAAIError("protocol does not preserve V1-test closure")
    policy = protocol.get("prediction_policy", {})
    if policy.get("native_output_classes") != 120:
        raise TrackBEAAIError("protocol native output-space drift")
    if policy.get("mapped_subset_logit_renormalization") is not False:
        raise TrackBEAAIError(
            "mapped-subset logit renormalization must remain disabled"
        )
    return protocol


def load_class_map(path: Path) -> dict[str, int]:
    obj = load_json(path)
    result = {str(key): int(value) for key, value in obj.items()}
    if len(result) != 120 or set(result.values()) != set(range(120)):
        raise TrackBEAAIError("class map is not a 120-way bijection")
    return result
