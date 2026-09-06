from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

from .hashing import require_sha256, sha256_file, sha256_json
from .tensor_identity import TENSOR_IDENTITY_ALGORITHM
from .smoke_handoff import validate_terminal_smoke_b_evidence

AUTHORITY_ID = "EAAI-JE-SDL-v2.1-QA"
AUTHORITY_SHA256 = "aab17b65b0873dcb1ecedb061eb02ff60ccb09f8b830184f5e2231a600278f74"
MANIFEST_SHA256 = "bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2"
CLASS_MAP_SHA256 = "46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2"
MNV4_MODEL_NAME = "mobilenetv4_conv_medium.e500_r256_in1k"
TIMM_VERSION = "1.0.26"
TEACHER_SHA256 = "74b4701b8931976c9227845ead50788ae47e3596f575f2817b7352a715f53b79"
PAIR_SPECS = {
    "S1": {"pair_id": "MNV4-PAIR-S1", "seed": 21270083, "consumers": ["R04-MNV4-DIRECT-S1", "R05-MNV4-TEACHER-S1"]},
    "S2": {"pair_id": "MNV4-PAIR-S2", "seed": 606135704, "consumers": ["R04-MNV4-DIRECT-S2", "R05-MNV4-TEACHER-S2"]},
    "S3": {"pair_id": "MNV4-PAIR-S3", "seed": 1153870846, "consumers": ["R04-MNV4-DIRECT-S3", "R05-MNV4-TEACHER-S3"]},
}


class G1Error(RuntimeError):
    pass


def _load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def hash_without_field(payload: dict[str, Any], field: str) -> str:
    clean = dict(payload)
    clean.pop(field, None)
    return sha256_json(clean)


def dependency_lock_hash(lock: dict[str, Any]) -> str:
    return hash_without_field(lock, "dependency_lock_sha256")


def validate_dependency_lock_object(lock: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected = dependency_lock_hash(lock)
    if lock.get("dependency_lock_sha256") != expected:
        errors.append("dependency lock self-hash mismatch")
    if lock.get("python") != "3.12.13":
        errors.append("dependency lock Python is not frozen 3.12.13")
    required = {
        "torch": "2.12.1",
        "torchvision": "0.27.1",
        "timm": "1.0.26",
        "numpy": "2.5.2",
        "Pillow": "12.3.0",
        "safetensors": "0.8.0",
        "kaggle": "2.2.4",
        "huggingface-hub": "1.30.0",
        "transformers": "5.0.0",
    }
    packages = lock.get("packages", {})
    for name, version in required.items():
        if packages.get(name) != version:
            errors.append(f"dependency lock mismatch for {name}: expected {version}, got {packages.get(name)}")
    return errors


def validate_dependency_environment(lock: dict[str, Any]) -> list[str]:
    errors = validate_dependency_lock_object(lock)
    if platform.python_version() != lock.get("python"):
        errors.append(f"Python drift: expected {lock.get('python')}, got {platform.python_version()}")
    for name, expected in lock.get("packages", {}).items():
        try:
            observed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            observed = None
        if observed != expected:
            errors.append(f"package drift {name}: expected {expected}, got {observed}")
    return errors



def validate_pretrained_provenance(record: dict[str, Any], *, artifact_path: str | Path | None = None) -> list[str]:
    errors: list[str] = []
    if record.get("model_name") != MNV4_MODEL_NAME:
        errors.append("MobileNetV4 provenance model identifier mismatch")
    if record.get("timm_version") != TIMM_VERSION:
        errors.append("MobileNetV4 provenance timm version mismatch")
    if record.get("verification") != "tensor_exact_match_against_timm_pretrained":
        errors.append("MobileNetV4 provenance lacks exact official-timm tensor match")
    if record.get("official_timm_tensor_match") is not True:
        errors.append("MobileNetV4 official-timm tensor match is not PASS")
    if record.get("source_kind") not in {"timm_pretrained_cfg_hf_hub", "timm_pretrained_cfg_url"}:
        errors.append("MobileNetV4 provenance source kind is not an official timm pretrained_cfg source")
    if not record.get("source_locator"):
        errors.append("MobileNetV4 provenance source locator missing")
    if record.get("tensor_identity_algorithm") != TENSOR_IDENTITY_ALGORITHM:
        errors.append("MobileNetV4 candidate tensor-identity algorithm mismatch")
    if record.get("official_tensor_identity_algorithm") != TENSOR_IDENTITY_ALGORITHM:
        errors.append("MobileNetV4 official tensor-identity algorithm mismatch")
    for field in (
        "artifact_sha256",
        "timm_pretrained_cfg_sha256",
        "tensor_identity_sha256",
        "official_tensor_identity_sha256",
    ):
        if len(str(record.get(field, ""))) != 64:
            errors.append(f"MobileNetV4 provenance {field} invalid")
    if record.get("tensor_identity_sha256") != record.get("official_tensor_identity_sha256"):
        errors.append("MobileNetV4 candidate tensor identity differs from official timm tensor identity")
    if record.get("candidate_serialization_format") not in {
        "safetensors_state_dict",
        "torch_state_dict",
    }:
        errors.append("MobileNetV4 candidate serialization format is not explicitly supported")
    if int(record.get("artifact_bytes", 0) or 0) <= 0:
        errors.append("MobileNetV4 provenance byte count invalid")
    if artifact_path is not None:
        p = Path(artifact_path)
        if not p.exists():
            errors.append("MobileNetV4 pretrained artifact missing")
        else:
            if sha256_file(p) != record.get("artifact_sha256"):
                errors.append("MobileNetV4 pretrained bytes differ from provenance")
            if p.stat().st_size != int(record.get("artifact_bytes", -1)):
                errors.append("MobileNetV4 pretrained byte count differs from provenance")
    return errors

def factory_bundle_hash(manifest: dict[str, Any]) -> str:
    canonical = {
        "schema_version": manifest.get("schema_version"),
        "entrypoint": manifest.get("entrypoint"),
        "output_order_transform": manifest.get("output_order_transform"),
        "files": sorted(manifest.get("files", []), key=lambda x: str(x.get("path", ""))),
    }
    return sha256_json(canonical)


def validate_teacher_factory_bundle(
    manifest: dict[str, Any],
    *,
    repo_root: str | Path | None = None,
    source_root: str | Path | None = None,
    expected_entrypoint: str | None = None,
) -> list[str]:
    errors: list[str] = []
    if expected_entrypoint and manifest.get("entrypoint") != expected_entrypoint:
        errors.append("teacher factory entrypoint differs from sealed bundle")
    if manifest.get("output_order_transform") != "none":
        errors.append("teacher factory bundle must declare no output-index reordering")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        errors.append("teacher factory bundle has no source files")
        return errors
    observed_paths = set()
    for row in files:
        rel = str(row.get("path", ""))
        if not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
            errors.append(f"invalid teacher factory relative source path: {rel!r}")
            continue
        if rel in observed_paths:
            errors.append(f"duplicate teacher factory source path: {rel}")
        observed_paths.add(rel)
        if len(str(row.get("sha256", ""))) != 64:
            errors.append(f"invalid teacher factory source SHA: {rel}")
        root = source_root if source_root is not None else repo_root
        if root is not None:
            path = Path(root) / rel
            if not path.exists():
                errors.append(f"teacher factory source file missing: {rel}")
            elif sha256_file(path) != row.get("sha256"):
                errors.append(f"teacher factory source drift: {rel}")
    expected_hash = factory_bundle_hash(manifest)
    if manifest.get("bundle_sha256") != expected_hash:
        errors.append("teacher factory bundle hash mismatch")
    return errors



def validate_teacher_class_order_evidence(
    evidence: dict[str, Any],
    *,
    class_map_sha256: str = CLASS_MAP_SHA256,
    factory_bundle_sha256: str | None = None,
) -> list[str]:
    errors: list[str] = []
    if evidence.get("status") != "PASS":
        errors.append("teacher class-order evidence is not PASS")
    if evidence.get("class_map_sha256") != class_map_sha256:
        errors.append("teacher class-order evidence does not bind frozen class map")
    if int(evidence.get("teacher_output_width", -1)) != 120:
        errors.append("teacher output width is not 120")
    if int(evidence.get("classifier_out_features", -1)) != 120:
        errors.append("historical teacher classifier is not 120-way")
    if int(evidence.get("classifier_in_features", -1)) != 768:
        errors.append("historical teacher classifier input width is not 768")
    if evidence.get("classifier_weight_key") != "head.weight":
        errors.append("historical teacher classifier key is not head.weight")
    if list(evidence.get("classifier_weight_shape", [])) != [120, 768]:
        errors.append("historical teacher classifier shape is not [120,768]")
    if evidence.get("historical_index_semantics") != "class_map_index":
        errors.append("teacher historical index semantics are not class_map_index")
    if evidence.get("factory_output_order_transform") != "none":
        errors.append("teacher factory class-order evidence permits output reordering")
    if evidence.get("not_inferred_from_shape_only") is not True:
        errors.append("teacher class order may not be inferred from shape alone")
    if factory_bundle_sha256 and evidence.get("teacher_factory_bundle_sha256") != factory_bundle_sha256:
        errors.append("teacher class-order evidence binds a different teacher factory bundle")
    sources = evidence.get("historical_evidence_sources")
    if not isinstance(sources, list) or not sources:
        errors.append("teacher class-order evidence lacks historical source evidence")
    else:
        for row in sources:
            if len(str(row.get("sha256", ""))) != 64 or not row.get("kind"):
                errors.append("teacher class-order historical evidence source is incomplete")
    return errors


def validate_teacher_canonical_state_evidence(
    evidence: dict[str, Any],
    *,
    factory_bundle_sha256: str | None = None,
) -> list[str]:
    errors: list[str] = []
    if evidence.get("status") != "PASS":
        errors.append("teacher canonical-state evidence is not PASS")
    if evidence.get("checkpoint_sha256") != TEACHER_SHA256:
        errors.append("teacher canonical-state evidence checkpoint SHA mismatch")
    if int(evidence.get("checkpoint_bytes", 0) or 0) <= 0:
        errors.append("teacher canonical-state evidence byte count invalid")
    if evidence.get("canonical_state") != "EMA":
        errors.append("teacher canonical state is not EMA")
    if evidence.get("ema_exact_complete_coverage") is not True:
        errors.append("teacher canonical state lacks exact complete EMA coverage")
    if evidence.get("canonical_floating_tensors_equal_ema") is not True:
        errors.append("teacher canonical floating tensors do not exactly equal EMA shadow")
    if evidence.get("finite_state") is not True:
        errors.append("teacher canonical state is not finite")
    if evidence.get("classifier_weight_key") != "head.weight":
        errors.append("teacher canonical classifier key is not head.weight")
    if list(evidence.get("classifier_shape", [])) != [120, 768]:
        errors.append("teacher canonical classifier shape is not [120,768]")
    if int(evidence.get("feature_dimension", -1)) != 768:
        errors.append("teacher canonical feature dimension is not 768")
    if evidence.get("manifest_sha256") != MANIFEST_SHA256:
        errors.append("teacher canonical evidence manifest SHA mismatch")
    if evidence.get("class_map_sha256") != CLASS_MAP_SHA256:
        errors.append("teacher canonical evidence class-map SHA mismatch")
    if evidence.get("output_order_transform") != "none":
        errors.append("teacher canonical evidence permits output reordering")
    if evidence.get("transformers_version") != "5.0.0":
        errors.append("teacher canonical evidence Transformers version mismatch")
    if evidence.get("scientific_metric_computed") is not False:
        errors.append("teacher canonical evidence unexpectedly reports a scientific metric")
    if evidence.get("protected_data_accessed") is not False:
        errors.append("teacher canonical evidence unexpectedly accessed protected data")
    for count_field in (
        "raw_model_state_key_count",
        "floating_state_key_count",
        "ema_shadow_key_count",
    ):
        if int(evidence.get(count_field, 0) or 0) <= 0:
            errors.append(f"teacher canonical evidence {count_field} invalid")
    if (
        int(evidence.get("floating_state_key_count", -1))
        != int(evidence.get("ema_shadow_key_count", -2))
    ):
        errors.append("teacher canonical evidence floating/EMA key counts differ")
    if factory_bundle_sha256 and evidence.get("factory_bundle_sha256") != factory_bundle_sha256:
        errors.append("teacher canonical evidence binds a different factory bundle")
    return errors


def g1_seal_hash(seal: dict[str, Any]) -> str:
    return hash_without_field(seal, "g1_seal_sha256")



def validate_g1_seal_object(seal: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if seal.get("schema_version") != "2.0":
        errors.append("unsupported G1 seal schema")
    if seal.get("authority", {}).get("id") != AUTHORITY_ID:
        errors.append("G1 authority ID mismatch")
    if seal.get("authority", {}).get("sha256") != AUTHORITY_SHA256:
        errors.append("G1 authority SHA mismatch")
    if seal.get("dataset", {}).get("manifest_sha256") != MANIFEST_SHA256:
        errors.append("G1 V1 manifest identity mismatch")
    if seal.get("dataset", {}).get("class_map_sha256") != CLASS_MAP_SHA256:
        errors.append("G1 class-map identity mismatch")
    if len(str(seal.get("dataset", {}).get("identity_evidence_sha256", ""))) != 64:
        errors.append("G1 frozen-V1 identity evidence SHA invalid")
    if seal.get("student", {}).get("model_name") != MNV4_MODEL_NAME:
        errors.append("G1 MobileNetV4 model identity mismatch")
    if seal.get("student", {}).get("timm_version") != TIMM_VERSION:
        errors.append("G1 timm identity mismatch")
    if len(str(seal.get("source_git_sha", ""))) != 40:
        errors.append("G1 source Git SHA invalid")
    if len(str(seal.get("dependency_lock_sha256", ""))) != 64:
        errors.append("G1 dependency lock SHA invalid")
    if len(str(seal.get("infra_smoke_evidence_sha256", ""))) != 64:
        errors.append("G1 infrastructure-smoke evidence SHA invalid")
    if len(str(seal.get("dual_gpu_smoke_evidence_sha256", ""))) != 64:
        errors.append("G1 dual-GPU-smoke evidence SHA invalid")

    pretrained = seal.get("student", {}).get("pretrained", {})
    if pretrained.get("tensor_identity_algorithm") != TENSOR_IDENTITY_ALGORITHM:
        errors.append("G1 MobileNetV4 tensor-identity algorithm mismatch")
    for field in ("sha256", "provenance_sha256", "tensor_identity_sha256"):
        if len(str(pretrained.get(field, ""))) != 64:
            errors.append(f"G1 MobileNetV4 pretrained {field} invalid")
    if int(pretrained.get("bytes", 0) or 0) <= 0:
        errors.append("G1 MobileNetV4 pretrained byte count invalid")
    if pretrained.get("source_kind") not in {"timm_pretrained_cfg_hf_hub", "timm_pretrained_cfg_url"}:
        errors.append("G1 MobileNetV4 provenance source kind invalid")
    if not pretrained.get("source_locator"):
        errors.append("G1 MobileNetV4 provenance source locator missing")
    if pretrained.get("candidate_serialization_format") not in {
        "safetensors_state_dict",
        "torch_state_dict",
    }:
        errors.append("G1 MobileNetV4 serialization format invalid")

    pairs = seal.get("pair_initializations", {})
    for key, spec in PAIR_SPECS.items():
        row = pairs.get(key, {})
        if row.get("pair_id") != spec["pair_id"] or int(row.get("seed", -1)) != spec["seed"]:
            errors.append(f"G1 {key} pair identity mismatch")
        if set(row.get("authorized_consumers", [])) != set(spec["consumers"]):
            errors.append(f"G1 {key} authorized consumer mismatch")
        if row.get("pretrained_sha256") != pretrained.get("sha256"):
            errors.append(f"G1 {key} does not bind the global pretrained SHA")
        if len(str(row.get("sha256", ""))) != 64 or int(row.get("bytes", 0) or 0) <= 0:
            errors.append(f"G1 {key} pair artifact identity invalid")

    teacher = seal.get("teacher", {})
    if teacher.get("checkpoint_sha256") != TEACHER_SHA256:
        errors.append("G1 teacher checkpoint SHA mismatch")
    if teacher.get("class_map_sha256") != CLASS_MAP_SHA256:
        errors.append("G1 teacher class-map binding mismatch")
    if int(teacher.get("checkpoint_bytes", 0) or 0) <= 0:
        errors.append("G1 teacher checkpoint byte count invalid")
    if teacher.get("canonical_state") != "EMA":
        errors.append("G1 teacher canonical state is not EMA")
    for field in (
        "byte_evidence_sha256",
        "factory_bundle_sha256",
        "factory_manifest_sha256",
        "class_order_evidence_sha256",
        "canonical_state_evidence_sha256",
        "adapter_parity_evidence_sha256",
    ):
        if len(str(teacher.get(field, ""))) != 64:
            errors.append(f"G1 teacher {field} invalid")
    if not teacher.get("factory_entrypoint"):
        errors.append("G1 teacher factory entrypoint missing")
    if seal.get("g1_seal_sha256") != g1_seal_hash(seal):
        errors.append("G1 seal self-hash mismatch")
    return errors



def validate_v1_test_access_identity(record: dict[str, Any]) -> tuple[list[str], str]:
    errors: list[str] = []
    canonical_present = "v1_test_accessed" in record
    legacy_present = "protected_test_accessed_during_g1" in record
    canonical = record.get("v1_test_accessed")
    legacy = record.get("protected_test_accessed_during_g1")

    if canonical_present:
        if canonical is not False:
            errors.append("frozen-V1 identity unexpectedly records V1-test access")
        if legacy_present and legacy is not canonical:
            errors.append("frozen-V1 identity canonical/legacy test-access fields contradict")
        return errors, "canonical" if not legacy_present else "canonical_plus_legacy"

    if legacy_present:
        if legacy is not False:
            errors.append("legacy frozen-V1 identity unexpectedly records V1-test access")
            return errors, "invalid_legacy"
        return errors, "legacy_alias_false"

    errors.append("frozen-V1 identity lacks explicit V1-test access field")
    return errors, "missing"


def validate_mounted_g1(
    seal: dict[str, Any],
    *,
    authorized_source_sha: str,
    manifest_path: str | Path,
    class_map_path: str | Path,
    pretrained_path: str | Path,
    pretrained_provenance_path: str | Path,
    pair_dir: str | Path,
    evidence_dir: str | Path,
    teacher_checkpoint_path: str | Path,
    teacher_factory_manifest_path: str | Path,
    teacher_factory_source_root: str | Path,
    teacher_class_order_evidence_path: str | Path,
    teacher_canonical_state_evidence_path: str | Path,
    dependency_lock_path: str | Path,
    infra_smoke_evidence_path: str | Path,
    dual_gpu_smoke_evidence_path: str | Path,
    repo_root: str | Path,
) -> list[str]:
    errors = validate_g1_seal_object(seal)
    if seal.get("source_git_sha") != authorized_source_sha:
        errors.append("G1 seal source SHA differs from authorized source SHA")
    try:
        require_sha256(manifest_path, MANIFEST_SHA256, "V1 manifest")
    except Exception as exc:
        errors.append(str(exc))
    try:
        require_sha256(class_map_path, CLASS_MAP_SHA256, "class map")
    except Exception as exc:
        errors.append(str(exc))

    provenance = _load(pretrained_provenance_path)
    errors.extend(validate_pretrained_provenance(provenance, artifact_path=pretrained_path))
    student = seal.get("student", {}).get("pretrained", {})
    if provenance.get("artifact_sha256") != student.get("sha256"):
        errors.append("mounted MobileNetV4 provenance differs from G1 seal")
    if provenance.get("tensor_identity_sha256") != student.get("tensor_identity_sha256"):
        errors.append("mounted MobileNetV4 tensor identity differs from G1 seal")
    if provenance.get("candidate_serialization_format") != student.get("candidate_serialization_format"):
        errors.append("mounted MobileNetV4 serialization identity differs from G1 seal")
    if sha256_json(provenance) != student.get("provenance_sha256"):
        errors.append("MobileNetV4 provenance evidence hash differs from G1 seal")

    pair_dir = Path(pair_dir)
    evidence_dir = Path(evidence_dir)
    global_pretrained = student.get("sha256")
    for key, spec in PAIR_SPECS.items():
        seal_row = seal.get("pair_initializations", {}).get(key, {})
        ev_path = evidence_dir / f"PAIR_INIT_{key}.json"
        if not ev_path.exists():
            errors.append(f"PAIR_INIT_{key}.json missing")
            continue
        ev = _load(ev_path)
        if ev.get("pretrained_sha256") != global_pretrained:
            errors.append(f"PAIR_INIT_{key} pretrained SHA differs from global G1 pretrained SHA")
        if ev.get("pair_id") != spec["pair_id"] or int(ev.get("seed", -1)) != spec["seed"]:
            errors.append(f"PAIR_INIT_{key} pair/seed mismatch")
        if set(ev.get("authorized_consumers", [])) != set(spec["consumers"]):
            errors.append(f"PAIR_INIT_{key} consumer mismatch")
        binary = pair_dir / str(ev.get("student_init_basename", ""))
        if not binary.exists():
            errors.append(f"PAIR_INIT_{key} binary missing")
        else:
            actual_sha = sha256_file(binary)
            if actual_sha != ev.get("student_init_sha256") or actual_sha != seal_row.get("sha256"):
                errors.append(f"PAIR_INIT_{key} binary SHA mismatch")
            if binary.stat().st_size != int(seal_row.get("bytes", -1)):
                errors.append(f"PAIR_INIT_{key} binary byte count mismatch")
        if seal_row.get("evidence_sha256") != sha256_json(ev):
            errors.append(f"PAIR_INIT_{key} evidence object differs from G1 seal")

    try:
        teacher_sha = require_sha256(
            teacher_checkpoint_path, TEACHER_SHA256, "historical DINO teacher"
        )
        if teacher_sha != seal.get("teacher", {}).get("checkpoint_sha256"):
            errors.append("mounted teacher differs from G1 seal")
    except Exception as exc:
        errors.append(str(exc))

    factory_manifest = _load(teacher_factory_manifest_path)
    errors.extend(validate_teacher_factory_bundle(
        factory_manifest,
        source_root=teacher_factory_source_root,
        expected_entrypoint=seal.get("teacher", {}).get("factory_entrypoint"),
    ))
    if factory_manifest.get("bundle_sha256") != seal.get("teacher", {}).get("factory_bundle_sha256"):
        errors.append("teacher factory bundle differs from G1 seal")
    if sha256_json(factory_manifest) != seal.get("teacher", {}).get("factory_manifest_sha256"):
        errors.append("teacher factory manifest evidence differs from G1 seal")

    teacher_ev_path = evidence_dir / "DINO_TEACHER.json"
    if not teacher_ev_path.exists():
        errors.append("DINO_TEACHER.json missing from G1 evidence bundle")
    else:
        teacher_ev = _load(teacher_ev_path)
        if teacher_ev.get("sha256") != TEACHER_SHA256:
            errors.append("teacher byte evidence SHA mismatch")
        if teacher_ev.get("class_map_sha256") != CLASS_MAP_SHA256:
            errors.append("teacher byte evidence class-map binding mismatch")
        if sha256_json(teacher_ev) != seal.get("teacher", {}).get("byte_evidence_sha256"):
            errors.append("teacher byte evidence differs from G1 seal")

    order = _load(teacher_class_order_evidence_path)
    errors.extend(validate_teacher_class_order_evidence(
        order,
        factory_bundle_sha256=factory_manifest.get("bundle_sha256"),
    ))
    if sha256_json(order) != seal.get("teacher", {}).get("class_order_evidence_sha256"):
        errors.append("teacher class-order evidence differs from G1 seal")

    canonical = _load(teacher_canonical_state_evidence_path)
    errors.extend(validate_teacher_canonical_state_evidence(
        canonical,
        factory_bundle_sha256=factory_manifest.get("bundle_sha256"),
    ))
    if sha256_json(canonical) != seal.get("teacher", {}).get("canonical_state_evidence_sha256"):
        errors.append("teacher canonical-state evidence differs from G1 seal")

    frozen_v1_path = evidence_dir / "FROZEN_V1_IDENTITY.json"
    if not frozen_v1_path.is_file():
        errors.append("FROZEN_V1_IDENTITY.json missing from G1 evidence bundle")
    else:
        frozen_v1 = _load(frozen_v1_path)
        if frozen_v1.get("manifest_sha256") != MANIFEST_SHA256:
            errors.append("frozen-V1 identity manifest SHA mismatch")
        if frozen_v1.get("class_map_sha256") != CLASS_MAP_SHA256:
            errors.append("frozen-V1 identity class-map SHA mismatch")
        v1_access_errors, _v1_access_mode = validate_v1_test_access_identity(frozen_v1)
        errors.extend(v1_access_errors)
        if sha256_json(frozen_v1) != seal.get("dataset", {}).get("identity_evidence_sha256"):
            errors.append("frozen-V1 identity evidence differs from G1 seal")

    adapter_path = evidence_dir / "TEACHER_ADAPTER_PARITY_EVIDENCE.json"
    if not adapter_path.is_file():
        errors.append("TEACHER_ADAPTER_PARITY_EVIDENCE.json missing from G1 evidence bundle")
    else:
        adapter = _load(adapter_path)
        if adapter.get("status") != "PASS" or adapter.get("direct_vs_adapter_parity") is not True:
            errors.append("teacher adapter parity evidence is not PASS")
        if adapter.get("protected_data_accessed") is not False:
            errors.append("teacher adapter parity evidence accessed protected data")
        if adapter.get("scientific_metric_computed") is not False:
            errors.append("teacher adapter parity evidence computed a scientific metric")
        if sha256_json(adapter) != seal.get("teacher", {}).get("adapter_parity_evidence_sha256"):
            errors.append("teacher adapter parity evidence differs from G1 seal")

    dep = _load(dependency_lock_path)
    errors.extend(validate_dependency_lock_object(dep))
    if dep.get("dependency_lock_sha256") != seal.get("dependency_lock_sha256"):
        errors.append("dependency lock differs from G1 seal")
    errors.extend(validate_dependency_environment(dep))

    smoke = _load(infra_smoke_evidence_path)
    errors.extend(
        "mounted infrastructure-smoke evidence: " + message
        for message in validate_terminal_smoke_b_evidence(
            smoke,
            expected_source_sha=authorized_source_sha,
            expected_dependency_lock_sha256=dep.get("dependency_lock_sha256", ""),
            require_batch=True,
        )
    )
    smoke_sha = sha256_json(smoke)
    if smoke_sha != seal.get("infra_smoke_evidence_sha256"):
        errors.append("infrastructure-smoke evidence differs from G1 seal")

    dual = _load(dual_gpu_smoke_evidence_path)
    from .envelope import AMENDMENT_ID, AMENDMENT_SHA256
    from .smoke_handoff import validate_terminal_dual_gpu_smoke_evidence
    errors.extend(
        "mounted dual-GPU-smoke evidence: " + message
        for message in validate_terminal_dual_gpu_smoke_evidence(
            dual,
            expected_source_sha=authorized_source_sha,
            expected_dependency_lock_sha256=dep.get("dependency_lock_sha256", ""),
            expected_amendment_id=AMENDMENT_ID,
            expected_amendment_sha256=AMENDMENT_SHA256,
            expected_smoke_b_evidence_sha256=smoke_sha,
            require_batch=True,
        )
    )
    if sha256_json(dual) != seal.get("dual_gpu_smoke_evidence_sha256"):
        errors.append("dual-GPU-smoke evidence differs from G1 seal")
    return errors

def assert_g1_creation_target_fresh(bundle_dir: str | Path) -> None:
    bundle = Path(bundle_dir)
    seal = bundle / "G1_MODEL_IDENTITY_SEAL.json"
    if seal.exists():
        raise G1Error("G1 seal already exists; immutable G1 cannot be regenerated in place")
    for key in PAIR_SPECS:
        candidates = (
            bundle / "private" / f"PAIR_INIT_{key}.pt",
            bundle / "evidence" / f"PAIR_INIT_{key}.json",
        )
        if any(path.exists() for path in candidates):
            raise G1Error(f"pre-existing PAIR_INIT_{key} material found; invalidate the prior G1 attempt explicitly before regeneration")
