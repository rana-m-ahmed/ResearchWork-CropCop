from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hashing import require_sha256, sha256_file, sha256_json
from .models import _state_dict_from_file, create_convnext_tiny_from_pretrained
from .surfaces import validate_training_config
from .tensor_identity import TENSOR_IDENTITY_ALGORITHM

AUTHORITY_ID = "EAAI-JE-SDL-v2.1-QA"
AUTHORITY_SHA256 = "aab17b65b0873dcb1ecedb061eb02ff60ccb09f8b830184f5e2231a600278f74"
MANIFEST_SHA256 = "bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2"
CLASS_MAP_SHA256 = "46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2"
PRINCIPAL_SCIENCE_SOURCE_SHA = "f171309fc7e9dc22241ecc137ebbb8e4bcdc5433"
PRINCIPAL_G1_SEAL_SHA256 = "442d9e7708749efedcb82ef9f4fd131211549770eecad985177eaaf4117052cd"
PRINCIPAL_S1_INIT_SHA256 = "7040e48fe697c539dbbdfa8f57ffd20e9327def82f6485267f224dcf4c34cce1"
PRINCIPAL_MNV4_PRETRAINED_SHA256 = "35ca23dc46c0075d9acdcab06d30e5395c4d97e422d8cb5e5e627823aeb7c1fa"
MNV4_MODEL_NAME = "mobilenetv4_conv_medium.e500_r256_in1k"
TEACHER_SHA256 = "74b4701b8931976c9227845ead50788ae47e3596f575f2817b7352a715f53b79"
TORCHVISION_VERSION = "0.27.1"
S1_SEED = 21270083
S1_PAIR_ID = "MNV4-PAIR-S1"

BASELINE_SPECS = {
    "effb0": {
        "model_name": "torchvision.models.efficientnet_b0",
        "weight_enum": "EfficientNet_B0_Weights.IMAGENET1K_V1",
        "official_filename": "efficientnet_b0_rwightman-7f5810bc.pth",
        "official_sha256_prefix": "7f5810bc",
        "init_basename": "EFFB0_INIT_S1.pt",
        "init_evidence_basename": "EFFB0_INIT_S1.json",
        "pretrained_basename": "EFFB0_PRETRAINED.pth",
        "pretrained_evidence_basename": "EFFB0_PRETRAINED_PROVENANCE.json",
        "consumer": "R06-EFFB0-CONTEXT-S1",
    },
    "cnxtt": {
        "model_name": "torchvision.models.convnext_tiny",
        "weight_enum": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
        "official_filename": "convnext_tiny-983f1562.pth",
        "official_sha256_prefix": "983f1562",
        "init_basename": "CNXTT_INIT_S1.pt",
        "init_evidence_basename": "CNXTT_INIT_S1.json",
        "pretrained_basename": "CNXTT_PRETRAINED.pth",
        "pretrained_evidence_basename": "CNXTT_PRETRAINED_PROVENANCE.json",
        "consumer": "R07-CNXTT-CONTEXT-S1",
    },
}

SECONDARY_CONFIG_SPECS = {
    "R12-MNV4-LOGITS-S1": {"seed": S1_SEED, "model_family": "mnv4", "model_name": MNV4_MODEL_NAME, "condition": "teacher", "objective": {"ce": 0.65, "kd": 0.35, "feature": 0.0}, "pair_id": S1_PAIR_ID},
    "R12-MNV4-FEATURE-S1": {"seed": S1_SEED, "model_family": "mnv4", "model_name": MNV4_MODEL_NAME, "condition": "teacher", "objective": {"ce": 0.85, "kd": 0.0, "feature": 0.15}, "pair_id": S1_PAIR_ID},
    "R06-EFFB0-CONTEXT-S1": {"seed": S1_SEED, "model_family": "effb0", "model_name": BASELINE_SPECS["effb0"]["model_name"], "condition": "direct", "objective": {"ce": 1.0, "kd": 0.0, "feature": 0.0}, "weight_enum": BASELINE_SPECS["effb0"]["weight_enum"]},
    "R07-CNXTT-CONTEXT-S1": {"seed": S1_SEED, "model_family": "cnxtt", "model_name": BASELINE_SPECS["cnxtt"]["model_name"], "condition": "direct", "objective": {"ce": 1.0, "kd": 0.0, "feature": 0.0}, "weight_enum": BASELINE_SPECS["cnxtt"]["weight_enum"]},
}

SECONDARY_ENVELOPES = {
    "SEC-MECHANISM-T4X2-V1": {"lane": "K1", "children": [{"experiment_id": "R12-MNV4-LOGITS-S1", "slot": 0}, {"experiment_id": "R12-MNV4-FEATURE-S1", "slot": 1}]},
    "SEC-CONTEXT-T4X2-V1": {"lane": "K2", "children": [{"experiment_id": "R06-EFFB0-CONTEXT-S1", "slot": 0}, {"experiment_id": "R07-CNXTT-CONTEXT-S1", "slot": 1}]},
}

PRINCIPAL_BACKFILL_ENVELOPES = {
    "VAL-BACKFILL-S1-T4X2-V1": {"lane": "K1", "children": [{"experiment_id": "R04-MNV4-DIRECT-S1", "slot": 0}, {"experiment_id": "R05-MNV4-TEACHER-S1", "slot": 1}]},
    "VAL-BACKFILL-S2-T4X2-V1": {"lane": "K2", "children": [{"experiment_id": "R04-MNV4-DIRECT-S2", "slot": 0}, {"experiment_id": "R05-MNV4-TEACHER-S2", "slot": 1}]},
    "VAL-BACKFILL-S3-T4X2-V1": {"lane": "K3", "children": [{"experiment_id": "R04-MNV4-DIRECT-S3", "slot": 0}, {"experiment_id": "R05-MNV4-TEACHER-S3", "slot": 1}]},
}


class SecondaryGateError(RuntimeError):
    pass


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def secondary_g1_hash(seal: dict[str, Any]) -> str:
    clean = dict(seal)
    clean.pop("secondary_g1_seal_sha256", None)
    return sha256_json(clean)


def validate_secondary_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        validate_training_config(config)
    except Exception as exc:
        return [str(exc)]
    eid = str(config.get("experiment_id", ""))
    expected = SECONDARY_CONFIG_SPECS.get(eid)
    if expected is None:
        return [f"experiment is not in the frozen secondary Track-A set: {eid}"]
    fixed = {
        "authority_id": AUTHORITY_ID,
        "manifest_sha256": MANIFEST_SHA256,
        "class_map_sha256": CLASS_MAP_SHA256,
        "num_classes": 120,
        "train_surface": "DS-V1-TRAIN",
        "validation_surface": "DS-V1-VAL",
    }
    for field, value in fixed.items():
        if config.get(field) != value:
            errors.append(f"secondary config drift for {eid}: {field}")
    for field in ("seed", "model_family", "model_name", "condition", "objective"):
        if config.get(field) != expected.get(field):
            errors.append(f"secondary config drift for {eid}: {field}")
    if "pair_id" in expected and config.get("pair_id") != expected["pair_id"]:
        errors.append(f"secondary config pair_id drift for {eid}")
    if "weight_enum" in expected:
        if config.get("torchvision_version") != TORCHVISION_VERSION:
            errors.append(f"secondary config torchvision version drift for {eid}")
        if config.get("weight_enum") != expected["weight_enum"]:
            errors.append(f"secondary config weight enum drift for {eid}")
    if eid.startswith("R12-") and config.get("teacher_checkpoint_sha256") != TEACHER_SHA256:
        errors.append(f"secondary config teacher identity drift for {eid}")
    forbidden = set(config.get("forbidden_surfaces", []))
    if "DS-V1-TEST-CONSUMED" not in forbidden or "DS-HIST-COMPARE" not in forbidden or "DS-EXT-*-SEALED" not in forbidden:
        errors.append(f"secondary config forbidden-surface policy drift for {eid}")
    return errors


def _torchvision_spec(model_key: str) -> dict[str, Any]:
    try:
        return BASELINE_SPECS[model_key]
    except KeyError as exc:
        raise SecondaryGateError(f"unknown baseline model key: {model_key}") from exc


def official_torchvision_identity(model_key: str):
    import torchvision
    from torchvision.models import ConvNeXt_Tiny_Weights, EfficientNet_B0_Weights, convnext_tiny, efficientnet_b0
    if torchvision.__version__.split("+", 1)[0] != TORCHVISION_VERSION:
        raise SecondaryGateError(f"torchvision version drift: expected {TORCHVISION_VERSION}, got {torchvision.__version__}")
    if model_key == "effb0":
        return efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1, _torchvision_spec(model_key)
    if model_key == "cnxtt":
        return convnext_tiny, ConvNeXt_Tiny_Weights.IMAGENET1K_V1, _torchvision_spec(model_key)
    raise SecondaryGateError(f"unknown baseline model key: {model_key}")


def validate_torchvision_provenance(record: dict[str, Any], *, model_key: str, artifact_path: str | Path | None = None) -> list[str]:
    spec = _torchvision_spec(model_key)
    errors: list[str] = []
    expected = {
        "status": "PASS",
        "torchvision_version": TORCHVISION_VERSION,
        "model_key": model_key,
        "model_name": spec["model_name"],
        "weight_enum": spec["weight_enum"],
        "official_filename": spec["official_filename"],
        "source_kind": "torchvision_weight_enum_url",
        "tensor_identity_algorithm": TENSOR_IDENTITY_ALGORITHM,
        "official_tensor_identity_algorithm": TENSOR_IDENTITY_ALGORITHM,
        "official_tensor_match": True,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            errors.append(f"TorchVision pretrained provenance {field} mismatch")
    if not str(record.get("artifact_sha256", "")).startswith(spec["official_sha256_prefix"]):
        errors.append("TorchVision pretrained SHA does not match official hash prefix")
    if int(record.get("artifact_bytes", 0) or 0) <= 0:
        errors.append("TorchVision pretrained byte count invalid")
    if not str(record.get("source_locator", "")).startswith("https://download.pytorch.org/models/"):
        errors.append("TorchVision pretrained provenance source locator is not official")
    if record.get("tensor_identity_sha256") != record.get("official_tensor_identity_sha256"):
        errors.append("TorchVision pretrained tensor identity differs from official state")
    if artifact_path is not None:
        path = Path(artifact_path)
        if not path.is_file():
            errors.append("TorchVision pretrained artifact missing")
        else:
            if sha256_file(path) != record.get("artifact_sha256"):
                errors.append("TorchVision pretrained artifact SHA mismatch")
            if path.stat().st_size != int(record.get("artifact_bytes", -1)):
                errors.append("TorchVision pretrained artifact byte-count mismatch")
    return errors


class TorchvisionEfficientNetB0Adapter:
    def __init__(self, model): self.model = model
    def __getattr__(self, name): return object.__getattribute__(self, name) if name == "model" else getattr(self.model, name)
    def to(self, *args, **kwargs): self.model.to(*args, **kwargs); return self
    def train(self, mode=True): self.model.train(mode); return self
    def eval(self): self.model.eval(); return self
    @property
    def training(self): return self.model.training
    def parameters(self): return self.model.parameters()
    def named_parameters(self): return self.model.named_parameters()
    def state_dict(self): return self.model.state_dict()
    def load_state_dict(self, state, strict=True): return self.model.load_state_dict(state, strict=strict)
    def get_classifier(self): return self.model.classifier[1]
    def forward_features(self, x): return self.model.avgpool(self.model.features(x))
    def forward_head(self, features, pre_logits=False):
        import torch
        x = torch.flatten(features, 1)
        return x if pre_logits else self.model.classifier(x)
    def __call__(self, x): return self.forward_head(self.forward_features(x), pre_logits=False)


def create_baseline_from_pretrained(pretrained_path: str | Path, *, model_key: str, seed: int = S1_SEED, num_classes: int = 120):
    import torch
    import torchvision
    if torchvision.__version__.split("+", 1)[0] != TORCHVISION_VERSION:
        raise SecondaryGateError(f"torchvision version drift: expected {TORCHVISION_VERSION}, got {torchvision.__version__}")
    if model_key == "cnxtt":
        return create_convnext_tiny_from_pretrained(pretrained_path, seed=seed, num_classes=num_classes)
    if model_key != "effb0":
        raise SecondaryGateError(f"unknown baseline model key: {model_key}")
    from torchvision.models import efficientnet_b0
    model = efficientnet_b0(weights=None, num_classes=1000)
    model.load_state_dict(_state_dict_from_file(pretrained_path), strict=True)
    in_features = model.classifier[1].in_features
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    return TorchvisionEfficientNetB0Adapter(model)


def create_empty_baseline(*, model_key: str, num_classes: int = 120):
    import torchvision
    if torchvision.__version__.split("+", 1)[0] != TORCHVISION_VERSION:
        raise SecondaryGateError(f"torchvision version drift: expected {TORCHVISION_VERSION}, got {torchvision.__version__}")
    if model_key == "effb0":
        from torchvision.models import efficientnet_b0
        return TorchvisionEfficientNetB0Adapter(efficientnet_b0(weights=None, num_classes=num_classes))
    if model_key == "cnxtt":
        from torchvision.models import convnext_tiny
        from .models import TorchvisionConvNeXtTinyAdapter
        return TorchvisionConvNeXtTinyAdapter(convnext_tiny(weights=None, num_classes=num_classes))
    raise SecondaryGateError(f"unknown baseline model key: {model_key}")


def save_baseline_initialization(model, path: str | Path, *, model_key: str, seed: int, pretrained_sha256: str, authorized_consumers: list[str]) -> str:
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": "1.0", "model_key": model_key, "model_name": BASELINE_SPECS[model_key]["model_name"], "seed": int(seed), "num_classes": 120, "pretrained_sha256": pretrained_sha256, "authorized_consumers": list(authorized_consumers), "model_state": model.state_dict()}, path)
    return sha256_file(path)


def load_baseline_initialization(path: str | Path, *, expected_sha256: str, model_key: str, seed: int, experiment_id: str):
    import torch
    require_sha256(path, expected_sha256, f"{model_key} baseline initialization")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model_key") != model_key or int(payload.get("seed", -1)) != int(seed) or int(payload.get("num_classes", -1)) != 120:
        raise SecondaryGateError("baseline initialization identity mismatch")
    if experiment_id not in set(payload.get("authorized_consumers", [])):
        raise SecondaryGateError("experiment is not authorized to consume baseline initialization")
    model = create_empty_baseline(model_key=model_key, num_classes=120)
    model.load_state_dict(payload["model_state"], strict=True)
    return model, payload


def validate_secondary_g1_seal_object(seal: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    fixed = {
        "schema_version": "1.0",
        "principal_science_source_sha": PRINCIPAL_SCIENCE_SOURCE_SHA,
        "principal_g1_seal_sha256": PRINCIPAL_G1_SEAL_SHA256,
    }
    for field, value in fixed.items():
        if seal.get(field) != value: errors.append(f"secondary G1 {field} mismatch")
    if seal.get("authority", {}).get("id") != AUTHORITY_ID or seal.get("authority", {}).get("sha256") != AUTHORITY_SHA256:
        errors.append("secondary G1 authority mismatch")
    if len(str(seal.get("source_git_sha", ""))) != 40: errors.append("secondary G1 source Git SHA invalid")
    if seal.get("dataset", {}).get("manifest_sha256") != MANIFEST_SHA256 or seal.get("dataset", {}).get("class_map_sha256") != CLASS_MAP_SHA256:
        errors.append("secondary G1 dataset identity mismatch")
    if len(str(seal.get("dependency_lock_sha256", ""))) != 64: errors.append("secondary G1 dependency lock invalid")
    for field in ("infra_smoke_evidence_sha256", "dual_gpu_smoke_evidence_sha256"):
        if len(str(seal.get(field, ""))) != 64: errors.append(f"secondary G1 {field} invalid")
    mnv4 = seal.get("mnv4_s1", {})
    if mnv4.get("pair_id") != S1_PAIR_ID or int(mnv4.get("seed", -1)) != S1_SEED: errors.append("secondary G1 MNV4 S1 pair mismatch")
    if mnv4.get("student_init_sha256") != PRINCIPAL_S1_INIT_SHA256 or mnv4.get("pretrained_sha256") != PRINCIPAL_MNV4_PRETRAINED_SHA256: errors.append("secondary G1 MNV4 anchor mismatch")
    if set(mnv4.get("authorized_consumers", [])) != {"R12-MNV4-LOGITS-S1", "R12-MNV4-FEATURE-S1"}: errors.append("secondary G1 MNV4 consumer mismatch")
    teacher = seal.get("teacher", {})
    if teacher.get("checkpoint_sha256") != TEACHER_SHA256: errors.append("secondary G1 teacher checkpoint mismatch")
    for field in ("factory_bundle_sha256", "factory_manifest_sha256", "class_order_evidence_sha256", "canonical_state_evidence_sha256", "adapter_parity_evidence_sha256"):
        if len(str(teacher.get(field, ""))) != 64: errors.append(f"secondary G1 teacher {field} invalid")
    for model_key, spec in BASELINE_SPECS.items():
        row = seal.get("baselines", {}).get(model_key, {})
        if row.get("model_name") != spec["model_name"] or row.get("weight_enum") != spec["weight_enum"] or row.get("torchvision_version") != TORCHVISION_VERSION: errors.append(f"secondary G1 {model_key} model identity mismatch")
        if row.get("authorized_consumers") != [spec["consumer"]]: errors.append(f"secondary G1 {model_key} consumer mismatch")
        for field in ("pretrained_sha256", "pretrained_provenance_sha256", "init_sha256", "init_evidence_sha256"):
            if len(str(row.get(field, ""))) != 64: errors.append(f"secondary G1 {model_key} {field} invalid")
        if not str(row.get("pretrained_sha256", "")).startswith(spec["official_sha256_prefix"]): errors.append(f"secondary G1 {model_key} pretrained SHA prefix mismatch")
    if seal.get("secondary_g1_seal_sha256") != secondary_g1_hash(seal): errors.append("secondary G1 seal self-hash mismatch")
    return errors


def validate_secondary_g1_bundle(bundle_dir: str | Path) -> tuple[dict[str, Any], list[str]]:
    root = Path(bundle_dir)
    seal_path = root / "SECONDARY_G1_MODEL_IDENTITY_SEAL.json"
    if not seal_path.is_file(): return {}, ["secondary G1 seal missing"]
    seal = load_json(seal_path)
    errors = validate_secondary_g1_seal_object(seal)
    private, evidence = root / "private", root / "evidence"
    mnv4 = seal.get("mnv4_s1", {})
    pair = private / str(mnv4.get("student_init_basename", ""))
    if not pair.is_file() or sha256_file(pair) != PRINCIPAL_S1_INIT_SHA256: errors.append("secondary G1 MNV4 S1 pair bytes mismatch")
    pair_ev = evidence / "PAIR_INIT_S1.json"
    if not pair_ev.is_file(): errors.append("secondary G1 MNV4 pair evidence missing")
    else:
        obj = load_json(pair_ev)
        if sha256_json(obj) != mnv4.get("student_init_evidence_sha256") or obj.get("student_init_sha256") != PRINCIPAL_S1_INIT_SHA256 or obj.get("pretrained_sha256") != PRINCIPAL_MNV4_PRETRAINED_SHA256: errors.append("secondary G1 MNV4 pair evidence mismatch")
    teacher = seal.get("teacher", {})
    tpath = private / str(teacher.get("artifact_basename", ""))
    if not tpath.is_file() or sha256_file(tpath) != TEACHER_SHA256: errors.append("secondary G1 teacher bytes mismatch")
    for basename, field in (("TEACHER_FACTORY_BUNDLE.json", "factory_manifest_sha256"), ("TEACHER_CLASS_ORDER_EVIDENCE.json", "class_order_evidence_sha256"), ("TEACHER_CANONICAL_STATE_EVIDENCE.json", "canonical_state_evidence_sha256"), ("TEACHER_ADAPTER_PARITY_EVIDENCE.json", "adapter_parity_evidence_sha256")):
        p = evidence / basename
        if not p.is_file() or sha256_json(load_json(p)) != teacher.get(field): errors.append(f"secondary G1 teacher evidence mismatch: {basename}")
    for key, spec in BASELINE_SPECS.items():
        row = seal.get("baselines", {}).get(key, {})
        pretrained = private / str(row.get("pretrained_basename", "")); init = private / str(row.get("init_basename", "")); prov = evidence / spec["pretrained_evidence_basename"]; init_ev = evidence / spec["init_evidence_basename"]
        if not pretrained.is_file() or sha256_file(pretrained) != row.get("pretrained_sha256"): errors.append(f"secondary G1 {key} pretrained bytes mismatch")
        if not init.is_file() or sha256_file(init) != row.get("init_sha256"): errors.append(f"secondary G1 {key} init bytes mismatch")
        if not prov.is_file(): errors.append(f"secondary G1 {key} provenance missing")
        else:
            pobj = load_json(prov)
            if sha256_json(pobj) != row.get("pretrained_provenance_sha256"): errors.append(f"secondary G1 {key} provenance hash mismatch")
            errors.extend(f"secondary G1 {key} provenance: {e}" for e in validate_torchvision_provenance(pobj, model_key=key, artifact_path=pretrained))
        if not init_ev.is_file() or sha256_json(load_json(init_ev)) != row.get("init_evidence_sha256"): errors.append(f"secondary G1 {key} init evidence mismatch")
    return seal, errors


def experiment_config_path(experiment_id: str) -> str:
    mapping = {"R12-MNV4-LOGITS-S1": "journal_extension/configs/r12_logits/s1.json", "R12-MNV4-FEATURE-S1": "journal_extension/configs/r12_feature/s1.json", "R06-EFFB0-CONTEXT-S1": "journal_extension/configs/r06_effb0/s1.json", "R07-CNXTT-CONTEXT-S1": "journal_extension/configs/r07_cnxtt/s1.json"}
    try: return mapping[experiment_id]
    except KeyError as exc: raise SecondaryGateError(f"unknown secondary experiment ID: {experiment_id}") from exc


def calibration_forecast_key(experiment_id: str) -> str:
    if experiment_id.startswith("R12-"): return "CAL-MNV4-TEACHER"
    if experiment_id == "R06-EFFB0-CONTEXT-S1": return "CAL-EFFB0"
    if experiment_id == "R07-CNXTT-CONTEXT-S1": return "CAL-CNXTT"
    raise SecondaryGateError(f"no frozen calibration mapping for {experiment_id}")


def _validate_envelope(config: dict[str, Any], *, expected_map: dict[str, Any], phase: str, label: str) -> list[str]:
    errors: list[str] = []
    if config.get("schema_version") != "1.0": errors.append(f"unsupported {label} envelope schema")
    if config.get("phase") != phase: errors.append(f"{label} envelope phase mismatch")
    if config.get("hardware_profile") != "T4X2": errors.append(f"{label} envelope hardware profile must be T4X2")
    expected = expected_map.get(str(config.get("envelope_id", "")))
    if expected is None: return errors + [f"{label} envelope ID is not frozen"]
    if config.get("logical_lane") != expected["lane"]: errors.append(f"{label} envelope logical lane mismatch")
    children = config.get("children")
    if not isinstance(children, list) or len(children) != 2: return errors + [f"{label} envelope must contain exactly two children"]
    actual = [(str(c.get("experiment_id", "")), c.get("slot")) for c in children]; frozen = [(c["experiment_id"], c["slot"]) for c in expected["children"]]
    if actual != frozen: errors.append(f"{label} envelope experiment/slot mapping changed")
    return errors


def validate_secondary_envelope_config(config: dict[str, Any]) -> list[str]:
    errors = _validate_envelope(config, expected_map=SECONDARY_ENVELOPES, phase="secondary-scientific-dual", label="secondary")
    children = config.get("children", [])
    ids = [str(c.get("child_id", "")) for c in children] if isinstance(children, list) else []
    if len(ids) == 2 and (any(not x for x in ids) or len(set(ids)) != 2): errors.append("secondary envelope child IDs missing/non-unique")
    return errors


def validate_backfill_envelope_config(config: dict[str, Any]) -> list[str]:
    return _validate_envelope(config, expected_map=PRINCIPAL_BACKFILL_ENVELOPES, phase="principal-validation-backfill-dual", label="backfill")


def operational_worker_count(env: dict[str, str] | None = None) -> int:
    import os
    source = os.environ if env is None else env
    explicit = str(source.get("CROPCOP_NUM_WORKERS_PER_CHILD", "")).strip()
    if explicit:
        value = int(explicit)
        if not 0 <= value <= 8: raise SecondaryGateError("CROPCOP_NUM_WORKERS_PER_CHILD must be between 0 and 8")
        return value
    return max(1, min(4, (os.cpu_count() or 4) // 2))
