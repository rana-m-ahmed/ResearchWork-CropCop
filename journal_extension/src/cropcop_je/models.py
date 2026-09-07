from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from typing import Any

from .g1 import factory_bundle_hash, validate_teacher_factory_bundle
from .hashing import require_sha256, sha256_file

MNV4_MODEL_NAME = "mobilenetv4_conv_medium.e500_r256_in1k"
TEACHER_SHA256 = "74b4701b8931976c9227845ead50788ae47e3596f575f2817b7352a715f53b79"


def _state_dict_from_file(path: str | Path):
    import torch
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError("pretrained object does not contain a state_dict")
    if state and all(str(k).startswith("module.") for k in state):
        state = {str(k)[7:]: v for k, v in state.items()}
    return state


def create_student_from_pretrained(pretrained_path: str | Path, *, seed: int, num_classes: int = 120):
    import torch
    import timm
    if timm.__version__ != "1.0.26":
        raise RuntimeError(f"timm version drift: required 1.0.26, got {timm.__version__}")
    model = timm.create_model(MNV4_MODEL_NAME, pretrained=False, num_classes=1000)
    model.load_state_dict(_state_dict_from_file(pretrained_path), strict=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model.reset_classifier(num_classes)
    return model


def save_pair_initialization(model, path: str | Path, *, pair_id: str, seed: int, pretrained_sha256: str):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": "1.0",
        "pair_id": pair_id,
        "seed": int(seed),
        "model_name": MNV4_MODEL_NAME,
        "num_classes": 120,
        "pretrained_sha256": pretrained_sha256,
        "model_state": model.state_dict(),
    }, path)
    return sha256_file(path)


def load_pair_initialization(path: str | Path, *, expected_sha256: str, pair_id: str, seed: int):
    import timm
    import torch
    require_sha256(path, expected_sha256, "paired student initialization")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("pair_id") != pair_id or int(payload.get("seed", -1)) != int(seed):
        raise ValueError("paired student initialization identity mismatch")
    if payload.get("model_name") != MNV4_MODEL_NAME or int(payload.get("num_classes", -1)) != 120:
        raise ValueError("paired student initialization model identity mismatch")
    model = timm.create_model(MNV4_MODEL_NAME, pretrained=False, num_classes=120)
    model.load_state_dict(payload["model_state"], strict=True)
    return model, payload


def teacher_factory_identity(factory_spec: str) -> dict[str, Any]:
    """Legacy single-file identity retained only for historical diagnostics."""
    if ":" not in factory_spec:
        raise ValueError("teacher factory must be specified as module:function")
    module_name, fn_name = factory_spec.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name)
    source_file = inspect.getsourcefile(fn) or getattr(module, "__file__", None)
    if not source_file:
        raise RuntimeError(f"cannot bind teacher-factory source file for {factory_spec}")
    source_path = Path(source_file).resolve()
    return {
        "factory_spec": factory_spec,
        "module": module_name,
        "function": fn_name,
        "source_basename": source_path.name,
        "source_sha256": sha256_file(source_path),
        "identity_scope": "legacy_single_file_only",
    }


def teacher_factory_bundle_identity(
    factory_spec: str,
    *,
    factory_bundle_manifest: str | Path,
    repo_root: str | Path,
    factory_source_root: str | Path | None = None,
) -> dict[str, Any]:
    import json
    manifest = json.loads(Path(factory_bundle_manifest).read_text(encoding="utf-8"))
    errors = validate_teacher_factory_bundle(
        manifest,
        source_root=factory_source_root or repo_root,
        expected_entrypoint=factory_spec,
    )
    if errors:
        raise RuntimeError("teacher factory bundle invalid: " + "; ".join(errors))
    return {
        "factory_spec": factory_spec,
        "bundle_sha256": manifest["bundle_sha256"],
        "manifest_sha256": __import__("cropcop_je.hashing", fromlist=["sha256_json"]).sha256_json(manifest),
        "output_order_transform": manifest.get("output_order_transform"),
        "files": manifest.get("files", []),
    }


def load_exact_teacher(
    checkpoint_path: str | Path,
    *,
    factory_spec: str,
    factory_bundle_manifest: str | Path | None = None,
    repo_root: str | Path = ".",
    factory_source_root: str | Path | None = None,
):
    import torch
    require_sha256(checkpoint_path, TEACHER_SHA256, "historical DINO teacher")
    if not factory_bundle_manifest:
        raise ValueError("sealed teacher factory bundle manifest is required")
    identity = teacher_factory_bundle_identity(
        factory_spec,
        factory_bundle_manifest=factory_bundle_manifest,
        repo_root=repo_root,
        factory_source_root=factory_source_root,
    )
    module_name, fn_name = factory_spec.split(":", 1)
    if factory_source_root:
        source_root = str(Path(factory_source_root).resolve())
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
    teacher = getattr(importlib.import_module(module_name), fn_name)(str(checkpoint_path))
    if not isinstance(teacher, torch.nn.Module):
        raise TypeError("teacher factory did not return torch.nn.Module")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher, identity


def build_projection_without_state_drift(student, teacher, *, seed: int):
    import torch
    student_was_training = student.training
    teacher_was_training = teacher.training
    student.eval()
    teacher.eval()
    try:
        with torch.no_grad():
            dummy = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
            student_feat, _ = prelogits_and_logits(student, dummy)
            teacher_feat, _ = prelogits_and_logits(teacher, dummy)
    finally:
        student.train(student_was_training)
        teacher.train(teacher_was_training)
    if student_feat.ndim != 2 or teacher_feat.ndim != 2:
        raise RuntimeError(
            f"pre-classifier features must be 2D; student={tuple(student_feat.shape)}, teacher={tuple(teacher_feat.shape)}"
        )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        projection = torch.nn.Linear(student_feat.shape[1], teacher_feat.shape[1], bias=False)
        torch.nn.init.xavier_uniform_(projection.weight)
    return projection


def prelogits_and_logits(model, x):
    if not hasattr(model, "forward_features") or not hasattr(model, "forward_head"):
        raise TypeError("model adapter must expose forward_features and forward_head")
    features = model.forward_features(x)
    return model.forward_head(features, pre_logits=True), model.forward_head(features, pre_logits=False)


class TorchvisionConvNeXtTinyAdapter:
    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        if name == "model":
            return object.__getattribute__(self, name)
        return getattr(self.model, name)

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        return self

    def train(self, mode=True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self

    @property
    def training(self):
        return self.model.training

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state, strict=True):
        return self.model.load_state_dict(state, strict=strict)

    def get_classifier(self):
        return self.model.classifier[2]

    def forward_features(self, x):
        return self.model.avgpool(self.model.features(x))

    def forward_head(self, features, pre_logits=False):
        x = self.model.classifier[0](features)
        x = self.model.classifier[1](x)
        return x if pre_logits else self.model.classifier[2](x)

    def __call__(self, x):
        return self.forward_head(self.forward_features(x), pre_logits=False)


def create_convnext_tiny_from_pretrained(pretrained_path: str | Path, *, seed: int, num_classes: int = 120):
    import torch
    import torchvision
    from torchvision.models import convnext_tiny
    if torchvision.__version__.split("+", 1)[0] != "0.27.1":
        raise RuntimeError(f"torchvision version drift: required 0.27.1, got {torchvision.__version__}")
    model = convnext_tiny(weights=None, num_classes=1000)
    model.load_state_dict(_state_dict_from_file(pretrained_path), strict=True)
    in_features = model.classifier[2].in_features
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model.classifier[2] = torch.nn.Linear(in_features, num_classes)
    return TorchvisionConvNeXtTinyAdapter(model)
