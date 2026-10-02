from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from transformers import DINOv3ConvNextConfig, DINOv3ConvNextModel

TEACHER_SHA256 = "74b4701b8931976c9227845ead50788ae47e3596f575f2817b7352a715f53b79"
MANIFEST_SHA256 = "bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2"
CLASS_MAP_SHA256 = "46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2"
SEMANTIC_FINGERPRINT = "7c368e6e3d8be3bb3a9a3a5f961075d4faa125bcac2e98a3b55e1a1c61f1c523"
EXPERIMENT_ID = "stage1_dino_tiny_ce_256"
MODEL_KEY = "dino_tiny"
MODEL_ID = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
RESOLUTION = 256
NUM_CLASSES = 120
FEATURE_WIDTH = 768
DROPOUT = 0.10
CLASSIFIER_WEIGHT_KEY = "head.weight"
CANONICAL_STATE = "EMA"
OUTPUT_ORDER_TRANSFORM = "none"
SCIENTIFIC_ABI = "cropcop-v5-ramsafe-ampstep-segmented-1"


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class HistoricalDINOv3Classifier(torch.nn.Module):
    """Exact public-safe reconstruction of the historical CropCop DINO wrapper."""

    def __init__(self) -> None:
        super().__init__()
        config = DINOv3ConvNextConfig()
        if list(config.hidden_sizes) != [96, 192, 384, 768]:
            raise RuntimeError(f"unexpected DINOv3 tiny hidden_sizes: {config.hidden_sizes}")
        if list(config.depths) != [3, 3, 9, 3]:
            raise RuntimeError(f"unexpected DINOv3 tiny depths: {config.depths}")
        self.backbone = DINOv3ConvNextModel(config)
        hidden = int(self.backbone.config.hidden_sizes[-1])
        if hidden != FEATURE_WIDTH:
            raise RuntimeError(f"unexpected DINOv3 tiny feature width: {hidden}")
        self.dropout = torch.nn.Dropout(DROPOUT)
        self.head = torch.nn.Linear(hidden, NUM_CLASSES)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(pixel_values=x).pooler_output

    def forward_head(self, features: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if pre_logits:
            return features
        return self.head(self.dropout(features))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x), pre_logits=False)


def _require_checkpoint_identity(checkpoint: dict[str, Any]) -> None:
    required = {
        "model", "ema", "config", "preprocess", "training_engine_abi",
        "manifest_file_sha256", "class_map_sha256",
        "dataset_manifest_fingerprint", "best_metric", "best_epoch",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise RuntimeError(f"historical teacher checkpoint missing keys: {missing}")
    if checkpoint.get("training_engine_abi") != SCIENTIFIC_ABI:
        raise RuntimeError("historical teacher training-engine ABI mismatch")
    if checkpoint.get("manifest_file_sha256") != MANIFEST_SHA256:
        raise RuntimeError("historical teacher manifest SHA mismatch")
    if checkpoint.get("class_map_sha256") != CLASS_MAP_SHA256:
        raise RuntimeError("historical teacher class-map SHA mismatch")
    if checkpoint.get("dataset_manifest_fingerprint") != SEMANTIC_FINGERPRINT:
        raise RuntimeError("historical teacher semantic fingerprint mismatch")
    cfg = checkpoint.get("config") or {}
    if cfg.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("historical teacher experiment ID mismatch")
    if cfg.get("model_key") != MODEL_KEY:
        raise RuntimeError("historical teacher model key mismatch")
    if int(cfg.get("resolution", -1)) != RESOLUTION:
        raise RuntimeError("historical teacher resolution mismatch")
    raw = checkpoint.get("model")
    if not isinstance(raw, dict) or not raw:
        raise RuntimeError("historical teacher raw model state missing")
    if any(str(key).startswith("module.") for key in raw):
        raise RuntimeError("historical teacher raw state contains forbidden module.* prefix")
    ema = checkpoint.get("ema")
    if not isinstance(ema, dict) or not isinstance(ema.get("shadow"), dict) or not ema["shadow"]:
        raise RuntimeError("historical teacher EMA shadow missing")


def _construct_canonical(
    checkpoint_path: str | Path,
) -> tuple[HistoricalDINOv3Classifier, dict[str, Any], dict[str, Any]]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"historical teacher checkpoint missing: {path}")
    actual_sha = _sha256_file(path)
    if actual_sha != TEACHER_SHA256:
        raise RuntimeError(
            f"historical teacher SHA mismatch: expected {TEACHER_SHA256}, got {actual_sha}"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("historical teacher checkpoint root must be a dict")
    _require_checkpoint_identity(checkpoint)

    model = HistoricalDINOv3Classifier()
    raw_state = checkpoint["model"]
    model.load_state_dict(raw_state, strict=True)

    state = model.state_dict()
    floating_keys = {key for key, value in state.items() if torch.is_floating_point(value)}
    shadow = checkpoint["ema"]["shadow"]
    shadow_keys = set(shadow)
    if shadow_keys != floating_keys:
        missing = sorted(floating_keys - shadow_keys)
        extra = sorted(shadow_keys - floating_keys)
        raise RuntimeError(
            "historical teacher EMA coverage mismatch "
            f"missing={missing[:12]} extra={extra[:12]}"
        )

    with torch.no_grad():
        for key in sorted(floating_keys):
            target = state[key]
            value = shadow[key]
            if not torch.is_tensor(value):
                raise RuntimeError(f"EMA shadow entry is not a tensor: {key}")
            if tuple(value.shape) != tuple(target.shape):
                raise RuntimeError(f"EMA shadow shape mismatch for {key}")
            if value.dtype != target.dtype:
                raise RuntimeError(
                    f"EMA shadow dtype mismatch for {key}: {value.dtype} != {target.dtype}"
                )
            if not torch.isfinite(value).all():
                raise RuntimeError(f"non-finite EMA shadow tensor: {key}")
            target.copy_(value)

    canonical = model.state_dict()
    for key, value in canonical.items():
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            raise RuntimeError(f"non-finite canonical teacher tensor: {key}")
    if CLASSIFIER_WEIGHT_KEY not in canonical:
        raise RuntimeError("historical teacher classifier weight key missing")
    if tuple(canonical[CLASSIFIER_WEIGHT_KEY].shape) != (NUM_CLASSES, FEATURE_WIDTH):
        raise RuntimeError(
            f"historical teacher classifier shape mismatch: "
            f"{tuple(canonical[CLASSIFIER_WEIGHT_KEY].shape)}"
        )

    equality_failures = [
        key for key in sorted(floating_keys)
        if not torch.equal(canonical[key].detach().cpu(), shadow[key].detach().cpu())
    ]
    if equality_failures:
        raise RuntimeError(
            "canonical teacher state differs from EMA shadow: "
            + repr(equality_failures[:12])
        )

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    facts = {
        "checkpoint_sha256": actual_sha,
        "checkpoint_bytes": path.stat().st_size,
        "raw_model_state_key_count": len(raw_state),
        "floating_state_key_count": len(floating_keys),
        "ema_shadow_key_count": len(shadow_keys),
        "ema_exact_complete_coverage": True,
        "canonical_floating_tensors_equal_ema": True,
        "canonical_state": CANONICAL_STATE,
        "classifier_weight_key": CLASSIFIER_WEIGHT_KEY,
        "classifier_shape": [NUM_CLASSES, FEATURE_WIDTH],
        "feature_dimension": FEATURE_WIDTH,
        "finite_state": True,
        "output_order_transform": OUTPUT_ORDER_TRANSFORM,
        "model_id": MODEL_ID,
        "experiment_id": EXPERIMENT_ID,
        "model_key": MODEL_KEY,
        "resolution": RESOLUTION,
        "manifest_sha256": MANIFEST_SHA256,
        "class_map_sha256": CLASS_MAP_SHA256,
        "semantic_manifest_fingerprint": SEMANTIC_FINGERPRINT,
    }
    return model, checkpoint, facts


def inspect_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    _model, _checkpoint, facts = _construct_canonical(checkpoint_path)
    return facts


def build_teacher(checkpoint_path: str | Path) -> HistoricalDINOv3Classifier:
    model, _checkpoint, _facts = _construct_canonical(checkpoint_path)
    return model
