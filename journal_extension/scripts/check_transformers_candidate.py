from __future__ import annotations

import importlib.metadata
import json
import os
import socket
from contextlib import contextmanager
from pathlib import Path

EXPECTED = {
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


@contextmanager
def no_network_connect():
    original = socket.socket.connect

    def blocked(self, address):
        raise RuntimeError(f"network connection attempted during offline DINO construction: {address!r}")

    socket.socket.connect = blocked
    try:
        yield
    finally:
        socket.socket.connect = original


def main() -> int:
    observed = {name: importlib.metadata.version(name) for name in EXPECTED}
    errors = []
    for name, expected in EXPECTED.items():
        if observed[name] != expected:
            errors.append(f"version mismatch {name}: expected {expected}, got {observed[name]}")

    import platform
    if platform.python_version() != "3.12.13":
        errors.append(f"Python mismatch: expected 3.12.13, got {platform.python_version()}")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from transformers import DINOv3ConvNextConfig, DINOv3ConvNextModel

    cfg = DINOv3ConvNextConfig()
    if list(cfg.hidden_sizes) != [96, 192, 384, 768]:
        errors.append(f"unexpected tiny hidden_sizes: {cfg.hidden_sizes}")
    if list(cfg.depths) != [3, 3, 9, 3]:
        errors.append(f"unexpected tiny depths: {cfg.depths}")
    if int(cfg.hidden_sizes[-1]) != 768:
        errors.append("default DINOv3 ConvNeXt feature width is not 768")

    class HistoricalWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = DINOv3ConvNextModel(DINOv3ConvNextConfig())
            hidden = int(self.backbone.config.hidden_sizes[-1])
            self.dropout = torch.nn.Dropout(0.10)
            self.head = torch.nn.Linear(hidden, 120)

        def forward(self, x):
            return self.head(self.dropout(self.backbone(pixel_values=x).pooler_output))

        def forward_features(self, x):
            return self.backbone(pixel_values=x).pooler_output

        def forward_head(self, features, pre_logits=False):
            return features if pre_logits else self.head(self.dropout(features))

    torch.manual_seed(12345)
    with no_network_connect():
        first = HistoricalWrapper()
    state = {k: v.detach().clone() for k, v in first.state_dict().items()}
    with no_network_connect():
        second = HistoricalWrapper()
    incompatible = second.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        errors.append(
            "synthetic strict load returned incompatible keys: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    if tuple(second.head.weight.shape) != (120, 768):
        errors.append(f"historical wrapper head shape mismatch: {tuple(second.head.weight.shape)}")

    report = {
        "schema_version": "1.0",
        "status": "PASS" if not errors else "FAIL",
        "python": platform.python_version(),
        "packages": observed,
        "dinov3_default_hidden_sizes": list(cfg.hidden_sizes),
        "dinov3_default_depths": list(cfg.depths),
        "feature_width": int(cfg.hidden_sizes[-1]),
        "historical_wrapper_head_shape": list(second.head.weight.shape),
        "direct_config_construction_network_attempted": False,
        "synthetic_strict_load": not bool(incompatible.missing_keys or incompatible.unexpected_keys),
        "errors": errors,
    }
    Path("transformers-candidate-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 3


if __name__ == "__main__":
    raise SystemExit(main())
