from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.g1 import validate_teacher_factory_bundle

ABS_TOL = 1e-7
REL_TOL = 1e-6
SYNTHETIC_SHAPE = (1, 3, 256, 256)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--teacher-factory-root", required=True)
    ap.add_argument("--teacher-factory", required=True)
    ap.add_argument("--teacher-factory-manifest", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = Path(args.teacher_factory_root).resolve()
    manifest = json.loads(Path(args.teacher_factory_manifest).read_text(encoding="utf-8"))
    errors = validate_teacher_factory_bundle(
        manifest,
        source_root=root,
        expected_entrypoint=args.teacher_factory,
    )
    if errors:
        raise SystemExit("teacher factory bundle invalid: " + "; ".join(errors))
    if ":" not in args.teacher_factory:
        raise SystemExit("teacher factory spec must be module:function")
    module_name, fn_name = args.teacher_factory.split(":", 1)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    factory = getattr(importlib.import_module(module_name), fn_name)
    teacher = factory(args.checkpoint)
    teacher.eval()

    import torch
    values = torch.linspace(-1.0, 1.0, steps=3 * 256 * 256, dtype=torch.float32)
    x = values.reshape(SYNTHETIC_SHAPE)
    with torch.no_grad():
        direct = teacher(x)
        features = teacher.forward_features(x)
        adapter = teacher.forward_head(features, pre_logits=False)
        prelogits = teacher.forward_head(features, pre_logits=True)

    if tuple(features.shape) != (1, 768):
        raise SystemExit(f"synthetic feature shape mismatch: {tuple(features.shape)}")
    if tuple(prelogits.shape) != tuple(features.shape):
        raise SystemExit("pre_logits adapter does not return historical features")
    if tuple(direct.shape) != (1, 120) or tuple(adapter.shape) != (1, 120):
        raise SystemExit(
            f"synthetic logits shape mismatch direct={tuple(direct.shape)} adapter={tuple(adapter.shape)}"
        )
    for name, value in (("features", features), ("direct", direct), ("adapter", adapter)):
        if not torch.isfinite(value).all():
            raise SystemExit(f"non-finite synthetic {name}")

    delta = (direct - adapter).abs()
    max_abs = float(delta.max().item())
    mean_abs = float(delta.mean().item())
    parity = bool(torch.allclose(direct, adapter, atol=ABS_TOL, rtol=REL_TOL))
    record = {
        "schema_version": "1.0",
        "status": "PASS" if parity else "FAIL",
        "synthetic_input_shape": list(SYNTHETIC_SHAPE),
        "feature_shape": list(features.shape),
        "logits_shape": list(direct.shape),
        "absolute_tolerance": ABS_TOL,
        "relative_tolerance": REL_TOL,
        "max_absolute_difference": max_abs,
        "mean_absolute_difference": mean_abs,
        "direct_vs_adapter_parity": parity,
        "finite_values": True,
        "protected_data_accessed": False,
        "scientific_metric_computed": False,
    }
    atomic_write_json(args.output, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if parity else 3


if __name__ == "__main__":
    raise SystemExit(main())
