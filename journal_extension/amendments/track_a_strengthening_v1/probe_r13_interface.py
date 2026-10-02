from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_ID = "vit_dlittle_patch16_reg1_gap_256.sbb_nadamuon_in1k"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    import timm
    import torch

    model = timm.create_model(MODEL_ID, pretrained=False, num_classes=1000)
    model.eval()

    if not hasattr(model, "blocks") or len(model.blocks) == 0:
        raise RuntimeError("R13 model has no transformer blocks")

    final_index = len(model.blocks) - 1
    final_block = model.blocks[final_index]
    prefix = f"blocks.{final_index}"

    module_inventory = []
    hooks = []
    observed_shapes: dict[str, object] = {}

    for rel_name, module in final_block.named_modules():
        full_name = prefix if not rel_name else f"{prefix}.{rel_name}"
        module_inventory.append(
            {
                "path": full_name,
                "type": f"{module.__class__.__module__}.{module.__class__.__qualname__}",
            }
        )
        if rel_name in {"norm1", "norm2", "in_norm"}:
            def make_hook(name: str):
                def hook(_module, _inputs, output):
                    if isinstance(output, torch.Tensor):
                        observed_shapes[name] = list(output.shape)
                    elif isinstance(output, (tuple, list)):
                        observed_shapes[name] = [
                            list(x.shape) if isinstance(x, torch.Tensor) else type(x).__name__
                            for x in output
                        ]
                    else:
                        observed_shapes[name] = type(output).__name__
                return hook
            hooks.append(module.register_forward_hook(make_hook(full_name)))

    with torch.no_grad():
        x = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
        features = model.forward_features(x)
        logits = model.forward_head(features, pre_logits=False)

    for handle in hooks:
        handle.remove()

    grid = tuple(int(v) for v in getattr(model.patch_embed, "grid_size", ()))
    num_prefix_tokens = int(getattr(model, "num_prefix_tokens", -1))
    feature_shape = list(features.shape)

    report = {
        "schema_version": "1.0",
        "purpose": "Pre-science architecture-interface qualification only; no dataset or scientific outputs consumed.",
        "model_id": MODEL_ID,
        "timm_version": timm.__version__,
        "torch_version": torch.__version__,
        "model_class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "block_count": len(model.blocks),
        "final_block_path": prefix,
        "final_block_type": f"{final_block.__class__.__module__}.{final_block.__class__.__qualname__}",
        "final_block_modules": module_inventory,
        "candidate_activation_shapes": observed_shapes,
        "num_prefix_tokens": num_prefix_tokens,
        "patch_grid": list(grid),
        "forward_features_shape": feature_shape,
        "forward_logits_shape": list(logits.shape),
        "token_contract_pass": (
            len(feature_shape) == 3
            and grid == (16, 16)
            and num_prefix_tokens >= 1
            and feature_shape[1] - num_prefix_tokens == 256
        ),
        "science_authorized": False,
    }

    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["token_contract_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
