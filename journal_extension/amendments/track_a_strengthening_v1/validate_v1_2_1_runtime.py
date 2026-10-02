from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL_ID = "vit_dlittle_patch16_reg1_gap_256.sbb_nadamuon_in1k"
EXPECTED_TARGET = "blocks.13.norm1"
ROOT = Path(__file__).resolve().parents[3]
AMEND = ROOT / "journal_extension" / "amendments" / "track_a_strengthening_v1"
QUALIFICATION = AMEND / "r13_xai_interface_qualification_v1_2_1.json"


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def resolve_module(model, path: str):
    current = model
    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def validate() -> dict:
    errors: list[str] = []
    try:
        import numpy as np
        import timm
        import torch
        import torchvision
    except Exception as exc:
        return {"status": "FAIL", "errors": [f"runtime imports failed: {exc}"], "science_authorized": False}

    require(torch.__version__.split("+", 1)[0] == "2.12.1", f"torch drift: {torch.__version__}", errors)
    require(torchvision.__version__.split("+", 1)[0] == "0.27.1", f"torchvision drift: {torchvision.__version__}", errors)
    require(timm.__version__ == "1.0.26", f"timm drift: {timm.__version__}", errors)
    require(np.__version__ == "2.5.2", f"numpy drift: {np.__version__}", errors)

    qualification = json.loads(QUALIFICATION.read_text(encoding="utf-8"))
    require(qualification.get("scientific_outputs_observed_before_correction") is False, "qualification was not pre-science", errors)
    require(qualification.get("dataset_rows_consumed_by_probe") == 0, "qualification probe consumed dataset rows", errors)
    target = qualification.get("qualified_xai_target", {})
    require(target.get("module_path") == EXPECTED_TARGET, "qualified R13 XAI target drift", errors)
    require(target.get("same_target_across_all_r13_seeds") is True, "R13 target is not seed-invariant", errors)

    model = None
    try:
        model = timm.create_model(MODEL_ID, pretrained=False, num_classes=1000)
    except Exception as exc:
        errors.append(f"R13 construction failed: {exc}")

    observed_target_shape = None
    parity = None
    if model is not None:
        require(hasattr(model, "forward_features"), "R13 missing forward_features", errors)
        require(hasattr(model, "forward_head"), "R13 missing forward_head", errors)
        require(hasattr(model, "patch_embed") and hasattr(model.patch_embed, "proj"), "R13 missing patch_embed.proj", errors)
        require(hasattr(model, "blocks") and len(model.blocks) == 14, f"R13 block inventory drift: {len(model.blocks) if hasattr(model, 'blocks') else 'missing'}", errors)
        require(int(getattr(model, "num_prefix_tokens", -1)) == 1, f"R13 prefix-token drift: {getattr(model, 'num_prefix_tokens', None)}", errors)
        grid = tuple(int(x) for x in getattr(model.patch_embed, "grid_size", ()))
        require(grid == (16, 16), f"R13 patch grid mismatch: {grid}", errors)
        require(getattr(model.patch_embed.proj, "bias", None) is not None, "R13 patch_embed.proj bias absent", errors)

        try:
            module = resolve_module(model, EXPECTED_TARGET)
            require(module.__class__.__module__ == "timm.layers.norm", f"R13 target module namespace drift: {module.__class__.__module__}", errors)
            require(module.__class__.__qualname__ == "LayerNorm", f"R13 target module type drift: {module.__class__.__qualname__}", errors)
            holder: dict[str, object] = {}

            def hook(_module, _inputs, output):
                if isinstance(output, torch.Tensor):
                    holder["shape"] = list(output.shape)

            handle = module.register_forward_hook(hook)
            model.eval()
            with torch.no_grad():
                x = torch.zeros((2, 3, 256, 256), dtype=torch.float32)
                features = model.forward_features(x)
                logits = model.forward_head(features, pre_logits=False)
                prelogits = model.forward_head(features, pre_logits=True)
            handle.remove()
            observed_target_shape = holder.get("shape")
            require(observed_target_shape == [2, 257, 320], f"R13 qualified target shape mismatch: {observed_target_shape}", errors)
            require(list(features.shape) == [2, 257, 320], f"R13 forward_features shape mismatch: {list(features.shape)}", errors)
            require(list(logits.shape) == [2, 1000], f"R13 logits shape mismatch: {list(logits.shape)}", errors)
            require(prelogits.ndim == 2 and prelogits.shape[0] == 2, f"R13 prelogits shape mismatch: {list(prelogits.shape)}", errors)
            require(int(features.shape[1]) - int(model.num_prefix_tokens) == 256, "R13 patch-token count mismatch", errors)
        except Exception as exc:
            errors.append(f"R13 qualified target/forward check failed: {exc}")

        try:
            base = timm.create_model(MODEL_ID, pretrained=False, num_classes=1000)
            conv = base.patch_embed.proj
            native_mean = torch.tensor([0.5, 0.5, 0.5], dtype=conv.weight.dtype)
            native_std = torch.tensor([0.5, 0.5, 0.5], dtype=conv.weight.dtype)
            ctc_mean = torch.tensor([0.485, 0.456, 0.406], dtype=conv.weight.dtype)
            ctc_std = torch.tensor([0.229, 0.224, 0.225], dtype=conv.weight.dtype)
            w_native = conv.weight.detach().clone()
            b_native = conv.bias.detach().clone()
            w_ctc = w_native * (ctc_std / native_std).view(1, 3, 1, 1)
            offset = ((ctc_mean - native_mean) / native_std).view(1, 3, 1, 1)
            b_ctc = b_native + (w_native * offset).sum(dim=(1, 2, 3))
            g = torch.Generator(device="cpu")
            g.manual_seed(120013)
            rgb = torch.rand((4, 3, 256, 256), generator=g)
            native_x = (rgb - native_mean.view(1, 3, 1, 1)) / native_std.view(1, 3, 1, 1)
            ctc_x = (rgb - ctc_mean.view(1, 3, 1, 1)) / ctc_std.view(1, 3, 1, 1)
            y_native = torch.nn.functional.conv2d(native_x, w_native, b_native, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups)
            y_ctc = torch.nn.functional.conv2d(ctc_x, w_ctc, b_ctc, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups)
            parity = float((y_native - y_ctc).abs().max().item())
            require(parity <= 1e-5, f"normalization-equivalence algebra smoke failed: {parity}", errors)
        except Exception as exc:
            errors.append(f"normalization-equivalence smoke failed: {exc}")

    return {
        "schema_version": "1.2.1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "model_id": MODEL_ID,
        "qualified_target": EXPECTED_TARGET,
        "observed_target_shape": observed_target_shape,
        "normalization_equivalence_max_abs_difference": parity,
        "versions": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "timm": timm.__version__,
            "numpy": np.__version__,
        },
        "science_authorized": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    report = validate()
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
