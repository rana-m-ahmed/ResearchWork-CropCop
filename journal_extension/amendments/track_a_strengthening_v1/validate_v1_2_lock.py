from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AMEND = ROOT / "journal_extension" / "amendments" / "track_a_strengthening_v1"
LOCK = AMEND / "AMENDMENT_V1_2_CONTENT_LOCK.json"
MODEL_ID = "vit_dlittle_patch16_reg1_gap_256.sbb_nadamuon_in1k"
R13_RUNS = {
    "R13-VIT-DLITTLE-DIFF-CONTEXT-S1",
    "R13-VIT-DLITTLE-DIFF-CONTEXT-S2",
    "R13-VIT-DLITTLE-DIFF-CONTEXT-S3",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob(path: Path) -> str:
    return subprocess.check_output(
        ["git", "hash-object", str(path.relative_to(ROOT))], cwd=ROOT, text=True
    ).strip()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def static_validate() -> dict:
    errors: list[str] = []
    lock = load_json(LOCK)
    hashes = {}
    for rel, expected_blob in lock["files_git_blob_sha"].items():
        path = ROOT / rel
        require(path.is_file(), f"missing locked file: {rel}", errors)
        if not path.is_file():
            continue
        observed_blob = git_blob(path)
        observed_sha256 = sha256_file(path)
        hashes[rel] = {"git_blob_sha": observed_blob, "sha256": observed_sha256}
        require(observed_blob == expected_blob, f"git blob drift: {rel}", errors)

    selection = load_json(AMEND / "model_selection_protocol_v1_2.json")
    robustness = load_json(AMEND / "robustness_protocol_v1_2.json")
    xai = load_json(AMEND / "xai_gradcampp_protocol_v1_2.json")
    plan = load_json(AMEND / "parallel_execution_plan_v1_2.json")
    registry = load_json(AMEND / "amendment_registry_v1_2.json")
    analysis = load_json(AMEND / "analysis_protocol_v1_2.json")
    r13 = load_json(AMEND / "r13_pretrained_identity_and_normalization_contract_v1_2.json")

    candidates = selection["candidate_architectures"]
    require(len(candidates) == 4, "model-selection pool must contain exactly four families", errors)
    require({row["family"] for row in candidates} == {"R04", "R06", "R07", "R13"}, "wrong model-selection families", errors)
    require(selection["status"] == "PRE_EXECUTION_LOCK", "model-selection protocol not locked", errors)
    require("No new architecture may enter after the v1.2 pre-execution lock." in selection["anti_fishing"], "missing v1.2 anti-fishing clause", errors)

    expected_direct = {
        f"R04-MNV4-DIRECT-S{i}" for i in (1, 2, 3)
    } | {
        f"R06-EFFB0-CONTEXT-S{i}" for i in (1, 2, 3)
    } | {
        f"R07-CNXTT-CONTEXT-S{i}" for i in (1, 2, 3)
    } | R13_RUNS
    require(set(robustness["models"]) == expected_direct, "robustness model inventory must equal 12 direct states", errors)
    require(set(xai["models"]) == expected_direct, "XAI model inventory must equal 12 direct states", errors)
    require(len(robustness["corruptions"]) == 5, "robustness corruption count drift", errors)

    r13_states = registry["new_r13_training_states"]
    require({row["experiment_id"] for row in r13_states} == R13_RUNS, "R13 run registry mismatch", errors)
    require({int(row["seed"]) for row in r13_states} == {21270083, 606135704, 1153870846}, "R13 seed mismatch", errors)
    require(all(row["objective"] == {"ce": 1.0, "feature": 0.0, "kd": 0.0} for row in r13_states), "R13 objective drift", errors)
    require(registry["candidate_pool_closed_after_this_registry"] is True, "candidate pool not closed", errors)

    require(r13["model"]["model_id"] == MODEL_ID, "R13 model ID mismatch", errors)
    require(r13["model"]["required_timm_version"] == "1.0.26", "R13 timm version drift", errors)
    require(r13["model"]["model_safetensors_sha256"] == "92ec2d996329be8c9a449d4e38f847e79c34fadc32b6491739bacdaf425ab0ed", "R13 upstream weight hash drift", errors)
    require(int(r13["model"]["model_safetensors_bytes"]) == 90095744, "R13 upstream byte-count drift", errors)
    require(r13["model"]["pretraining_dataset"] == "ImageNet-1K", "R13 pretraining dataset drift", errors)
    require(float(r13["parity_gate"]["required_max_abs_difference"]) == 1e-5, "R13 normalization parity tolerance drift", errors)

    require(len(analysis["architecture_three_seed_summary"]["families"]) == 4, "analysis protocol must contain four architecture families", errors)
    require(any("R13" in x for x in analysis["architecture_three_seed_summary"]["families"]), "R13 absent from analysis protocol", errors)

    scheduled = set()
    for section in ("wave_1_existing_amendment", "wave_1_modern_vit", "wave_2"):
        for row in plan[section]:
            for key in ("gpu0", "gpu1"):
                value = row.get(key)
                if isinstance(value, str) and value.startswith("R"):
                    scheduled.add(value)
    require(R13_RUNS.issubset(scheduled), "parallel plan does not schedule all R13 states", errors)

    forbidden_text = "DS-V1-TEST-CONSUMED"
    require(forbidden_text in json.dumps(registry), "protected V1-test prohibition missing from registry", errors)
    require(selection["selection_timing"]["forbidden_before_selection"].count("opening DS-V1-TEST-CONSUMED") == 1, "protected V1-test gate missing from selection protocol", errors)

    return {
        "schema_version": "1.0",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "locked_file_hashes": hashes,
        "candidate_families": [row["family"] for row in candidates],
        "direct_state_count": len(expected_direct),
        "r13_state_count": len(R13_RUNS),
        "science_authorized": False,
    }


def runtime_validate() -> dict:
    errors: list[str] = []
    try:
        import numpy as np
        import timm
        import torch
        import torchvision
    except Exception as exc:
        return {"status": "FAIL", "errors": [f"runtime imports failed: {exc}"]}

    require(torch.__version__.split("+", 1)[0] == "2.12.1", f"torch drift: {torch.__version__}", errors)
    require(torchvision.__version__.split("+", 1)[0] == "0.27.1", f"torchvision drift: {torchvision.__version__}", errors)
    require(timm.__version__ == "1.0.26", f"timm drift: {timm.__version__}", errors)
    require(np.__version__ == "2.5.2", f"numpy drift: {np.__version__}", errors)

    model = None
    try:
        model = timm.create_model(MODEL_ID, pretrained=False, num_classes=1000)
    except Exception as exc:
        errors.append(f"R13 construction failed: {exc}")
    if model is not None:
        require(hasattr(model, "forward_features"), "R13 missing forward_features", errors)
        require(hasattr(model, "forward_head"), "R13 missing forward_head", errors)
        require(hasattr(model, "get_classifier"), "R13 missing get_classifier", errors)
        require(hasattr(model, "reset_classifier"), "R13 missing reset_classifier", errors)
        require(hasattr(model, "patch_embed") and hasattr(model.patch_embed, "proj"), "R13 missing patch_embed.proj", errors)
        require(hasattr(model, "blocks") and len(model.blocks) > 0, "R13 missing blocks", errors)
        if hasattr(model, "blocks") and len(model.blocks) > 0:
            require(hasattr(model.blocks[-1], "in_norm"), "R13 final differential block missing in_norm", errors)
        require(int(getattr(model, "num_prefix_tokens", -1)) >= 1, "R13 prefix-token metadata invalid", errors)
        grid = tuple(int(x) for x in getattr(model.patch_embed, "grid_size", ())) if hasattr(model, "patch_embed") else ()
        require(grid == (16, 16), f"R13 patch grid mismatch: {grid}", errors)
        require(getattr(model.patch_embed.proj, "bias", None) is not None, "R13 patch_embed.proj bias absent; normalization-equivalence contract must fail closed", errors)
        try:
            model.eval()
            with torch.no_grad():
                x = torch.zeros((2, 3, 256, 256), dtype=torch.float32)
                features = model.forward_features(x)
                logits = model.forward_head(features, pre_logits=False)
                prelogits = model.forward_head(features, pre_logits=True)
            require(features.ndim == 3, f"R13 forward_features must be token tensor, got {tuple(features.shape)}", errors)
            require(logits.shape == (2, 1000), f"R13 logits shape mismatch: {tuple(logits.shape)}", errors)
            require(prelogits.ndim == 2 and prelogits.shape[0] == 2, f"R13 prelogits shape mismatch: {tuple(prelogits.shape)}", errors)
            n_tokens = int(features.shape[1])
            require(n_tokens - int(model.num_prefix_tokens) == 256, f"R13 token inventory mismatch: tokens={n_tokens}, prefix={model.num_prefix_tokens}", errors)
        except Exception as exc:
            errors.append(f"R13 forward interface check failed: {exc}")

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
            g = torch.Generator(device="cpu"); g.manual_seed(120013)
            rgb = torch.rand((4, 3, 256, 256), generator=g)
            native_x = (rgb - native_mean.view(1,3,1,1)) / native_std.view(1,3,1,1)
            ctc_x = (rgb - ctc_mean.view(1,3,1,1)) / ctc_std.view(1,3,1,1)
            y_native = torch.nn.functional.conv2d(native_x, w_native, b_native, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups)
            y_ctc = torch.nn.functional.conv2d(ctc_x, w_ctc, b_ctc, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups)
            parity = float((y_native - y_ctc).abs().max().item())
            require(parity <= 1e-5, f"normalization-equivalence algebra smoke failed: {parity}", errors)
        except Exception as exc:
            errors.append(f"normalization-equivalence smoke failed: {exc}")

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "model_id": MODEL_ID,
        "versions": {
            "torch": torch.__version__ if 'torch' in locals() else None,
            "torchvision": torchvision.__version__ if 'torchvision' in locals() else None,
            "timm": timm.__version__ if 'timm' in locals() else None,
            "numpy": np.__version__ if 'np' in locals() else None,
        },
        "science_authorized": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--runtime", action="store_true")
    args = ap.parse_args()
    report = {"static": static_validate()}
    if args.runtime:
        report["runtime"] = runtime_validate()
    overall = all(v.get("status") == "PASS" for v in report.values())
    report["overall_status"] = "PASS" if overall else "FAIL"
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
