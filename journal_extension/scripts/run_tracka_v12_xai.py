from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.hashing import sha256_file
from cropcop_je.source_state import verify_clean_source
from cropcop_je.tracka_v12_xai import (
    deletion_faithfulness,
    deterministic_randomize_classifier,
    gradcampp,
    image_to_normalized_tensor,
    pil_working_image,
    restore_classifier,
    spearman_flat,
    target_for_family,
    xai_sample_plan,
)
from run_tracka_v12_direct_evidence import load_json, load_model, validation_rows


def family_for_experiment(experiment_id: str) -> str:
    if experiment_id.startswith("R04-"):
        return "R04"
    if experiment_id.startswith("R06-"):
        return "R06"
    if experiment_id.startswith("R07-"):
        return "R07"
    if experiment_id.startswith("R13-"):
        return "R13"
    raise ValueError(f"experiment is not a direct XAI candidate: {experiment_id}")


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--analysis-source-git-commit", required=True)
    ap.add_argument("--run-record", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--checkpoint-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--class-map", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--g1a-bundle", default="")
    ap.add_argument("--secondary-g1-bundle", default="")
    ap.add_argument("--principal-config", default="")
    ap.add_argument("--principal-pair-init", default="")
    ap.add_argument("--principal-pair-evidence", default="")
    ap.add_argument("--durable-store-kind", choices=["", "filesystem", "kaggle-dataset"], default="")
    ap.add_argument("--durable-store-locator", default="")
    ap.add_argument("--row-id-column", default="record_key")
    ap.add_argument("--path-column", default="portable_relpath")
    ap.add_argument("--split-column", default="split")
    ap.add_argument("--label-column", default="label")
    ap.add_argument("--class-index-column", default="")
    ap.add_argument("--train-split-value", default="train")
    ap.add_argument("--val-split-value", default="val")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image, ImageOps

    repo = Path(args.repo_root).resolve()
    output = Path(args.output_dir).resolve()
    analysis_source = args.analysis_source_git_commit.strip()
    if len(analysis_source) != 40:
        raise SystemExit("analysis source Git commit must be a full 40-character SHA")
    verify_clean_source(repo, authorized_source_sha=analysis_source, output_roots=[output])

    run_record = load_json(args.run_record)
    if run_record.get("run_id") != args.run_id or run_record.get("status") != "PASS":
        raise SystemExit("XAI requires the requested terminal PASS scientific run")
    if run_record.get("v1_test_accessed") not in {None, False}:
        raise SystemExit("XAI run record indicates V1-test access")
    if run_record.get("protected_external_surface_accessed") not in {None, False}:
        raise SystemExit("XAI run record indicates protected external-surface access")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA XAI requested but CUDA is unavailable")
    device = torch.device(args.device)

    model, _config, _checkpoint_path, _checkpoint_payload, selected_sha = load_model(args, run_record)
    model = model.to(device).eval()
    rows = validation_rows(args)
    row_by_id = {str(row.stable_row_id): row for row in rows}
    plan = xai_sample_plan(rows)
    family = family_for_experiment(str(run_record["experiment_id"]))
    target_path, target_module, reshape = target_for_family(model, family, device)

    private = output / "private_xai"
    public = output / "public_xai"
    panel_dir = private / "qualitative_panel"
    private.mkdir(parents=True, exist_ok=False)
    public.mkdir(parents=True, exist_ok=False)
    panel_dir.mkdir(parents=True, exist_ok=False)

    analysis_rows = []
    map_cache = {}
    input_cache = {}
    for row_id in plan["sample_240"]:
        row = row_by_id[row_id]
        path = (Path(args.image_root).resolve() / row.relative_path).resolve()
        with Image.open(path) as image:
            working = pil_working_image(image)
        tensor = image_to_normalized_tensor(working).unsqueeze(0).to(device)
        result = gradcampp(model, tensor, target_module, reshape_transform=reshape)
        heatmap = result["heatmap"]
        faithfulness = []
        if heatmap is not None:
            faithfulness = deletion_faithfulness(
                model,
                working,
                heatmap,
                stable_row_id=row_id,
                model_run_id=args.run_id,
                target_class=int(result["target_class_index"]),
                device=device,
            )
            map_cache[row_id] = heatmap.detach().cpu()
            input_cache[row_id] = working.copy()
        analysis_rows.append({
            "stable_row_id": row_id,
            "target_class_index": int(row.class_index),
            "predicted_class_index": int(result["predicted_class_index"]),
            "attribution_target_class_index": int(result["target_class_index"]),
            "top1_confidence": float(result["target_probability"]),
            "finite_map": not bool(result["nonfinite"]),
            "degenerate_map": bool(result["degenerate"]),
            "faithfulness": faithfulness,
        })

    baseline_by_id = {row["stable_row_id"]: row for row in analysis_rows}
    randomization = []
    seed = int.from_bytes(hashlib.sha256(f"{args.run_id}|XAI-RANDOMIZE".encode()).digest()[:8], "big")
    head, original_state = deterministic_randomize_classifier(model, seed=seed)
    try:
        for row_id in plan["randomization_30"]:
            baseline = baseline_by_id[row_id]
            if row_id not in map_cache:
                randomization.append({"stable_row_id": row_id, "spearman": None, "status": "BASELINE_MAP_UNAVAILABLE"})
                continue
            tensor = image_to_normalized_tensor(input_cache[row_id]).unsqueeze(0).to(device)
            randomized = gradcampp(model, tensor, target_module, target_class=int(baseline["attribution_target_class_index"]), reshape_transform=reshape)
            if randomized["heatmap"] is None:
                correlation = None
                status = "RANDOMIZED_MAP_UNAVAILABLE"
            else:
                correlation = spearman_flat(map_cache[row_id].numpy(), randomized["heatmap"].detach().cpu().numpy())
                status = "PASS"
            randomization.append({"stable_row_id": row_id, "spearman": correlation, "status": status})
    finally:
        restore_classifier(head, original_state)

    flip = []
    for row_id in plan["flip_30"]:
        baseline = baseline_by_id[row_id]
        if row_id not in map_cache:
            flip.append({"stable_row_id": row_id, "spearman": None, "status": "BASELINE_MAP_UNAVAILABLE"})
            continue
        flipped_image = ImageOps.mirror(input_cache[row_id])
        tensor = image_to_normalized_tensor(flipped_image).unsqueeze(0).to(device)
        flipped = gradcampp(model, tensor, target_module, target_class=int(baseline["attribution_target_class_index"]), reshape_transform=reshape)
        if flipped["heatmap"] is None:
            correlation = None
            status = "FLIPPED_MAP_UNAVAILABLE"
        else:
            inverse = torch.flip(flipped["heatmap"].detach().cpu(), dims=[1])
            correlation = spearman_flat(map_cache[row_id].numpy(), inverse.numpy())
            status = "PASS"
        flip.append({"stable_row_id": row_id, "spearman": correlation, "status": status})

    panel_manifest = []
    for row_id in plan["panel_12"]:
        if row_id not in map_cache:
            panel_manifest.append({"stable_row_id": row_id, "status": "MAP_UNAVAILABLE"})
            continue
        image = input_cache[row_id]
        heat = np.asarray(map_cache[row_id], dtype=np.float32)
        base = np.asarray(image, dtype=np.float32)
        overlay = base.copy()
        overlay[..., 0] = np.clip(0.55 * base[..., 0] + 0.45 * heat * 255.0, 0, 255)
        overlay[..., 1] = np.clip(0.70 * base[..., 1], 0, 255)
        overlay[..., 2] = np.clip(0.70 * base[..., 2], 0, 255)
        panel = Image.fromarray(overlay.astype(np.uint8), mode="RGB")
        panel_path = panel_dir / f"{row_id}.png"
        panel.save(panel_path, format="PNG", optimize=False)
        panel_manifest.append({"stable_row_id": row_id, "status": "PASS", "sha256": sha256_file(panel_path)})

    rows_path = private / "xai_rows.jsonl"
    with rows_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in analysis_rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    random_path = private / "classifier_randomization.json"
    flip_path = private / "flip_consistency.json"
    panel_manifest_path = private / "qualitative_panel_manifest.json"
    atomic_write_json(random_path, randomization)
    atomic_write_json(flip_path, flip)
    atomic_write_json(panel_manifest_path, panel_manifest)

    finite_count = sum(bool(row["finite_map"]) for row in analysis_rows)
    degenerate_count = sum(bool(row["degenerate_map"]) for row in analysis_rows)
    advantages = [cell["saliency_minus_random_confidence_drop_advantage"] for row in analysis_rows for cell in row["faithfulness"]]
    random_corr = [row["spearman"] for row in randomization if row["spearman"] is not None]
    flip_corr = [row["spearman"] for row in flip if row["spearman"] is not None]
    scientific_source = str(run_record["source_git_commit"])
    summary = {
        "schema_version": "1.2.4",
        "status": "PASS" if finite_count == len(analysis_rows) else "WARNING_NONFINITE_MAPS",
        "experiment_id": run_record["experiment_id"],
        "run_id": run_record["run_id"],
        "source_git_commit": scientific_source,
        "scientific_source_git_commit": scientific_source,
        "analysis_source_git_commit": analysis_source,
        "family": family,
        "selected_checkpoint_sha256": selected_sha,
        "method": "Grad-CAM++",
        "target_module_path": target_path,
        "sample_count": len(analysis_rows),
        "finite_map_rate": finite_count / len(analysis_rows),
        "degenerate_map_rate": degenerate_count / len(analysis_rows),
        "mean_saliency_minus_random_confidence_drop_advantage": mean(advantages),
        "classifier_randomization_mean_spearman": mean(random_corr),
        "flip_consistency_mean_spearman": mean(flip_corr),
        "randomization_subset_count": len(randomization),
        "flip_subset_count": len(flip),
        "qualitative_panel_count": len(panel_manifest),
        "training_or_adaptation_performed": False,
        "xai_used_as_weighted_selector": False,
        "v1_test_accessed": False,
        "external_surface_accessed": False,
        "private_evidence_sha256": {
            "xai_rows.jsonl": sha256_file(rows_path),
            "classifier_randomization.json": sha256_file(random_path),
            "flip_consistency.json": sha256_file(flip_path),
            "qualitative_panel_manifest.json": sha256_file(panel_manifest_path),
        },
        "sample_plan": plan,
    }
    atomic_write_json(public / "XAI_EVIDENCE_GATE.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
