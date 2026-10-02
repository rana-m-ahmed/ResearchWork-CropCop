from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.checkpointing import verify_selected
from cropcop_je.data import CropCopManifestDataset
from cropcop_je.environment import capture_environment, validate_locked_core
from cropcop_je.frozen_v1_manifest import load_frozen_v1_rows
from cropcop_je.hashing import sha256_file
from cropcop_je.models import MNV4_MODEL_NAME
from cropcop_je.persistence import build_store
from cropcop_je.secondary import CLASS_MAP_SHA256, MANIFEST_SHA256, PRINCIPAL_SCIENCE_SOURCE_SHA
from cropcop_je.train import _identity as checkpoint_identity

VAL_COUNT = 16368


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-record", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--class-map", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--row-id-column", required=True)
    ap.add_argument("--path-column", required=True)
    ap.add_argument("--split-column", required=True)
    ap.add_argument("--label-column", required=True)
    ap.add_argument("--class-index-column", default="")
    ap.add_argument("--train-split-value", default="train")
    ap.add_argument("--val-split-value", default="val")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    import timm
    import torch
    from torch.utils.data import DataLoader

    record = load_json(args.run_record)
    run_id = str(record.get("run_id", ""))
    eid = str(record.get("experiment_id", ""))
    if eid not in {
        "R04-MNV4-DIRECT-S1", "R04-MNV4-DIRECT-S2", "R04-MNV4-DIRECT-S3",
        "R05-MNV4-TEACHER-S1", "R05-MNV4-TEACHER-S2", "R05-MNV4-TEACHER-S3",
    }:
        raise SystemExit("run record is not one of the six frozen principal states")
    if record.get("status") != "PASS" or record.get("continuation_required"):
        raise SystemExit("principal run record is not terminal PASS")
    if record.get("source_git_commit") != PRINCIPAL_SCIENCE_SOURCE_SHA:
        raise SystemExit("principal run record does not bind frozen f171309 source")
    if record.get("manifest_sha256") != MANIFEST_SHA256 or record.get("class_map_sha256") != CLASS_MAP_SHA256:
        raise SystemExit("principal run record dataset identity mismatch")

    env = capture_environment()
    drift = validate_locked_core(env)
    if drift:
        raise SystemExit(f"locked software identity mismatch: {json.dumps(drift, sort_keys=True)}")
    if not env.get("cuda_available"):
        raise SystemExit("validation backfill requires CUDA")
    if timm.__version__ != "1.0.26":
        raise SystemExit(f"timm drift: expected 1.0.26, got {timm.__version__}")

    locator = record.get("durable_store", {}).get("locator") or record.get("artifact_locators", {}).get("selected_checkpoint", {}).get("durable_locator")
    if not locator:
        raise SystemExit("principal run record lacks durable checkpoint locator")
    store = build_store("kaggle-dataset", locator)
    output = Path(args.output_dir)
    checkpoint_root = output / "restored_checkpoints"
    store.restore(checkpoint_root, run_id=run_id)

    selected_sha = record.get("result_summary", {}).get("selected_checkpoint_sha256")
    selected_path, payload = verify_selected(
        checkpoint_root,
        expected_identity=checkpoint_identity(record),
        expected_sha256=selected_sha,
    )
    if payload.get("identity", {}).get("source_git_commit") != PRINCIPAL_SCIENCE_SOURCE_SHA:
        raise SystemExit("selected checkpoint payload source identity mismatch")

    model = timm.create_model(MNV4_MODEL_NAME, pretrained=False, num_classes=120)
    model.load_state_dict(payload["student"], strict=True)
    device = torch.device("cuda")
    model.to(device).eval()

    rows = load_frozen_v1_rows(
        args.manifest, args.class_map,
        expected_manifest_sha256=MANIFEST_SHA256,
        expected_class_map_sha256=CLASS_MAP_SHA256,
        surface="DS-V1-VAL",
        row_id_column=args.row_id_column,
        path_column=args.path_column,
        split_column=args.split_column,
        label_column=args.label_column,
        class_index_column=args.class_index_column,
        train_split_value=args.train_split_value,
        val_split_value=args.val_split_value,
        expected_count=VAL_COUNT,
    )
    dataset = CropCopManifestDataset(rows, args.image_root, training_seed=int(record["seed"]), train=False)
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": 16,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": False,
    }
    if args.num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = DataLoader(**loader_kwargs)

    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "validation_predictions.jsonl"
    conf = torch.zeros((120, 120), dtype=torch.int64)
    nll_sum = 0.0
    count = 0
    correct = 0
    with predictions_path.open("w", encoding="utf-8", newline="\n") as fh:
        with torch.no_grad():
            for x, y, row_ids in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                logp = torch.log_softmax(logits, dim=1)
                nll = -logp.gather(1, y[:, None]).squeeze(1)
                pred = logits.argmax(dim=1)
                correct += int((pred == y).sum().item())
                count += int(y.numel())
                nll_sum += float(nll.sum().item())
                idx = (y * 120 + pred).detach().cpu()
                conf += torch.bincount(idx, minlength=120 * 120).reshape(120, 120)
                y_cpu = y.detach().cpu().tolist()
                p_cpu = pred.detach().cpu().tolist()
                nll_cpu = nll.detach().cpu().tolist()
                for row_id, target, prediction, row_nll in zip(row_ids, y_cpu, p_cpu, nll_cpu):
                    fh.write(json.dumps({
                        "stable_row_id": str(row_id),
                        "target_class_index": int(target),
                        "predicted_class_index": int(prediction),
                        "true_class_nll": float(row_nll),
                    }, separators=(",", ":"), sort_keys=True) + "\n")

    if count != VAL_COUNT:
        raise SystemExit(f"validation prediction count mismatch: expected {VAL_COUNT}, got {count}")
    tp = conf.diag().to(torch.float64)
    actual = conf.sum(dim=1).to(torch.float64)
    predicted = conf.sum(dim=0).to(torch.float64)
    recall = torch.where(actual > 0, tp / actual, torch.zeros_like(tp))
    precision = torch.where(predicted > 0, tp / predicted, torch.zeros_like(tp))
    f1 = torch.where(precision + recall > 0, 2 * precision * recall / (precision + recall), torch.zeros_like(precision))
    metrics = {
        "validation_macro_f1": float(f1.mean()),
        "validation_balanced_accuracy": float(recall.mean()),
        "validation_nll": nll_sum / count,
        "validation_accuracy": correct / count,
    }
    selected = record.get("result_summary", {}).get("selected_metrics", {})
    diffs = {key: abs(metrics[key] - float(selected[key])) for key in metrics}
    tolerance = 1e-6
    if any((not math.isfinite(v)) or v > tolerance for v in diffs.values()):
        raise SystemExit("selected-checkpoint validation replay differs from frozen metrics: " + json.dumps(diffs, sort_keys=True))

    evidence = {
        "schema_version": "1.0",
        "status": "PASS",
        "run_id": run_id,
        "experiment_id": eid,
        "principal_science_source_sha": PRINCIPAL_SCIENCE_SOURCE_SHA,
        "selected_checkpoint_sha256": selected_sha,
        "selected_checkpoint_verified": True,
        "selected_checkpoint_epoch": int(payload.get("epoch", -1)),
        "manifest_sha256": MANIFEST_SHA256,
        "class_map_sha256": CLASS_MAP_SHA256,
        "validation_surface": "DS-V1-VAL",
        "validation_row_count": count,
        "validation_predictions_sha256": sha256_file(predictions_path),
        "validation_predictions_basename": predictions_path.name,
        "recomputed_selected_metrics": metrics,
        "frozen_selected_metrics": {key: float(selected[key]) for key in metrics},
        "absolute_metric_differences": diffs,
        "metric_match_tolerance": tolerance,
        "training_performed": False,
        "scientific_execution_relaunched": False,
        "optimizer_state_advanced": False,
        "v1_test_accessed": False,
        "protected_external_surface_accessed": False,
        "checkpoint_material_public": False,
        "restored_selected_checkpoint_sha256": sha256_file(selected_path),
    }
    evidence_path = output / "validation_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
