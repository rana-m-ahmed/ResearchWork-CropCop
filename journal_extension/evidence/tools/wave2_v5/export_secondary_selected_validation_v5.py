from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path

import _bootstrap  # runtime-local SHA-bound bootstrap

from cropcop_je.checkpointing import verify_selected
from cropcop_je.data import CropCopManifestDataset
from cropcop_je.environment import capture_environment, validate_locked_core
from cropcop_je.frozen_v1_manifest import load_frozen_v1_rows
from cropcop_je.hashing import sha256_file
from cropcop_je.models import MNV4_MODEL_NAME
from cropcop_je.persistence import build_store
from cropcop_je.secondary import CLASS_MAP_SHA256, MANIFEST_SHA256, create_empty_baseline
from cropcop_je.train import _identity as checkpoint_identity

VAL_COUNT = 16368
SECONDARY_SOURCE_SHA = "8904b100d223e4319776199c87ab397db23600ce"
AUTHORITY_ID = "EAAI-JE-SDL-v2.1-QA"
SEED = 21270083
EXPORTER_ID = "CROPCOP-SECONDARY-SELECTED-VALIDATION-EXPORT-V5"

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def require_equal(name, observed, expected):
    if observed != expected:
        raise SystemExit(f"{name} mismatch: expected={expected!r}, observed={observed!r}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-record", required=True)
    ap.add_argument("--contract", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--class-map", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    import timm
    import torch
    from torch.utils.data import DataLoader

    record = load_json(args.run_record)
    contract = load_json(args.contract)

    required_contract_fields = {
        "schema_version", "run_id", "experiment_id", "lane_id", "envelope_id",
        "original_gpu_slot", "authority_id", "source_git_sha", "seed",
        "config_sha256", "ctc_v2_sha256", "dependency_lock_sha256",
        "g1_seal_sha256", "g2_barrier_sha256", "software_stack_sha256",
        "manifest_sha256", "class_map_sha256", "durable_locator",
        "student_init_sha256", "pretrained_sha256", "teacher_sha256",
        "teacher_factory_sha256", "teacher_factory_bundle_sha256", "model_family",
        "metrics_sha256", "selected_checkpoint_sha256", "selected_epoch", "selected_metrics",
    }
    missing_contract = sorted(required_contract_fields.difference(contract))
    if missing_contract:
        raise SystemExit("validation contract schema incomplete: missing=" + json.dumps(missing_contract))
    if str(contract.get("schema_version")) != "5.0":
        raise SystemExit(f"validation contract schema mismatch: {contract.get('schema_version')!r}")

    eid = str(contract["experiment_id"])
    rid = str(contract["run_id"])
    require_equal("contract authority", contract["authority_id"], AUTHORITY_ID)
    require_equal("contract source", contract["source_git_sha"], SECONDARY_SOURCE_SHA)
    require_equal("contract seed", int(contract["seed"]), SEED)
    require_equal("contract manifest", contract["manifest_sha256"], MANIFEST_SHA256)
    require_equal("contract class map", contract["class_map_sha256"], CLASS_MAP_SHA256)

    require_equal("record run_id", record.get("run_id"), rid)
    require_equal("record experiment_id", record.get("experiment_id"), eid)
    require_equal("record status", record.get("status"), "PASS")
    require_equal("record continuation_required", record.get("continuation_required"), False)
    require_equal("record mode", record.get("mode"), "scientific")
    require_equal("record secondary_track", record.get("secondary_track"), True)
    require_equal("record authority_id", record.get("authority_id"), AUTHORITY_ID)
    require_equal("record source", record.get("source_git_commit"), SECONDARY_SOURCE_SHA)
    require_equal("record lane", record.get("lane_id"), contract["lane_id"])
    require_equal("record seed", int(record.get("seed", -1)), SEED)
    require_equal("record config", record.get("config_sha256"), contract["config_sha256"])
    require_equal("record CTC", record.get("ctc_v2_sha256"), contract["ctc_v2_sha256"])
    require_equal("record dependency lock", record.get("dependency_lock_sha256"), contract["dependency_lock_sha256"])
    require_equal("record G1", record.get("g1_seal_sha256"), contract["g1_seal_sha256"])
    require_equal("record G2", record.get("g2_barrier_sha256"), contract["g2_barrier_sha256"])
    require_equal("record software stack", record.get("software_stack_sha256"), contract["software_stack_sha256"])
    require_equal("record manifest", record.get("manifest_sha256"), MANIFEST_SHA256)
    require_equal("record class map", record.get("class_map_sha256"), CLASS_MAP_SHA256)
    require_equal("record surfaces", record.get("allowed_surfaces"), ["DS-V1-TRAIN", "DS-V1-VAL"])
    require_equal("record selected checkpoint", record.get("result_summary", {}).get("selected_checkpoint_sha256"), contract["selected_checkpoint_sha256"])
    require_equal("record selected epoch", int(record.get("result_summary", {}).get("selected_epoch", -1)), int(contract["selected_epoch"]))
    require_equal("record durable kind", record.get("durable_store", {}).get("kind"), "kaggle-dataset")
    require_equal("record durable required", record.get("durable_store", {}).get("required"), True)
    require_equal("record durable locator", record.get("durable_store", {}).get("locator"), contract["durable_locator"])
    require_equal("record persistence status", record.get("persistence_status", {}).get("status"), "PASS")
    require_equal("record persistence locator", record.get("persistence_status", {}).get("locator"), contract["durable_locator"])
    require_equal("record init", record.get("student_init_sha256"), contract["student_init_sha256"])
    require_equal("record pretrained", record.get("pretrained_sha256"), contract["pretrained_sha256"])
    require_equal("record teacher", record.get("teacher_sha256"), contract.get("teacher_sha256"))
    require_equal("record teacher factory", record.get("teacher_factory_sha256"), contract.get("teacher_factory_sha256"))
    require_equal("record teacher factory bundle", record.get("teacher_factory_bundle_sha256"), contract.get("teacher_factory_bundle_sha256"))

    execution = record.get("environment", {}).get("execution", {})
    require_equal("record envelope", execution.get("CROPCOP_ENVELOPE_ID"), contract["envelope_id"])
    require_equal("record original GPU slot", execution.get("CROPCOP_PHYSICAL_GPU_SLOT"), str(contract["original_gpu_slot"]))

    selected_metrics = record.get("result_summary", {}).get("selected_metrics", {})
    for key, expected in contract["selected_metrics"].items():
        require_equal(f"record selected metric {key}", float(selected_metrics.get(key)), float(expected))

    env = capture_environment()
    drift = validate_locked_core(env)
    if drift:
        raise SystemExit("locked software identity mismatch: " + json.dumps(drift, sort_keys=True))
    if not env.get("cuda_available"):
        raise SystemExit("selected validation export requires CUDA")
    if timm.__version__ != "1.0.26":
        raise SystemExit(f"timm drift: {timm.__version__}")

    output = Path(args.output_dir)
    checkpoint_root = output / "restored_checkpoints"
    if checkpoint_root.exists():
        import shutil
        shutil.rmtree(checkpoint_root)

    store = build_store("kaggle-dataset", contract["durable_locator"])
    restored = store.restore(checkpoint_root, run_id=rid)
    if restored is not True:
        raise SystemExit(f"durable selected-checkpoint restore returned false for {rid}")

    selected_path, payload = verify_selected(
        checkpoint_root,
        expected_identity=checkpoint_identity(record),
        expected_sha256=contract["selected_checkpoint_sha256"],
    )
    require_equal("restored selected checkpoint SHA", sha256_file(selected_path), contract["selected_checkpoint_sha256"])
    require_equal("checkpoint source identity", payload.get("identity", {}).get("source_git_commit"), SECONDARY_SOURCE_SHA)
    require_equal("checkpoint experiment identity", payload.get("identity", {}).get("experiment_id"), eid)
    require_equal("checkpoint lane identity", payload.get("identity", {}).get("lane_id"), contract["lane_id"])

    family = contract["model_family"]
    if family == "mnv4":
        model = timm.create_model(MNV4_MODEL_NAME, pretrained=False, num_classes=120)
    elif family == "effb0":
        model = create_empty_baseline(model_key="effb0", num_classes=120)
    elif family == "cnxtt":
        model = create_empty_baseline(model_key="cnxtt", num_classes=120)
    else:
        raise SystemExit(f"unsupported frozen model family: {family}")
    model.load_state_dict(payload["student"], strict=True)

    device = torch.device("cuda")
    model.to(device).eval()

    rows = load_frozen_v1_rows(
        args.manifest,
        args.class_map,
        expected_manifest_sha256=MANIFEST_SHA256,
        expected_class_map_sha256=CLASS_MAP_SHA256,
        surface="DS-V1-VAL",
        row_id_column="record_key",
        path_column="portable_relpath",
        split_column="split",
        label_column="label",
        class_index_column="",
        train_split_value="train",
        val_split_value="val",
        expected_count=VAL_COUNT,
    )
    dataset = CropCopManifestDataset(rows, args.image_root, training_seed=SEED, train=False)
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
    tmp_predictions = output / "validation_predictions.jsonl.tmp"
    predictions = output / "validation_predictions.jsonl"
    tmp_predictions.unlink(missing_ok=True)
    predictions.unlink(missing_ok=True)

    conf = torch.zeros((120, 120), dtype=torch.int64)
    nll_sum = 0.0
    count = 0
    correct = 0
    row_ids_seen = set()

    with tmp_predictions.open("w", encoding="utf-8", newline="\n") as fh:
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

                for row_id, target, predicted, row_nll in zip(
                    row_ids,
                    y.detach().cpu().tolist(),
                    pred.detach().cpu().tolist(),
                    nll.detach().cpu().tolist(),
                ):
                    sid = str(row_id)
                    if sid in row_ids_seen:
                        raise SystemExit(f"duplicate validation stable_row_id: {sid}")
                    row_ids_seen.add(sid)
                    fh.write(json.dumps({
                        "stable_row_id": sid,
                        "target_class_index": int(target),
                        "predicted_class_index": int(predicted),
                        "true_class_nll": float(row_nll),
                    }, separators=(",", ":"), sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

    if count != VAL_COUNT or len(row_ids_seen) != VAL_COUNT:
        raise SystemExit(f"validation count/row-ID mismatch: count={count}, unique={len(row_ids_seen)}")
    os.replace(tmp_predictions, predictions)

    tp = conf.diag().to(torch.float64)
    actual = conf.sum(1).to(torch.float64)
    predicted = conf.sum(0).to(torch.float64)
    recall = torch.where(actual > 0, tp / actual, torch.zeros_like(tp))
    precision = torch.where(predicted > 0, tp / predicted, torch.zeros_like(tp))
    f1 = torch.where(
        precision + recall > 0,
        2 * precision * recall / (precision + recall),
        torch.zeros_like(precision),
    )
    metrics = {
        "validation_macro_f1": float(f1.mean()),
        "validation_balanced_accuracy": float(recall.mean()),
        "validation_nll": nll_sum / count,
        "validation_accuracy": correct / count,
    }
    diffs = {key: abs(metrics[key] - float(contract["selected_metrics"][key])) for key in metrics}
    tolerance = 1e-6
    if any((not math.isfinite(value)) or value > tolerance for value in diffs.values()):
        raise SystemExit(
            "selected-checkpoint validation replay differs from frozen metrics: "
            + json.dumps(diffs, sort_keys=True)
        )

    evidence = {
        "schema_version": "2.0",
        "status": "PASS",
        "evidence_exporter_id": EXPORTER_ID,
        "run_id": rid,
        "experiment_id": eid,
        "model_family": family,
        "secondary_source_sha": SECONDARY_SOURCE_SHA,
        "authority_id": AUTHORITY_ID,
        "lane_id": contract["lane_id"],
        "source_envelope_id": contract["envelope_id"],
        "config_sha256": contract["config_sha256"],
        "selected_checkpoint_sha256": contract["selected_checkpoint_sha256"],
        "selected_checkpoint_verified": True,
        "checkpoint_payload_epoch_cursor": int(payload.get("epoch", -1)),
        "frozen_selected_epoch": int(contract["selected_epoch"]),
        "manifest_sha256": MANIFEST_SHA256,
        "class_map_sha256": CLASS_MAP_SHA256,
        "validation_surface": "DS-V1-VAL",
        "validation_row_count": count,
        "validation_unique_row_id_count": len(row_ids_seen),
        "validation_predictions_sha256": sha256_file(predictions),
        "validation_predictions_bytes": predictions.stat().st_size,
        "validation_predictions_basename": predictions.name,
        "recomputed_selected_metrics": metrics,
        "frozen_selected_metrics": {k: float(v) for k, v in contract["selected_metrics"].items()},
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
    tmp_evidence = output / "validation_evidence.json.tmp"
    tmp_evidence.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_evidence, evidence_path)
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
