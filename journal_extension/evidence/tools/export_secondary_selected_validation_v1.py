from __future__ import annotations
import argparse, json, math, hashlib, sys
from pathlib import Path
import _bootstrap  # noqa: F401

from cropcop_je.checkpointing import verify_selected
from cropcop_je.data import CropCopManifestDataset
from cropcop_je.environment import capture_environment, validate_locked_core
from cropcop_je.frozen_v1_manifest import load_frozen_v1_rows
from cropcop_je.hashing import sha256_file
from cropcop_je.models import MNV4_MODEL_NAME
from cropcop_je.persistence import build_store
from cropcop_je.secondary import CLASS_MAP_SHA256, MANIFEST_SHA256, create_empty_baseline
from cropcop_je.train import _identity as checkpoint_identity

VAL_COUNT=16368
SECONDARY_SOURCE_SHA="8904b100d223e4319776199c87ab397db23600ce"
EXPORTER_ID="CROPCOP-SECONDARY-SELECTED-VALIDATION-EXPORT-V1"

def load_json(p): return json.loads(Path(p).read_text(encoding="utf-8"))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-record",required=True)
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--class-map",required=True)
    ap.add_argument("--image-root",required=True)
    ap.add_argument("--output-dir",required=True)
    ap.add_argument("--num-workers",type=int,default=2)
    args=ap.parse_args()

    import timm, torch
    from torch.utils.data import DataLoader

    record=load_json(args.run_record)
    eid=str(record.get("experiment_id",""))
    allowed={"R12-MNV4-LOGITS-S1","R12-MNV4-FEATURE-S1","R06-EFFB0-CONTEXT-S1","R07-CNXTT-CONTEXT-S1"}
    if eid not in allowed: raise SystemExit(f"unauthorized secondary experiment: {eid}")
    if record.get("status")!="PASS" or record.get("continuation_required"):
        raise SystemExit("secondary run is not terminal PASS")
    if record.get("source_git_commit")!=SECONDARY_SOURCE_SHA:
        raise SystemExit("secondary source identity mismatch")
    if record.get("manifest_sha256")!=MANIFEST_SHA256 or record.get("class_map_sha256")!=CLASS_MAP_SHA256:
        raise SystemExit("secondary dataset identity mismatch")
    if record.get("allowed_surfaces")!=["DS-V1-TRAIN","DS-V1-VAL"]:
        raise SystemExit("secondary allowed-surface mismatch")

    env=capture_environment()
    drift=validate_locked_core(env)
    if drift: raise SystemExit("locked software identity mismatch: "+json.dumps(drift,sort_keys=True))
    if not env.get("cuda_available"): raise SystemExit("selected validation export requires CUDA")
    if timm.__version__!="1.0.26": raise SystemExit(f"timm drift: {timm.__version__}")

    locator=record.get("durable_store",{}).get("locator")
    selected_sha=record.get("result_summary",{}).get("selected_checkpoint_sha256")
    if not locator or not selected_sha: raise SystemExit("secondary durable/selected identity missing")
    output=Path(args.output_dir)
    checkpoint_root=output/"restored_checkpoints"
    store=build_store("kaggle-dataset",locator)
    store.restore(checkpoint_root,run_id=record["run_id"])
    selected_path,payload=verify_selected(
        checkpoint_root,
        expected_identity=checkpoint_identity(record),
        expected_sha256=selected_sha,
    )
    if payload.get("identity",{}).get("source_git_commit")!=SECONDARY_SOURCE_SHA:
        raise SystemExit("selected checkpoint source identity mismatch")

    if eid.startswith("R12-"):
        model=timm.create_model(MNV4_MODEL_NAME,pretrained=False,num_classes=120)
        family="mnv4"
    elif eid=="R06-EFFB0-CONTEXT-S1":
        model=create_empty_baseline(model_key="effb0",num_classes=120)
        family="effb0"
    else:
        model=create_empty_baseline(model_key="cnxtt",num_classes=120)
        family="cnxtt"
    model.load_state_dict(payload["student"],strict=True)
    device=torch.device("cuda")
    model.to(device).eval()

    rows=load_frozen_v1_rows(
        args.manifest,args.class_map,
        expected_manifest_sha256=MANIFEST_SHA256,
        expected_class_map_sha256=CLASS_MAP_SHA256,
        surface="DS-V1-VAL",
        row_id_column="record_key",path_column="portable_relpath",
        split_column="split",label_column="label",class_index_column="",
        train_split_value="train",val_split_value="val",
        expected_count=VAL_COUNT,
    )
    dataset=CropCopManifestDataset(rows,args.image_root,training_seed=int(record["seed"]),train=False)
    kw=dict(dataset=dataset,batch_size=16,shuffle=False,num_workers=args.num_workers,pin_memory=True,drop_last=False)
    if args.num_workers>0: kw.update(persistent_workers=True,prefetch_factor=2)
    loader=DataLoader(**kw)

    output.mkdir(parents=True,exist_ok=True)
    pred_path=output/"validation_predictions.jsonl"
    conf=torch.zeros((120,120),dtype=torch.int64)
    nll_sum=0.0;count=0;correct=0
    with pred_path.open("w",encoding="utf-8",newline="\n") as fh:
        with torch.no_grad():
            for x,y,row_ids in loader:
                x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
                logits=model(x)
                logp=torch.log_softmax(logits,dim=1)
                nll=-logp.gather(1,y[:,None]).squeeze(1)
                pred=logits.argmax(dim=1)
                correct+=int((pred==y).sum().item());count+=int(y.numel());nll_sum+=float(nll.sum().item())
                idx=(y*120+pred).detach().cpu()
                conf+=torch.bincount(idx,minlength=120*120).reshape(120,120)
                for rid,t,p,n in zip(row_ids,y.detach().cpu().tolist(),pred.detach().cpu().tolist(),nll.detach().cpu().tolist()):
                    fh.write(json.dumps({
                        "stable_row_id":str(rid),
                        "target_class_index":int(t),
                        "predicted_class_index":int(p),
                        "true_class_nll":float(n),
                    },separators=(",",":"),sort_keys=True)+"\n")
    if count!=VAL_COUNT: raise SystemExit(f"validation count mismatch: {count}")

    tp=conf.diag().to(torch.float64);actual=conf.sum(1).to(torch.float64);predicted=conf.sum(0).to(torch.float64)
    recall=torch.where(actual>0,tp/actual,torch.zeros_like(tp))
    precision=torch.where(predicted>0,tp/predicted,torch.zeros_like(tp))
    f1=torch.where(precision+recall>0,2*precision*recall/(precision+recall),torch.zeros_like(precision))
    metrics={
        "validation_macro_f1":float(f1.mean()),
        "validation_balanced_accuracy":float(recall.mean()),
        "validation_nll":nll_sum/count,
        "validation_accuracy":correct/count,
    }
    frozen=record.get("result_summary",{}).get("selected_metrics",{})
    diffs={k:abs(metrics[k]-float(frozen[k])) for k in metrics}
    tol=1e-6
    if any((not math.isfinite(v)) or v>tol for v in diffs.values()):
        raise SystemExit("selected-checkpoint validation replay differs from frozen metrics: "+json.dumps(diffs,sort_keys=True))

    evidence={
        "schema_version":"1.0",
        "status":"PASS",
        "evidence_exporter_id":EXPORTER_ID,
        "run_id":record["run_id"],
        "experiment_id":eid,
        "model_family":family,
        "secondary_source_sha":SECONDARY_SOURCE_SHA,
        "config_sha256":record["config_sha256"],
        "selected_checkpoint_sha256":selected_sha,
        "selected_checkpoint_verified":True,
        "selected_checkpoint_epoch":int(payload.get("epoch",-1)),
        "manifest_sha256":MANIFEST_SHA256,
        "class_map_sha256":CLASS_MAP_SHA256,
        "validation_surface":"DS-V1-VAL",
        "validation_row_count":count,
        "validation_predictions_sha256":sha256_file(pred_path),
        "validation_predictions_basename":pred_path.name,
        "recomputed_selected_metrics":metrics,
        "frozen_selected_metrics":{k:float(frozen[k]) for k in metrics},
        "absolute_metric_differences":diffs,
        "metric_match_tolerance":tol,
        "training_performed":False,
        "scientific_execution_relaunched":False,
        "optimizer_state_advanced":False,
        "v1_test_accessed":False,
        "protected_external_surface_accessed":False,
        "checkpoint_material_public":False,
        "restored_selected_checkpoint_sha256":sha256_file(selected_path),
    }
    (output/"validation_evidence.json").write_text(json.dumps(evidence,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(evidence,indent=2,sort_keys=True))
    return 0
if __name__=="__main__": raise SystemExit(main())
