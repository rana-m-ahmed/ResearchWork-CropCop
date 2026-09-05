from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.data import CropCopManifestDataset, ManifestColumns, load_manifest_rows
from cropcop_je.hashing import require_sha256, sha256_file, sha256_json
from cropcop_je.models import load_exact_teacher, load_pair_initialization, prelogits_and_logits
from cropcop_je.runlog import environment_summary, write_run_record
from cropcop_je.surfaces import validate_training_config
from cropcop_je.train import run_training

EXPECTED = {"python":"3.12.13","torch":"2.12.1","torchvision":"0.27.1","timm":"1.0.26"}
TRAIN_COUNT = 76376
VAL_COUNT = 16368

def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def require_software_identity() -> dict:
    import torch, torchvision, timm
    observed = {
        "python": platform.python_version(),
        "torch": torch.__version__.split("+",1)[0],
        "torchvision": torchvision.__version__.split("+",1)[0],
        "timm": timm.__version__,
    }
    drift = {k:{"required":v,"observed":observed[k]} for k,v in EXPECTED.items() if observed[k] != v}
    if drift:
        raise RuntimeError(f"locked software identity mismatch: {json.dumps(drift, sort_keys=True)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this host cannot qualify G2 for the locked FP16 path")
    return observed

def build_projection(student, teacher, seed: int):
    import torch
    with torch.no_grad():
        dummy = torch.zeros((1,3,256,256), dtype=torch.float32)
        student_feat,_ = prelogits_and_logits(student,dummy)
        teacher_feat,_ = prelogits_and_logits(teacher,dummy)
    if student_feat.ndim != 2 or teacher_feat.ndim != 2:
        raise RuntimeError(f"pre-classifier features must be 2D; student={tuple(student_feat.shape)}, teacher={tuple(teacher_feat.shape)}")
    torch.manual_seed(int(seed))
    projection = torch.nn.Linear(student_feat.shape[1], teacher_feat.shape[1], bias=False)
    torch.nn.init.xavier_uniform_(projection.weight)
    return projection

def prepare(args):
    config = load_json(args.config)
    validate_training_config(config)
    if args.run_id.strip() == "":
        raise ValueError("run_id must be explicit and non-empty")
    software = require_software_identity()
    ctc_path = Path(args.repo_root) / config["ctc_config"]
    ctc = load_json(ctc_path)
    config_sha = sha256_json(config)
    ctc_sha = sha256_json(ctc)
    manifest_sha = require_sha256(args.manifest, config["manifest_sha256"], "V1 manifest")
    class_map_sha = require_sha256(args.class_map, config["class_map_sha256"], "120-way class map")
    cols = ManifestColumns(args.row_id_column,args.path_column,args.split_column,args.class_index_column)
    train_rows = load_manifest_rows(
        args.manifest, expected_sha256=config["manifest_sha256"], surface="DS-V1-TRAIN",
        columns=cols, train_split_value=args.train_split_value, val_split_value=args.val_split_value,
        expected_count=TRAIN_COUNT)
    val_rows = load_manifest_rows(
        args.manifest, expected_sha256=config["manifest_sha256"], surface="DS-V1-VAL",
        columns=cols, train_split_value=args.train_split_value, val_split_value=args.val_split_value,
        expected_count=VAL_COUNT)
    pair_evidence = load_json(args.pair_init_evidence)
    if pair_evidence.get("pair_id") != config["pair_id"] or int(pair_evidence.get("seed",-1)) != int(config["seed"]):
        raise ValueError("pair-init evidence does not match config pair/seed")
    init_sha = pair_evidence["student_init_sha256"]
    student,init_payload = load_pair_initialization(
        args.pair_init, expected_sha256=init_sha, pair_id=config["pair_id"], seed=config["seed"])
    pretrained_sha = pair_evidence["pretrained_sha256"]
    if init_payload.get("pretrained_sha256") != pretrained_sha:
        raise ValueError("pair-init payload/evidence disagree on pretrained SHA")
    teacher = None
    projection = None
    teacher_sha = None
    if config["condition"] == "teacher":
        if not args.teacher_checkpoint or not args.teacher_factory or not args.teacher_evidence:
            raise ValueError("teacher run requires --teacher-checkpoint, --teacher-factory and --teacher-evidence")
        teacher_evidence = load_json(args.teacher_evidence)
        teacher_sha = require_sha256(args.teacher_checkpoint, config["teacher_checkpoint_sha256"], "historical DINO teacher")
        if teacher_evidence.get("sha256") != teacher_sha:
            raise ValueError("teacher evidence SHA does not match actual teacher bytes")
        if teacher_evidence.get("class_map_sha256") != class_map_sha:
            raise ValueError("teacher evidence does not bind the frozen 120-way class map")
        teacher = load_exact_teacher(args.teacher_checkpoint, factory_spec=args.teacher_factory)
        projection = build_projection(student,teacher,config["seed"])
    train_ds = CropCopManifestDataset(train_rows,args.image_root,training_seed=config["seed"],train=True)
    val_ds = CropCopManifestDataset(val_rows,args.image_root,training_seed=config["seed"],train=False)
    run_identity = {
        "run_id":args.run_id,"experiment_id":config["experiment_id"],"authority_id":config["authority_id"],
        "source_git_commit":args.source_git_commit,"config_sha256":config_sha,"ctc_v2_sha256":ctc_sha,
        "manifest_sha256":manifest_sha,"class_map_sha256":class_map_sha,"seed":int(config["seed"]),
        "student_init_sha256":init_sha,"pretrained_sha256":pretrained_sha,"teacher_sha256":teacher_sha,
        "allowed_surfaces":["DS-V1-TRAIN","DS-V1-VAL"],
    }
    return config,ctc,student,teacher,projection,train_ds,val_ds,run_identity,software

def execute(args, *, max_optimizer_steps=None, resume_path=None, mode="scientific") -> dict:
    config,ctc,student,teacher,projection,train_ds,val_ds,run_identity,software = prepare(args)
    output = Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    run_record_path = output / "run_record.json"
    env = environment_summary()
    record = {**run_identity,"status":"RUNNING","mode":mode,"environment":env,
              "hardware_identity":{"accelerator":env.get("gpu")},"artifact_locators":{}}
    write_run_record(run_record_path,record)
    try:
        result = run_training(
            student=student,teacher=teacher,projection=projection,train_dataset=train_ds,val_dataset=val_ds,
            ctc=ctc,objective=config["objective"],run_identity=run_identity,
            output_dir=output/"private_checkpoints",num_workers=args.num_workers,resume_path=resume_path,
            max_optimizer_steps=max_optimizer_steps,validation_enabled=(mode=="scientific"),
            checkpoint_every_steps=args.checkpoint_every_steps)
        metrics_path = output / "metrics.json"
        metrics_path.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        record.update({
            "status":"PASS",
            "result_summary":{k:result.get(k) for k in (
                "mode","optimizer_steps","examples_seen","wall_seconds","sec_per_optimizer_step",
                "examples_per_second","dataloader_wait_seconds","dataloader_examples_per_wait_second",
                "peak_gpu_memory_bytes","checkpoint_save_seconds","selected_epoch","selected_metrics",
                "selected_checkpoint_sha256","latest_checkpoint_sha256")},
            "artifact_locators":{
                "metrics":{"basename":metrics_path.name,"sha256":sha256_file(metrics_path)},
                "latest_checkpoint":{"basename":Path(result["latest_checkpoint"]).name,
                                     "sha256":result["latest_checkpoint_sha256"],"public_git":False}}})
        if result.get("selected_checkpoint_sha256"):
            record["artifact_locators"]["selected_checkpoint"] = {
                "basename":Path(result["selected_checkpoint"]).name,
                "sha256":result["selected_checkpoint_sha256"],"public_git":False}
        write_run_record(run_record_path,record)
        return record
    except Exception as exc:
        record.update({"status":"FAIL","failure":{"type":type(exc).__name__,"reason":str(exc),
            "technical_retry_allowed_only_if_science_unchanged":True}})
        write_run_record(run_record_path,record)
        raise

def parser() -> argparse.ArgumentParser:
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo-root",default=".")
    ap.add_argument("--config",required=True)
    ap.add_argument("--manifest",required=True)
    ap.add_argument("--class-map",required=True)
    ap.add_argument("--image-root",required=True)
    ap.add_argument("--pair-init",required=True)
    ap.add_argument("--pair-init-evidence",required=True)
    ap.add_argument("--teacher-checkpoint",default="")
    ap.add_argument("--teacher-evidence",default="")
    ap.add_argument("--teacher-factory",default="")
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--source-git-commit",required=True)
    ap.add_argument("--output-dir",required=True)
    ap.add_argument("--row-id-column",required=True)
    ap.add_argument("--path-column",required=True)
    ap.add_argument("--split-column",required=True)
    ap.add_argument("--class-index-column",required=True)
    ap.add_argument("--train-split-value",default="train")
    ap.add_argument("--val-split-value",default="val")
    ap.add_argument("--num-workers",type=int,default=2)
    ap.add_argument("--checkpoint-every-steps",type=int,default=250)
    ap.add_argument("--resume",default="")
    ap.add_argument("--max-optimizer-steps",type=int,default=0)
    ap.add_argument("--mode",choices=["scientific","calibration"],default="scientific")
    return ap

def main() -> int:
    args=parser().parse_args()
    if args.mode=="scientific" and args.max_optimizer_steps:
        raise SystemExit("scientific mode must run the full locked 30 epochs")
    if args.mode=="calibration" and args.max_optimizer_steps<=0:
        raise SystemExit("calibration mode requires --max-optimizer-steps")
    record=execute(args,max_optimizer_steps=(args.max_optimizer_steps or None),
                   resume_path=(args.resume or None),mode=args.mode)
    print(json.dumps(record,indent=2,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
