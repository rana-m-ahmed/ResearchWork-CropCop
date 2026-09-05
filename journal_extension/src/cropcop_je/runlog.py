from __future__ import annotations
import json,platform,sys
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
from .hashing import sha256_json

REQUIRED_RUN_FIELDS={"run_id","experiment_id","authority_id","source_git_commit","config_sha256","manifest_sha256","class_map_sha256","seed","allowed_surfaces","status"}
IDENTITY_FIELDS=("experiment_id","authority_id","config_sha256","manifest_sha256","class_map_sha256","seed","student_init_sha256","pretrained_sha256","teacher_sha256")

def environment_summary()->dict[str,Any]:
    s={"python":sys.version.split()[0],"platform":platform.platform()}
    try:
        import torch
        s.update({"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),"cuda_version":torch.version.cuda,
                  "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    except Exception as e: s["torch_error"]=repr(e)
    try:
        import torchvision; s["torchvision"]=torchvision.__version__
    except Exception as e: s["torchvision_error"]=repr(e)
    try:
        import timm; s["timm"]=timm.__version__
    except Exception as e: s["timm_error"]=repr(e)
    return s

def validate_run_record(record:dict)->None:
    missing=REQUIRED_RUN_FIELDS.difference(record)
    if missing: raise ValueError(f"run record missing required fields: {sorted(missing)}")
    if record["status"] not in {"RUNNING","PASS","FAIL","INCONCLUSIVE","INTERRUPTED"}: raise ValueError(f"invalid run status: {record['status']}")
    if "DS-V1-TEST-CONSUMED" in record.get("allowed_surfaces",[]): raise ValueError("training run cannot authorize V1 test")

def write_run_record(path:str|Path,record:dict)->str:
    record=dict(record); record.setdefault("updated_at_utc",datetime.now(timezone.utc).isoformat()); validate_run_record(record)
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return sha256_json(record)

def assert_resume_identity(saved:dict,current:dict)->None:
    mismatches={f:{"saved":saved.get(f),"current":current.get(f)} for f in IDENTITY_FIELDS if saved.get(f)!=current.get(f)}
    if mismatches: raise ValueError(f"resume identity mismatch: {json.dumps(mismatches,sort_keys=True)}")
