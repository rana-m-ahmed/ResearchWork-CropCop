from __future__ import annotations
import importlib
from pathlib import Path
from .hashing import require_sha256,sha256_file

MNV4_MODEL_NAME="mobilenetv4_conv_medium.e500_r256_in1k"
TEACHER_SHA256="74b4701b8931976c9227845ead50788ae47e3596f575f2817b7352a715f53b79"

def _state_dict_from_file(path:str|Path):
    import torch
    path=Path(path)
    if path.suffix==".safetensors":
        from safetensors.torch import load_file
        state=load_file(str(path),device="cpu")
    else:
        state=torch.load(path,map_location="cpu",weights_only=False)
    if isinstance(state,dict):
        for key in ("state_dict","model","model_state_dict"):
            if key in state and isinstance(state[key],dict): state=state[key]; break
    if not isinstance(state,dict): raise TypeError("pretrained object does not contain a state_dict")
    if state and all(str(k).startswith("module.") for k in state): state={str(k)[7:]:v for k,v in state.items()}
    return state

def create_student_from_pretrained(pretrained_path:str|Path,*,seed:int,num_classes:int=120):
    import torch,timm
    if timm.__version__!="1.0.26": raise RuntimeError(f"timm version drift: required 1.0.26, got {timm.__version__}")
    model=timm.create_model(MNV4_MODEL_NAME,pretrained=False,num_classes=1000)
    model.load_state_dict(_state_dict_from_file(pretrained_path),strict=True)
    torch.manual_seed(int(seed)); model.reset_classifier(num_classes)
    return model

def save_pair_initialization(model,path:str|Path,*,pair_id:str,seed:int,pretrained_sha256:str):
    import torch
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"schema_version":"1.0","pair_id":pair_id,"seed":int(seed),"model_name":MNV4_MODEL_NAME,
                "num_classes":120,"pretrained_sha256":pretrained_sha256,"model_state":model.state_dict()},path)
    return sha256_file(path)

def load_pair_initialization(path:str|Path,*,expected_sha256:str,pair_id:str,seed:int):
    import timm,torch
    require_sha256(path,expected_sha256,"paired student initialization")
    payload=torch.load(path,map_location="cpu",weights_only=False)
    if payload.get("pair_id")!=pair_id or int(payload.get("seed",-1))!=int(seed):
        raise ValueError("paired student initialization identity mismatch")
    if payload.get("model_name")!=MNV4_MODEL_NAME or int(payload.get("num_classes",-1))!=120:
        raise ValueError("paired student initialization model identity mismatch")
    model=timm.create_model(MNV4_MODEL_NAME,pretrained=False,num_classes=120)
    model.load_state_dict(payload["model_state"],strict=True)
    return model,payload

def load_exact_teacher(checkpoint_path:str|Path,*,factory_spec:str):
    import torch
    require_sha256(checkpoint_path,TEACHER_SHA256,"historical DINO teacher")
    if ":" not in factory_spec: raise ValueError("teacher factory must be specified as module:function")
    module_name,fn_name=factory_spec.split(":",1)
    teacher=getattr(importlib.import_module(module_name),fn_name)(str(checkpoint_path))
    if not isinstance(teacher,torch.nn.Module): raise TypeError("teacher factory did not return torch.nn.Module")
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    return teacher

def prelogits_and_logits(model,x):
    if not hasattr(model,"forward_features") or not hasattr(model,"forward_head"):
        raise TypeError("model adapter must expose forward_features and forward_head")
    features=model.forward_features(x)
    return model.forward_head(features,pre_logits=True),model.forward_head(features,pre_logits=False)
