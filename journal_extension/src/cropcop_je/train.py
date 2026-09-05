from __future__ import annotations
import math,random,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Sampler
from .data import epoch_order_seed
from .evaluate import evaluate_classifier
from .hashing import sha256_file
from .models import prelogits_and_logits
from .runlog import assert_resume_identity

class EpochPermutationSampler(Sampler[int]):
    def __init__(self,dataset,*,training_seed:int,epoch:int):
        self.dataset=dataset; self.generator=torch.Generator(device="cpu")
        self.generator.manual_seed(epoch_order_seed(training_seed,epoch))
        self.order=torch.randperm(len(dataset),generator=self.generator).tolist()
        self.generator_state=self.generator.get_state()
    def __iter__(self): return iter(self.order)
    def __len__(self): return len(self.order)

def _seed(seed):
    random.seed(seed); np.random.seed(seed%(2**32)); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def build_optimizer(student,projection,ctc):
    new_ids={id(p) for p in student.get_classifier().parameters()}
    if projection is not None: new_ids.update(id(p) for p in projection.parameters())
    groups={}
    for name,p in list(student.named_parameters())+(list(projection.named_parameters()) if projection is not None else []):
        if not p.requires_grad: continue
        scope="new" if id(p) in new_ids else "base"; decay=not(name.endswith(".bias") or p.ndim<=1)
        groups.setdefault((scope,decay),[]).append(p)
    o=ctc["optimizer"]; pg=[]
    for (scope,decay),ps in groups.items():
        pg.append({"params":ps,"lr":o["new_parameter_lr"] if scope=="new" else o["backbone_lr"],
                   "weight_decay":o["weight_decay"] if decay else 0.0})
    return torch.optim.AdamW(pg,betas=tuple(o["betas"]),eps=o["eps"])

def build_scheduler(opt,total_steps,warmup_fraction,min_fraction):
    warm=max(1,round(total_steps*warmup_fraction))
    def f(step):
        if step<warm: return max((step+1)/warm,1e-12)
        p=min(1.0,(step-warm+1)/max(total_steps-warm,1))
        return min_fraction+.5*(1-min_fraction)*(1+math.cos(math.pi*p))
    return torch.optim.lr_scheduler.LambdaLR(opt,f)

def kd_loss(s,t,T): return (T*T)*F.kl_div(F.log_softmax(s/T,dim=1),F.softmax(t/T,dim=1),reduction="batchmean")
def feature_loss(s,t): return (1-F.cosine_similarity(F.normalize(s,p=2,dim=1),F.normalize(t,p=2,dim=1),dim=1)).mean()

def _identity(r):
    keys=("experiment_id","authority_id","config_sha256","manifest_sha256","class_map_sha256","seed","student_init_sha256","pretrained_sha256","teacher_sha256")
    return {k:r.get(k) for k in keys}

def save_checkpoint(path,*,student,projection,optimizer,scheduler,scaler,epoch,batch_in_epoch,optimizer_step,run_identity,selected_metrics=None):
    started=time.perf_counter(); path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"identity":_identity(run_identity),"student":student.state_dict(),
      "projection":projection.state_dict() if projection is not None else None,
      "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),
      "rng":{"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),
             "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
      "epoch":epoch,"batch_in_epoch":batch_in_epoch,"optimizer_step":optimizer_step,"selected_metrics":selected_metrics},path)
    return time.perf_counter()-started

def load_checkpoint(path,*,student,projection,optimizer,scheduler,scaler,run_identity):
    p=torch.load(path,map_location="cpu",weights_only=False); assert_resume_identity(p["identity"],_identity(run_identity))
    student.load_state_dict(p["student"],strict=True)
    if projection is not None:
        if p["projection"] is None: raise ValueError("resume checkpoint lacks teacher projection")
        projection.load_state_dict(p["projection"],strict=True)
    elif p["projection"] is not None: raise ValueError("direct run cannot resume teacher checkpoint")
    optimizer.load_state_dict(p["optimizer"]); scheduler.load_state_dict(p["scheduler"]); scaler.load_state_dict(p["scaler"])
    random.setstate(p["rng"]["python"]); np.random.set_state(p["rng"]["numpy"]); torch.set_rng_state(p["rng"]["torch"])
    if torch.cuda.is_available() and p["rng"]["cuda"] is not None: torch.cuda.set_rng_state_all(p["rng"]["cuda"])
    return p

def run_training(*,student,teacher,projection,train_dataset,val_dataset,ctc,objective,run_identity,output_dir,
                 num_workers=2,resume_path=None,max_optimizer_steps=None,validation_enabled=True,checkpoint_every_steps=250):
    _seed(int(run_identity["seed"])); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type!="cuda": raise RuntimeError("locked Stage-03R path requires qualified CUDA/FP16 host")
    student.to(device)
    if teacher is not None: teacher.to(device).eval()
    if projection is not None: projection.to(device)
    micro=ctc["training"]["micro_batch_size"]; accum=ctc["training"]["gradient_accumulation"]; epochs=ctc["schedule"]["epochs"]
    steps_per_epoch=math.ceil(math.ceil(len(train_dataset)/micro)/accum); total_steps=epochs*steps_per_epoch
    opt=build_optimizer(student,projection,ctc)
    sched=build_scheduler(opt,total_steps,ctc["schedule"]["warmup_fraction_optimizer_steps"],ctc["schedule"]["minimum_lr_fraction"])
    scaler=torch.amp.GradScaler("cuda",enabled=True); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    latest=out/"latest.ckpt"; selected=out/"selected.ckpt"; start_epoch=start_batch=step=0
    if resume_path:
        p=load_checkpoint(Path(resume_path),student=student,projection=projection,optimizer=opt,scheduler=sched,scaler=scaler,run_identity=run_identity)
        start_epoch=p["epoch"]; start_batch=p["batch_in_epoch"]; step=p["optimizer_step"]
    best=None; history=[]; examples=0; save_sec=0.; start=time.perf_counter(); opt.zero_grad(set_to_none=True); torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch,epochs):
        train_dataset.set_epoch(epoch); sampler=EpochPermutationSampler(train_dataset,training_seed=run_identity["seed"],epoch=epoch)
        loader=DataLoader(train_dataset,batch_size=micro,sampler=sampler,num_workers=num_workers,pin_memory=True,drop_last=False)
        student.train()
        for bi,(x,y,_ids) in enumerate(loader):
            if epoch==start_epoch and bi<start_batch: continue
            x=x.to(device); y=y.to(device); examples+=y.numel()
            with torch.amp.autocast("cuda",dtype=torch.float16):
                sf,sl=prelogits_and_logits(student,x); ce=F.cross_entropy(sl,y,label_smoothing=ctc["training"]["label_smoothing"])
                kd=feat=torch.zeros((),device=device)
                if teacher is not None:
                    with torch.no_grad(): tf,tl=prelogits_and_logits(teacher,x)
                    if objective.get("kd",0)>0: kd=kd_loss(sl,tl,ctc["teacher"]["temperature"])
                    if objective.get("feature",0)>0: feat=feature_loss(projection(sf),tf)
                loss=(objective["ce"]*ce+objective.get("kd",0)*kd+objective.get("feature",0)*feat)/accum
            scaler.scale(loss).backward()
            if (bi+1)%accum==0 or bi+1==len(loader):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(list(student.parameters())+(list(projection.parameters()) if projection is not None else []),ctc["training"]["gradient_clip_norm"])
                old=scaler.get_scale(); scaler.step(opt); scaler.update()
                if scaler.get_scale()>=old: sched.step(); step+=1
                opt.zero_grad(set_to_none=True)
                if checkpoint_every_steps and step and step%checkpoint_every_steps==0:
                    save_sec+=save_checkpoint(latest,student=student,projection=projection,optimizer=opt,scheduler=sched,scaler=scaler,epoch=epoch,batch_in_epoch=bi+1,optimizer_step=step,run_identity=run_identity)
                if max_optimizer_steps is not None and step>=max_optimizer_steps:
                    save_sec+=save_checkpoint(latest,student=student,projection=projection,optimizer=opt,scheduler=sched,scaler=scaler,epoch=epoch,batch_in_epoch=bi+1,optimizer_step=step,run_identity=run_identity)
                    elapsed=time.perf_counter()-start
                    return {"mode":"calibration_or_bounded_run","optimizer_steps":step,"examples_seen":examples,"wall_seconds":elapsed,
                            "sec_per_optimizer_step":elapsed/max(step,1),"examples_per_second":examples/max(elapsed,1e-9),
                            "peak_gpu_memory_bytes":int(torch.cuda.max_memory_allocated()),"checkpoint_save_seconds":save_sec,
                            "dataloader_wait_seconds":None,"dataloader_examples_per_wait_second":None,
                            "latest_checkpoint":str(latest),"latest_checkpoint_sha256":sha256_file(latest)}
        start_batch=0
        summary={"epoch":epoch+1}
        if validation_enabled:
            metrics=evaluate_classifier(student,DataLoader(val_dataset,batch_size=micro,shuffle=False,num_workers=num_workers),device)
            summary.update({"validation_macro_f1":metrics.macro_f1,"validation_balanced_accuracy":metrics.balanced_accuracy,
                            "validation_nll":metrics.nll,"validation_accuracy":metrics.accuracy})
            key=metrics.selection_key(epoch+1)
            if best is None or key>best["key"]:
                best={"key":key,"epoch":epoch+1,"metrics":summary.copy()}
                save_sec+=save_checkpoint(selected,student=student,projection=projection,optimizer=opt,scheduler=sched,scaler=scaler,
                    epoch=epoch+1,batch_in_epoch=0,optimizer_step=step,run_identity=run_identity,selected_metrics=best["metrics"])
        history.append(summary)
        save_sec+=save_checkpoint(latest,student=student,projection=projection,optimizer=opt,scheduler=sched,scaler=scaler,
            epoch=epoch+1,batch_in_epoch=0,optimizer_step=step,run_identity=run_identity,selected_metrics=best["metrics"] if best else None)
    if validation_enabled and best is None: raise RuntimeError("30-epoch run completed without validation-selected checkpoint")
    elapsed=time.perf_counter()-start
    return {"mode":"scientific_training","optimizer_steps":step,"examples_seen":examples,"wall_seconds":elapsed,
            "sec_per_optimizer_step":elapsed/max(step,1),"examples_per_second":examples/max(elapsed,1e-9),
            "peak_gpu_memory_bytes":int(torch.cuda.max_memory_allocated()),"checkpoint_save_seconds":save_sec,
            "dataloader_wait_seconds":None,"dataloader_examples_per_wait_second":None,"history":history,
            "selected_epoch":best["epoch"],"selected_metrics":best["metrics"],"selected_checkpoint":str(selected),
            "selected_checkpoint_sha256":sha256_file(selected),"latest_checkpoint":str(latest),"latest_checkpoint_sha256":sha256_file(latest)}
