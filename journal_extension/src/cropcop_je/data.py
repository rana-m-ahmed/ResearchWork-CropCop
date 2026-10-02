from __future__ import annotations
import csv,hashlib,io,math,random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from .hashing import require_sha256
from .surfaces import TRAIN,VAL,authorize_training_surface

@dataclass(frozen=True)
class ManifestColumns:
    stable_row_id:str; relative_path:str; split:str; class_index:str
@dataclass(frozen=True)
class ManifestRow:
    stable_row_id:str; relative_path:str; split:str; class_index:int

def row_augmentation_seed(training_seed:int,epoch:int,stable_row_id:str)->int:
    return int.from_bytes(hashlib.sha256(f"{training_seed}|{epoch}|{stable_row_id}".encode()).digest()[:8],"big") & 0x7FFF_FFFF_FFFF_FFFF
def epoch_order_seed(training_seed:int,epoch:int)->int:
    return int.from_bytes(hashlib.sha256(f"CropCop|CTC-v2|order|{training_seed}|{epoch}".encode()).digest()[:8],"big") & 0x7FFF_FFFF_FFFF_FFFF

def load_manifest_rows(manifest_path:str|Path,*,expected_sha256:str,surface:str,columns:ManifestColumns,
                       train_split_value:str="train",val_split_value:str="val",expected_count:int|None=None)->list[ManifestRow]:
    surface=authorize_training_surface(surface); require_sha256(manifest_path,expected_sha256,"V1 manifest")
    split_value=train_split_value if surface==TRAIN else val_split_value
    with Path(manifest_path).open(encoding="utf-8",newline="") as f:
        reader=csv.DictReader(f)
        required={columns.stable_row_id,columns.relative_path,columns.split,columns.class_index}
        missing=required.difference(reader.fieldnames or [])
        if missing: raise ValueError(f"manifest is missing required columns: {sorted(missing)}")
        rows=[]; seen=set()
        for raw in reader:
            if raw[columns.split]!=split_value: continue
            row_id=raw[columns.stable_row_id]
            if not row_id or row_id in seen: raise ValueError(f"missing/duplicate stable_row_id on {surface}: {row_id!r}")
            seen.add(row_id); idx=int(raw[columns.class_index])
            if not 0<=idx<120: raise ValueError(f"class index outside frozen 120-way map: {idx}")
            rows.append(ManifestRow(row_id,raw[columns.relative_path],raw[columns.split],idx))
    if expected_count is not None and len(rows)!=expected_count:
        raise ValueError(f"{surface} row count mismatch: expected {expected_count}, got {len(rows)}")
    return rows

def _deps():
    from PIL import Image,ImageOps,ImageStat
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as TF
    return Image,ImageOps,ImageStat,InterpolationMode,TF

def deterministic_pad_resize(image,size:int=256):
    Image,ImageOps,ImageStat,InterpolationMode,TF=_deps()
    image=ImageOps.exif_transpose(image).convert("RGB"); mean=tuple(int(round(v)) for v in ImageStat.Stat(image).mean[:3])
    w,h=image.size; side=max(w,h); left=(side-w)//2; top=(side-h)//2
    image=ImageOps.expand(image,border=(left,top,side-w-left,side-h-top),fill=mean)
    return TF.resize(image,[size,size],interpolation=InterpolationMode.BICUBIC,antialias=True)

def _rrc(image,rng,size=256):
    _,_,_,InterpolationMode,TF=_deps(); width,height=image.size; area=width*height
    for _ in range(10):
        target=area*rng.uniform(.70,1.0); aspect=math.exp(rng.uniform(math.log(.75),math.log(1.33)))
        cw=int(round(math.sqrt(target*aspect))); ch=int(round(math.sqrt(target/aspect)))
        if 0<cw<=width and 0<ch<=height:
            l=rng.randint(0,width-cw); t=rng.randint(0,height-ch)
            return TF.resized_crop(image,t,l,ch,cw,[size,size],interpolation=InterpolationMode.BICUBIC,antialias=True)
    return deterministic_pad_resize(image,size)

def ctc_v2_training_transform(image,*,seed:int,size:int=256):
    import torch
    from torchvision.transforms import functional as TF
    Image,ImageOps,_,InterpolationMode,_=_deps(); rng=random.Random(seed)
    image=ImageOps.exif_transpose(image).convert("RGB")
    image=_rrc(image,rng,size) if rng.random()<.78 else deterministic_pad_resize(image,size)
    if rng.random()<.5: image=TF.hflip(image)
    if rng.random()<.4:
        image=TF.affine(image,angle=rng.uniform(-12,12),
          translate=[int(round(rng.uniform(-.05,.05)*image.size[0])),int(round(rng.uniform(-.05,.05)*image.size[1]))],
          scale=rng.uniform(.9,1.1),shear=[0.,0.],interpolation=InterpolationMode.BILINEAR,fill=0)
    factors={"brightness":rng.uniform(.88,1.12),"contrast":rng.uniform(.88,1.12),
             "saturation":rng.uniform(.92,1.08),"hue":rng.uniform(-.01,.01)}
    order=list(factors); rng.shuffle(order)
    for name in order: image=getattr(TF,f"adjust_{name}")(image,factors[name])
    if rng.random()<.10:
        q=rng.randint(65,100); buf=io.BytesIO(); image.save(buf,format="JPEG",quality=q); buf.seek(0)
        image=Image.open(buf).convert("RGB").copy()
    if rng.random()<.05: image=TF.gaussian_blur(image,[5,5],sigma=[rng.uniform(.1,2.),rng.uniform(.1,2.)])
    x=TF.pil_to_tensor(image).to(torch.float32).div_(255.)
    if rng.random()<.05:
        sigma=rng.uniform(0,.02); g=torch.Generator(device="cpu"); g.manual_seed(seed ^ 0x5A1701)
        x=(x+torch.randn(x.shape,generator=g,dtype=x.dtype)*sigma).clamp_(0.,1.)
    return TF.normalize(x,[.485,.456,.406],[.229,.224,.225])

def ctc_v2_eval_transform(image,size:int=256):
    import torch
    from torchvision.transforms import functional as TF
    x=TF.pil_to_tensor(deterministic_pad_resize(image,size)).to(torch.float32).div_(255.)
    return TF.normalize(x,[.485,.456,.406],[.229,.224,.225])

class CropCopManifestDataset:
    def __init__(self,rows:Iterable[ManifestRow],image_root:str|Path,*,training_seed:int,train:bool):
        self.rows=list(rows); self.image_root=Path(image_root); self.training_seed=int(training_seed); self.train=bool(train); self.epoch=0
    def set_epoch(self,epoch:int): self.epoch=int(epoch)
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        from PIL import Image
        epoch=self.epoch
        if isinstance(index,tuple):
            if len(index)!=2: raise ValueError(f"invalid sampler key: {index!r}")
            index,epoch=index
        row=self.rows[int(index)]; path=(self.image_root/row.relative_path).resolve(); root=self.image_root.resolve()
        if root not in path.parents and path!=root: raise ValueError(f"manifest path escapes image root: {row.relative_path}")
        with Image.open(path) as im:
            x=ctc_v2_training_transform(im,seed=row_augmentation_seed(self.training_seed,int(epoch),row.stable_row_id)) if self.train else ctc_v2_eval_transform(im)
        return x,row.class_index,row.stable_row_id
