from __future__ import annotations
import fnmatch
TRAIN="DS-V1-TRAIN"; VAL="DS-V1-VAL"; V1_TEST="DS-V1-TEST-CONSUMED"
TRAINING_ALLOWED=frozenset({TRAIN,VAL})
ALWAYS_FORBIDDEN_TRAINING=(V1_TEST,"DS-EXT-*-SEALED","DS-HIST-COMPARE","DS-DEVICE-*")
class SurfaceAuthorizationError(PermissionError): pass
def authorize_training_surface(surface:str)->str:
    if surface not in TRAINING_ALLOWED: raise SurfaceAuthorizationError(f"training entrypoint denies surface {surface!r}; allowed={sorted(TRAINING_ALLOWED)}")
    if any(fnmatch.fnmatchcase(surface,p) for p in ALWAYS_FORBIDDEN_TRAINING): raise SurfaceAuthorizationError(f"protected surface denied: {surface}")
    return surface
def validate_training_config(config:dict)->None:
    if authorize_training_surface(config["train_surface"])!=TRAIN: raise SurfaceAuthorizationError("training surface must be DS-V1-TRAIN")
    if authorize_training_surface(config["validation_surface"])!=VAL: raise SurfaceAuthorizationError("selection surface must be DS-V1-VAL")
    non_denial=repr({k:v for k,v in config.items() if k!="forbidden_surfaces"})
    for literal in ("DS-V1-TEST-CONSUMED","DS-EXT-POTATO-SEALED","DS-EXT-AGRIVISION-SEALED"):
        if literal in non_denial: raise SurfaceAuthorizationError(f"config resolves forbidden surface {literal}")
