from __future__ import annotations
import argparse,json
from pathlib import Path
from .hashing import sha256_json
from .runlog import validate_run_record
from .surfaces import validate_training_config

EXPECTED_AUTHORITY_SHA="aab17b65b0873dcb1ecedb061eb02ff60ccb09f8b830184f5e2231a600278f74"
EXPECTED_STAGE04_SHA="a1bc1f8bb2d7a133a46c3f59f83dd2ba8ca3bf0fa8d77ae323553276931bc079"
EXPECTED_SEEDS=[21270083,606135704,1153870846]
EXPECTED_IDS={*(f"R04-MNV4-DIRECT-S{i}" for i in (1,2,3)),*(f"R05-MNV4-TEACHER-S{i}" for i in (1,2,3))}
def _load(path:Path): return json.loads(path.read_text(encoding="utf-8"))

def validate_static(repo_root:Path)->dict:
    errors=[]; je=repo_root/"journal_extension"; authority=_load(je/"locks/scientific_authority.json"); registry=_load(je/"locks/experiment_registry.json")
    if authority.get("authority_sha256")!=EXPECTED_AUTHORITY_SHA: errors.append("03R authority SHA mismatch")
    if authority.get("stage04_sha256")!=EXPECTED_STAGE04_SHA: errors.append("Stage-04 architecture SHA mismatch")
    principal=[x for x in registry["experiments"] if x["experiment_id"] in EXPECTED_IDS]
    if {x["experiment_id"] for x in principal}!=EXPECTED_IDS: errors.append("principal R04/R05 ID set mismatch")
    by_id={x["experiment_id"]:x for x in principal}
    for i,seed in enumerate(EXPECTED_SEEDS,1):
        d=by_id.get(f"R04-MNV4-DIRECT-S{i}",{}); t=by_id.get(f"R05-MNV4-TEACHER-S{i}",{})
        if d.get("seed")!=seed or t.get("seed")!=seed: errors.append(f"seed mismatch for S{i}")
        if d.get("pair_id")!=t.get("pair_id"): errors.append(f"paired identity mismatch for S{i}")
    hashes={}
    for condition,folder in (("direct","r04_direct"),("teacher","r05_teacher")):
        for i in (1,2,3):
            path=je/"configs"/folder/f"s{i}.json"; cfg=_load(path)
            try: validate_training_config(cfg)
            except Exception as exc: errors.append(f"{path}: {exc}")
            expected=f"{'R04-MNV4-DIRECT' if condition=='direct' else 'R05-MNV4-TEACHER'}-S{i}"
            if cfg.get("experiment_id")!=expected: errors.append(f"config experiment ID mismatch: {path}")
            if cfg.get("seed")!=EXPECTED_SEEDS[i-1]: errors.append(f"config seed mismatch: {path}")
            hashes[expected]=sha256_json(cfg)
    for i in (1,2,3):
        d=_load(je/"configs/r04_direct"/f"s{i}.json"); t=_load(je/"configs/r05_teacher"/f"s{i}.json")
        if d["required_student_init_evidence"]!=t["required_student_init_evidence"]: errors.append(f"S{i} pair-init evidence mismatch")
        if d["pair_id"]!=t["pair_id"]: errors.append(f"S{i} pair_id mismatch")
    gitignore=(repo_root/".gitignore").read_text(encoding="utf-8")
    for token in ("*.pt","*.pth","*.ckpt","*.pte","*.safetensors",".env"):
        if token not in gitignore: errors.append(f".gitignore lacks restricted-artifact guard {token}")
    runs=je/"runs"
    if runs.exists():
        for p in runs.glob("*.json"):
            try: validate_run_record(_load(p))
            except Exception as exc: errors.append(f"invalid run record {p.name}: {exc}")
    return {"status":"PASS" if not errors else "FAIL","authority_sha256":authority.get("authority_sha256"),
            "principal_config_sha256":hashes,"errors":errors}

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--repo-root",default="."); ap.add_argument("--json-report",default=""); a=ap.parse_args()
    report=validate_static(Path(a.repo_root).resolve()); text=json.dumps(report,indent=2,sort_keys=True); print(text)
    if a.json_report: Path(a.json_report).write_text(text+"\n",encoding="utf-8")
    return 0 if report["status"]=="PASS" else 1
if __name__=="__main__": raise SystemExit(main())
