from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.hashing import sha256_file
from cropcop_je.trackb_r07 import TrackBError
from cropcop_je.trackb_r07_audit import discover_candidate_images

SPECS = {
    "irish_potato": {
        "doi":"10.5281/zenodo.8286529", "version":"01", "count":58709, "subtree":None,
        "supports":{"earlyblt":17772,"healthy":20438,"lateblt":20499},
    },
    "agrivision_v2": {
        "doi":"10.17632/8t6k37ztxc.2", "version":"2", "count":5266, "subtree":"Original_Images",
        "supports":{"Tomato Healthy":288,"Tomato Mosaic":195,"Papaya Healthy Leaf":189},
    },
}


def _rel(root: Path, value: str, *, must_file=False, must_dir=False) -> Path:
    p=(root/value).resolve()
    if root not in p.parents and p != root: raise TrackBError(f"path escapes package root: {value}")
    if must_file and not p.is_file(): raise FileNotFoundError(p)
    if must_dir and not p.is_dir(): raise FileNotFoundError(p)
    return p


def main() -> int:
    ap=argparse.ArgumentParser(description="Validate and seal a raw Track-B public-candidate Kaggle input package.")
    ap.add_argument("--role", choices=sorted(SPECS), required=True)
    ap.add_argument("--package-root", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--source-metadata-record", required=True)
    ap.add_argument("--mapping-semantic-record")
    args=ap.parse_args()
    spec=SPECS[args.role]
    root=Path(args.package_root).resolve(); root.mkdir(parents=True,exist_ok=True)
    data_root=_rel(root,args.data_root,must_dir=True)
    meta_path=_rel(root,args.source_metadata_record,must_file=True)
    meta=json.loads(meta_path.read_text(encoding="utf-8"))
    if str(meta.get("doi")) != spec["doi"] or str(meta.get("version")) != spec["version"]:
        raise TrackBError("source metadata DOI/version mismatch")
    for field in ("retrieved_at","license_or_access_text","source_url","lineage_review_status"):
        if not str(meta.get(field,"" )).strip(): raise TrackBError(f"source metadata missing {field}")
    known=meta.get("known_historical_contributor_relationship")
    if not isinstance(known,bool): raise TrackBError("source metadata must state known_historical_contributor_relationship as boolean")
    lineage=str(meta["lineage_review_status"])
    if lineage not in {"PASS_NO_KNOWN_RELATIONSHIP","RESIDUAL_UNCERTAINTY"}:
        raise TrackBError("unsupported lineage_review_status")
    unresolved=bool(known or lineage != "PASS_NO_KNOWN_RELATIONSHIP")

    items=discover_candidate_images(data_root, eligible_subtree=spec["subtree"], allowed_labels=None)
    if len(items) != spec["count"]:
        raise TrackBError(f"{args.role} expected {spec['count']} original images, found {len(items)}")
    supports={}
    for _p,label in items: supports[label]=supports.get(label,0)+1
    for label,count in spec["supports"].items():
        if supports.get(label,0) != count:
            raise TrackBError(f"{args.role} source support mismatch for {label}: expected {count}, got {supports.get(label,0)}")

    files={"source_metadata_record":{"path":meta_path.relative_to(root).as_posix(),"sha256":sha256_file(meta_path),"bytes":meta_path.stat().st_size}}
    if args.role == "agrivision_v2":
        if not args.mapping_semantic_record: raise TrackBError("Agri-Vision v2 requires --mapping-semantic-record")
        sem_path=_rel(root,args.mapping_semantic_record,must_file=True)
        sem=json.loads(sem_path.read_text(encoding="utf-8"))
        if (
            sem.get("status") not in {"PASS", "FAIL"} or sem.get("mapping") != "Tomato Mosaic -> tomato_mosaic_virus"
            or not str(sem.get("evidence_source", "")).strip() or not str(sem.get("rationale", "")).strip()
            or not str(sem.get("verified_at", "")).strip()
        ):
            raise TrackBError("Tomato Mosaic semantic record is incomplete or changes the frozen mapping candidate")
        files["mapping_semantic_record"]={"path":sem_path.relative_to(root).as_posix(),"sha256":sha256_file(sem_path),"bytes":sem_path.stat().st_size}

    manifest={
        "schema_version":"1.0", "role":args.role, "doi":spec["doi"], "version":spec["version"],
        "data_root":data_root.relative_to(root).as_posix(), "unresolved_lineage":unresolved,
        "preflight_original_count":len(items), "preflight_label_support":supports, "files":files,
    }
    atomic_write_json(root/"TRACKB_INPUT_MANIFEST.json",manifest)
    print(json.dumps({"status":"PASS","role":args.role,"unresolved_lineage":unresolved,"manifest":str(root/'TRACKB_INPUT_MANIFEST.json')},indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
