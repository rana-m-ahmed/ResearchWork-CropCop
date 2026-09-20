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
    "gvlid_v5": {
        "doi":"10.17632/wkymf8bhcg.5", "version":"5", "count":3477, "subtree":None,
        "supports":None,
        "required_labels":["Black Rot","Esca","Healthy","Leaf Blight"],
    },
    "irish_potato": {
        "doi":"10.5281/zenodo.8286529", "version":"01", "count":58709, "subtree":None,
        "supports":{"earlyblt":17772,"healthy":20438,"lateblt":20499},
        "required_labels":["earlyblt","healthy","lateblt"],
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
    args=ap.parse_args()
    spec=SPECS[args.role]
    root=Path(args.package_root).resolve(); root.mkdir(parents=True,exist_ok=True)
    data_root=_rel(root,args.data_root,must_dir=True)
    meta_path=_rel(root,args.source_metadata_record,must_file=True)
    meta=json.loads(meta_path.read_text(encoding="utf-8"))
    if str(meta.get("doi")) != spec["doi"] or str(meta.get("version")) != spec["version"]:
        raise TrackBError("source metadata DOI/version mismatch")
    for field in ("retrieved_at","license_or_access_text","source_url","lineage_review_status","lineage_review_id","lineage_review_sha256"):
        if not str(meta.get(field,"" )).strip(): raise TrackBError(f"source metadata missing {field}")
    known=meta.get("known_historical_contributor_relationship")
    if not isinstance(known,bool): raise TrackBError("source metadata must state known_historical_contributor_relationship as boolean")
    lineage=str(meta["lineage_review_status"])
    if lineage not in {"PASS_NO_KNOWN_RELATIONSHIP","RESIDUAL_UNCERTAINTY"}:
        raise TrackBError("unsupported lineage_review_status")
    unresolved=bool(known or lineage != "PASS_NO_KNOWN_RELATIONSHIP")

    lineage_sha=str(meta.get("lineage_review_sha256","")).strip().lower()
    if len(lineage_sha) != 64 or any(ch not in "0123456789abcdef" for ch in lineage_sha):
        raise TrackBError("source metadata lineage_review_sha256 is invalid")
    if args.role == "irish_potato":
        transport=meta.get("acquisition_transport")
        if not isinstance(transport,list) or len(transport) != 3:
            raise TrackBError("Irish Potato acquisition transport must contain exactly three source archives")
        if any(row.get("source_checksum_verified") is not True for row in transport):
            raise TrackBError("Irish Potato source archive checksum verification is incomplete")
    elif args.role == "gvlid_v5":
        integrity=meta.get("source_checksum_integrity")
        if not isinstance(integrity,dict) or integrity.get("status") != "PASS":
            raise TrackBError("GVLiD source checksum ledger verification is not PASS")
        if int(integrity.get("verified_image_count",-1)) != spec["count"]:
            raise TrackBError("GVLiD checksum ledger did not verify all frozen images")

    items=discover_candidate_images(data_root, eligible_subtree=spec["subtree"], allowed_labels=None)
    if len(items) != spec["count"]:
        raise TrackBError(f"{args.role} expected {spec['count']} original images, found {len(items)}")
    supports={}
    for _p,label in items: supports[label]=supports.get(label,0)+1
    required_labels=set(spec["required_labels"])
    if set(supports) != required_labels:
        raise TrackBError(
            f"{args.role} source labels differ from frozen scope: expected {sorted(required_labels)}, got {sorted(supports)}"
        )
    if spec["supports"] is not None:
        for label,count in spec["supports"].items():
            if supports.get(label,0) != count:
                raise TrackBError(f"{args.role} source support mismatch for {label}: expected {count}, got {supports.get(label,0)}")
    else:
        # GVLiD v5 has a published one-image arithmetic discrepancy between total and a displayed
        # class-count table. The bytes, not the inconsistent table, define the sealed support.
        if any(int(supports[label]) < 50 for label in required_labels):
            raise TrackBError(f"{args.role} observed support violates the 50-image pre-audit floor: {supports}")

    files={"source_metadata_record":{"path":meta_path.relative_to(root).as_posix(),"sha256":sha256_file(meta_path),"bytes":meta_path.stat().st_size}}

    manifest={
        "schema_version":"1.0", "role":args.role, "doi":spec["doi"], "version":spec["version"],
        "data_root":data_root.relative_to(root).as_posix(), "unresolved_lineage":unresolved,
        "lineage_review_id":str(meta["lineage_review_id"]),
        "lineage_review_sha256":lineage_sha,
        "source_checksum_verified":True,
        "preflight_original_count":len(items), "preflight_label_support":supports, "files":files,
    }
    atomic_write_json(root/"TRACKB_INPUT_MANIFEST.json",manifest)
    print(json.dumps({"status":"PASS","role":args.role,"unresolved_lineage":unresolved,"manifest":str(root/'TRACKB_INPUT_MANIFEST.json')},indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
