from __future__ import annotations

import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.hashing import sha256_file, sha256_json
from cropcop_je.tracka_v12_g1a_v121 import (
    R13_PARITY_CONTRACT_ID,
    R13_PARITY_HISTORICAL_V12_TOLERANCE,
    R13_PARITY_TOLERANCE,
    g1a_seal_hash,
    save_r13_initialization,
    validate_g1a_seal_object,
    validate_v12_g1a_seal_with_v121_tolerance,
)

import seal_tracka_v12_g1a as base  # noqa: E402

CONTRACT_REL = "journal_extension/amendments/track_a_strengthening_v1/r13_pretrained_identity_and_normalization_contract_v1_2_1.json"
EVIDENCE_REL = "journal_extension/amendments/track_a_strengthening_v1/r13_parity_preexecution_evidence_v1_2_1.json"


def _arg_value(name: str) -> str:
    try:
        idx = sys.argv.index(name)
        return sys.argv[idx + 1]
    except (ValueError, IndexError) as exc:
        raise SystemExit(f"versioned G1A sealer requires {name}") from exc


def _version_bundle() -> dict:
    bundle = Path(_arg_value("--bundle-dir")).resolve()
    repo = Path(_arg_value("--repo-root") if "--repo-root" in sys.argv else ".").resolve()
    seal_path = bundle / "TRACKA_V12_G1A_SEAL.json"
    parity_path = bundle / "evidence" / "R13_NORMALIZATION_PARITY.json"
    if not seal_path.is_file() or not parity_path.is_file():
        raise SystemExit("base G1A sealer returned success without required R13 evidence/seal")

    contract_path = repo / CONTRACT_REL
    amendment_evidence_path = repo / EVIDENCE_REL
    if not contract_path.is_file() or not amendment_evidence_path.is_file():
        raise SystemExit("versioned R13 parity contract/evidence missing from authorized source")

    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    parity.update(
        {
            "schema_version": "1.2.1",
            "parity_contract_id": R13_PARITY_CONTRACT_ID,
            "historical_v1_2_required_max_abs_difference": R13_PARITY_HISTORICAL_V12_TOLERANCE,
            "required_max_abs_difference": R13_PARITY_TOLERANCE,
            "parity_contract_sha256": sha256_file(contract_path),
            "preexecution_calibration_evidence_sha256": sha256_file(amendment_evidence_path),
        }
    )
    if float(parity.get("max_abs_difference", 1.0)) > R13_PARITY_TOLERANCE:
        raise SystemExit(
            f"R13 v1.2.1 normalization-equivalence parity failed: {parity.get('max_abs_difference')} > {R13_PARITY_TOLERANCE}"
        )
    atomic_write_json(parity_path, parity)
    parity_sha = sha256_json(parity)

    init_evidence_sha: dict[str, str] = {}
    for label in ("S1", "S2", "S3"):
        path = bundle / "evidence" / f"R13_VIT_DLITTLE_INIT_{label}.json"
        if not path.is_file():
            raise SystemExit(f"R13 {label} initialization evidence missing after base seal")
        row = json.loads(path.read_text(encoding="utf-8"))
        row.update(
            {
                "schema_version": "1.2.1",
                "normalization_parity_contract_id": R13_PARITY_CONTRACT_ID,
                "normalization_parity_required_max_abs": R13_PARITY_TOLERANCE,
                "normalization_parity_historical_v1_2_max_abs": R13_PARITY_HISTORICAL_V12_TOLERANCE,
                "normalization_parity_evidence_sha256": parity_sha,
            }
        )
        atomic_write_json(path, row)
        init_evidence_sha[label] = sha256_json(row)

    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    r13 = seal.setdefault("r13", {})
    r13.update(
        {
            "parity_contract_id": R13_PARITY_CONTRACT_ID,
            "historical_v1_2_required_max_abs_difference": R13_PARITY_HISTORICAL_V12_TOLERANCE,
            "required_max_abs_difference": R13_PARITY_TOLERANCE,
            "parity_contract_sha256": sha256_file(contract_path),
            "parity_preexecution_calibration_evidence_sha256": sha256_file(amendment_evidence_path),
            "parity_evidence_sha256": parity_sha,
        }
    )
    for label, evidence_sha in init_evidence_sha.items():
        r13["states"][label]["init_evidence_sha256"] = evidence_sha
    seal["g1a_seal_sha256"] = g1a_seal_hash(seal)
    errors = validate_g1a_seal_object(seal)
    if errors:
        raise SystemExit("Track-A v1.2.1 G1A self-validation failed: " + "; ".join(errors))
    atomic_write_json(seal_path, seal)
    return seal


def main() -> int:
    # The historical sealer remains untouched. Only this versioned wrapper
    # overrides its imported threshold/function references for the duration of
    # the v1.2.1 build.
    base.R13_PARITY_TOLERANCE = R13_PARITY_TOLERANCE
    base.save_r13_initialization = save_r13_initialization
    base.validate_g1a_seal_object = validate_v12_g1a_seal_with_v121_tolerance
    rc = base.main()
    if rc != 0:
        return int(rc)
    seal = _version_bundle()
    print(json.dumps({"v1_2_1_versioned_g1a_seal": seal}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
