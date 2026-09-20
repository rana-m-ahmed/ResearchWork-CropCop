from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from cropcop_je.atomic_io import atomic_write_json
from cropcop_je.hashing import sha256_file, sha256_json
from cropcop_je.trackb_r07 import (
    AUTHORITY_ID,
    TrackBError,
    discover_kaggle_inputs,
    load_json,
    resolve_bundle_file,
    verify_candidate_seal,
)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _support(rows: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        label = str(row["source_label"])
        out[label] = out.get(label, 0) + 1
    return out


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise TrackBError(f"{label} missing: {path}")
    observed = sha256_file(path)
    if observed != str(expected):
        raise TrackBError(f"{label} SHA mismatch: expected={expected}, observed={observed}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Independent verifier for CropCop Track-B prediction-blind qualification."
    )
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--output-root", required=True)
    args = ap.parse_args()

    output_root = Path(args.output_root).resolve()
    qualification_path = output_root / "TRACKB_PREINFERENCE_QUALIFICATION.json"
    firewall_path = output_root / "TRACKB_PREDICTION_FIREWALL.json"
    if not qualification_path.is_file() or not firewall_path.is_file():
        raise TrackBError("pre-inference qualification/firewall evidence is incomplete")

    qualification = load_json(qualification_path)
    clean = dict(qualification)
    observed_self_hash = str(clean.pop("qualification_sha256", ""))
    if len(observed_self_hash) != 64 or sha256_json(clean) != observed_self_hash:
        raise TrackBError("pre-inference qualification self-hash mismatch")
    if qualification.get("status") != "PASS_PREDICTION_BLIND_QUALIFICATION":
        raise TrackBError("pre-inference qualification is not terminal PASS")
    if qualification.get("authority_id") != AUTHORITY_ID:
        raise TrackBError("pre-inference qualification authority mismatch")
    if int(qualification.get("protected_external_prediction_count", -1)) != 0:
        raise TrackBError("qualification claims protected predictions were produced")
    if qualification.get("v1_test_accessed") is not False:
        raise TrackBError("qualification does not prove V1-test closure")
    if qualification.get("new_training_performed") is not False:
        raise TrackBError("qualification indicates new training")

    inputs = discover_kaggle_inputs(args.input_root)
    core = inputs["core"]
    historical = inputs["historical_compare"]

    if sha256_file(resolve_bundle_file(core, "downstream_authority")) != qualification["downstream_authority_sha256"]:
        raise TrackBError("qualification downstream-authority bytes differ from core")
    if sha256_file(resolve_bundle_file(core, "execution_lock")) != qualification["execution_lock_sha256"]:
        raise TrackBError("qualification execution-lock bytes differ from core")
    if sha256_file(resolve_bundle_file(core, "code_attestation")) != qualification["code_attestation_sha256"]:
        raise TrackBError("qualification code-attestation bytes differ from core")
    if sha256_file(historical.manifest_path) != qualification["historical_compare_input_manifest_sha256"]:
        raise TrackBError("qualification historical-comparison manifest mismatch")
    if historical.manifest.get("coverage_scope") != "V1_TRAIN_VAL_ONLY":
        raise TrackBError("qualification historical comparison is not V1_TRAIN_VAL_ONLY")
    if int(historical.manifest.get("image_count", -1)) != 92744:
        raise TrackBError("qualification historical comparison row count drift")
    if historical.manifest.get("maximum_evidence_grade") != "EXT-S":
        raise TrackBError("qualification historical comparison grade ceiling drift")
    if historical.manifest.get("v1_test_image_bytes_accessed") is not False:
        raise TrackBError("qualification historical comparison touched V1-test bytes")

    firewall = load_json(firewall_path)
    if firewall.get("status") != "PASS":
        raise TrackBError("prediction firewall is not PASS")
    if int(firewall.get("external_predictions_before_this_attempt_firewall", -1)) != 0:
        raise TrackBError("prediction firewall is not prediction-blind")
    if firewall.get("v1_test_accessed") is not False:
        raise TrackBError("prediction firewall does not preserve V1-test closure")
    if sha256_file(firewall_path) != qualification["prediction_firewall_sha256"]:
        raise TrackBError("prediction firewall changed after qualification")

    inference_root = output_root / "inference"
    if inference_root.exists() and any(p.is_file() for p in inference_root.rglob("*")):
        raise TrackBError("protected inference files exist in prediction-blind qualification output")
    unexpected_prediction_files = [
        p for p in output_root.rglob("predictions.jsonl") if p.is_file()
    ]
    if unexpected_prediction_files:
        raise TrackBError(
            "prediction files exist outside the permitted zero-prediction qualification surface: "
            + ", ".join(str(p.relative_to(output_root)) for p in unexpected_prediction_files)
        )
    forbidden_terminal = [
        output_root / "TRACKB_FINAL_CLOSURE.json",
        output_root / "TRACKB_FINAL_QA.json",
        output_root / "TRACKB_SCIENCE_MANIFEST.json",
        output_root / "TRACKB_ATTEMPT_STATE.json",
        output_root / "TRACKB_ATTEMPT_PUBLICATION.json",
    ]
    present_forbidden = [p.name for p in forbidden_terminal if p.exists()]
    if present_forbidden:
        raise TrackBError(
            "qualification output contains protected/terminal claim artifacts: "
            + ", ".join(present_forbidden)
        )

    candidate_qa = {}
    role_by_candidate = {"gvlid_grape": "gvlid_v5", "irish_potato": "irish_potato"}
    for cid in ("gvlid_grape", "irish_potato"):
        expected = (qualification.get("candidates") or {}).get(cid)
        if not isinstance(expected, dict):
            raise TrackBError(f"qualification missing candidate record: {cid}")
        root = output_root / "candidates" / cid
        seal_path = root / "seal.json"
        seal = load_json(seal_path)
        verify_candidate_seal(seal)
        if seal["seal_sha256"] != expected["seal_sha256"]:
            raise TrackBError(f"{cid}: seal identity mismatch")

        bundle = inputs[role_by_candidate[cid]]
        observed_input_manifest_sha = sha256_file(bundle.manifest_path)
        if observed_input_manifest_sha != str(seal["candidate_input_manifest_sha256"]):
            raise TrackBError(f"{cid}: candidate input manifest differs from seal")
        if observed_input_manifest_sha != str(expected["candidate_input_manifest_sha256"]):
            raise TrackBError(f"{cid}: candidate input manifest differs from qualification")
        source_metadata = resolve_bundle_file(bundle, "source_metadata_record")
        observed_source_metadata_sha = sha256_file(source_metadata)
        if observed_source_metadata_sha != str(seal["source_metadata_record_sha256"]):
            raise TrackBError(f"{cid}: source metadata differs from seal")
        if observed_source_metadata_sha != str(expected["source_metadata_record_sha256"]):
            raise TrackBError(f"{cid}: source metadata differs from qualification")

        bindings = {
            "source_manifest_sha256": root / "raw_manifest.csv",
            "family_graph_sha256": root / "families.jsonl",
            "representative_manifest_sha256": root / "sealed_representatives.jsonl",
            "accepted_historical_edges_sha256": root / "accepted_historical_edges.jsonl",
            "exclusion_ledger_sha256": root / "exclusions.jsonl",
            "decode_failure_ledger_sha256": root / "decode_failures.jsonl",
        }
        for field, path in bindings.items():
            expected_hash = str(seal[field])
            if str(expected.get(field)) != expected_hash:
                raise TrackBError(f"{cid}: qualification/seal binding differs: {field}")
            _require_hash(path, expected_hash, f"{cid}/{field}")

        reps = _read_jsonl(root / "sealed_representatives.jsonl")
        observed_support = _support(reps)
        sealed_support = {str(k): int(v) for k, v in seal["family_support"].items()}
        if observed_support != sealed_support or observed_support != {
            str(k): int(v) for k, v in expected["family_support"].items()
        }:
            raise TrackBError(f"{cid}: family support does not independently recompute")
        if seal["grade"] != expected["grade"] or seal["claim_mode"] != expected["claim_mode"]:
            raise TrackBError(f"{cid}: terminal grade/claim-mode mismatch")
        accepted_rows = _read_jsonl(root / "accepted_historical_edges.jsonl")
        if len(accepted_rows) != int(seal["accepted_historical_link_count"]):
            raise TrackBError(f"{cid}: accepted historical-edge ledger count does not match seal")
        if int(seal["accepted_historical_link_count"]) != int(expected["accepted_historical_link_count"]):
            raise TrackBError(f"{cid}: historical-link count mismatch")
        rep_ids = [str(row["representative_row_id"]) for row in reps]
        if len(rep_ids) != len(set(rep_ids)):
            raise TrackBError(f"{cid}: duplicate representative IDs in sealed qualification surface")

        candidate_qa[cid] = {
            "seal_sha256": seal["seal_sha256"],
            "grade": seal["grade"],
            "representative_count": len(reps),
            "family_support": observed_support,
            "accepted_historical_link_count": int(seal["accepted_historical_link_count"]),
        }

    qa = {
        "schema_version": "1.0",
        "status": "PASS_INDEPENDENT_PREINFERENCE_QA",
        "authority_id": AUTHORITY_ID,
        "qualification_sha256": qualification["qualification_sha256"],
        "protected_external_prediction_count": 0,
        "v1_test_accessed": False,
        "new_training_performed": False,
        "candidate_qa": candidate_qa,
    }
    qa["qa_sha256"] = sha256_json(qa)
    atomic_write_json(output_root / "TRACKB_PREINFERENCE_QA.json", qa)
    print(json.dumps(qa, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
