from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AMEND = ROOT / "journal_extension" / "amendments" / "track_a_strengthening_v1"
OLD_PATH = AMEND / "r13_pretrained_identity_and_normalization_contract_v1_2.json"
NEW_PATH = AMEND / "r13_pretrained_identity_and_normalization_contract_v1_2_1.json"
EVIDENCE_PATH = AMEND / "r13_parity_preexecution_evidence_v1_2_1.json"
NOTE_PATH = AMEND / "R13_PARITY_PREEXECUTION_AMENDMENT_v1_2_1.md"
LOCK_PATH = AMEND / "AMENDMENT_V1_2_1_PARITY_CONTENT_LOCK.json"

OLD_ID = "TRACKA-A1-R13-PRETRAINED-IDENTITY-V1.2"
NEW_ID = "TRACKA-A1-R13-PRETRAINED-IDENTITY-V1.2.1"
OLD_TOL = 1e-5
NEW_TOL = 5e-5
OBSERVED = 3.4332275390625e-05
MODEL_SHA = "92ec2d996329be8c9a449d4e38f847e79c34fadc32b6491739bacdaf425ab0ed"
MODEL_BYTES = 90095744


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def git_blob(path: Path) -> str:
    return subprocess.check_output(["git", "hash-object", str(path.relative_to(ROOT))], cwd=ROOT, text=True).strip()


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def scientific_core(contract: dict) -> dict:
    """Fields that must remain identical across the narrow parity amendment."""
    return {
        "ctc_input": contract.get("ctc_input"),
        "expected_model_structure": contract.get("expected_model_structure"),
        "model": contract.get("model"),
        "normalization_equivalence": {
            "bias_formula": contract.get("normalization_equivalence", {}).get("bias_formula"),
            "scope": contract.get("normalization_equivalence", {}).get("scope"),
            "weight_formula": contract.get("normalization_equivalence", {}).get("weight_formula"),
            "zero_or_absent_bias_rule": contract.get("normalization_equivalence", {}).get("zero_or_absent_bias_rule"),
        },
        "parity_inputs": contract.get("parity_gate", {}).get("inputs"),
        "parity_metric": contract.get("parity_gate", {}).get("metric"),
        "parity_sealed_before_science": contract.get("parity_gate", {}).get("result_must_be_sealed_before_science"),
        "training_or_validation_metrics_may_not_be_opened_before_parity_pass": contract.get(
            "training_or_validation_metrics_may_not_be_opened_before_parity_pass"
        ),
    }


def validate() -> dict:
    errors: list[str] = []
    for path in (OLD_PATH, NEW_PATH, EVIDENCE_PATH, NOTE_PATH, LOCK_PATH):
        require(path.is_file(), f"missing parity amendment file: {path.relative_to(ROOT)}", errors)
    if errors:
        return {"schema_version": "1.0", "status": "FAIL", "errors": errors, "science_authorized": False}

    old = load(OLD_PATH)
    new = load(NEW_PATH)
    evidence = load(EVIDENCE_PATH)
    lock = load(LOCK_PATH)

    require(old.get("contract_id") == OLD_ID, "historical v1.2 contract ID drift", errors)
    require(float(old.get("parity_gate", {}).get("required_max_abs_difference", -1)) == OLD_TOL, "historical v1.2 tolerance was modified", errors)
    require(new.get("contract_id") == NEW_ID, "v1.2.1 contract ID mismatch", errors)
    require(new.get("supersedes_contract_id") == OLD_ID, "v1.2.1 supersession link mismatch", errors)
    require(new.get("status") == "PRE_EXECUTION_LOCK", "v1.2.1 contract is not pre-execution locked", errors)
    require(new.get("science_authorized") is False, "v1.2.1 contract may not authorize science", errors)
    require(new.get("scientific_outputs_observed_before_amendment") is False, "amendment claims scientific outputs were observed", errors)
    require(float(new.get("parity_gate", {}).get("original_v1_2_required_max_abs_difference", -1)) == OLD_TOL, "v1.2.1 does not preserve old threshold provenance", errors)
    require(float(new.get("parity_gate", {}).get("required_max_abs_difference", -1)) == NEW_TOL, "v1.2.1 tolerance drift", errors)
    require(scientific_core(old) == scientific_core(new), "scientific core changed beyond parity-bound provenance", errors)

    model = new.get("model", {})
    require(model.get("model_safetensors_sha256") == MODEL_SHA, "R13 pretrained SHA drift", errors)
    require(int(model.get("model_safetensors_bytes", -1)) == MODEL_BYTES, "R13 pretrained byte-count drift", errors)

    unchanged = new.get("unchanged_scientific_semantics", {})
    require(unchanged and all(value is True for value in unchanged.values()), "v1.2.1 unchanged-science declarations incomplete", errors)

    require(evidence.get("status") == "PASS", "parity evidence status is not PASS", errors)
    require(evidence.get("science_authorized") is False, "parity evidence may not authorize science", errors)
    for key in ("scientific_metric_computed", "classifier_prediction_opened", "v1_test_accessed", "external_surface_accessed"):
        require(evidence.get(key) is False, f"parity evidence protected marker invalid: {key}", errors)
    require(evidence.get("model", {}).get("model_safetensors_sha256") == MODEL_SHA, "parity evidence model SHA drift", errors)
    require(int(evidence.get("model", {}).get("model_safetensors_bytes", -1)) == MODEL_BYTES, "parity evidence model bytes drift", errors)
    require(float(evidence.get("original_contract", {}).get("required_max_abs_difference", -1)) == OLD_TOL, "evidence old tolerance mismatch", errors)
    require(float(evidence.get("kaggle_observation", {}).get("observed_max_abs_difference", -1)) == OBSERVED, "Kaggle observed parity value drift", errors)
    require(evidence.get("kaggle_observation", {}).get("old_gate_pass") is False, "Kaggle evidence must show old gate failed", errors)
    require(evidence.get("kaggle_observation", {}).get("scientific_training_started") is False, "Kaggle evidence claims science started", errors)
    diag = evidence.get("independent_exact_pretrained_diagnostic", {})
    require(int(diag.get("github_workflow_run_id", -1)) == 34959233872, "diagnostic run ID drift", errors)
    require(diag.get("exact_kaggle_max_reproduced") is True, "independent diagnostic did not reproduce Kaggle maximum", errors)
    require(float(diag.get("current_implementation", {}).get("max_abs_difference", -1)) == OBSERVED, "independent diagnostic maximum drift", errors)
    for value in diag.get("alternative_parameter_construction_checks", {}).values():
        require(float(value) == OBSERVED, "alternative parameter construction did not reproduce exact maximum", errors)
    require(float(evidence.get("amendment_basis", {}).get("new_required_max_abs_difference", -1)) == NEW_TOL, "evidence new tolerance mismatch", errors)
    require(OLD_TOL < OBSERVED <= NEW_TOL, "observed parity does not justify the narrow threshold supersession", errors)

    require(lock.get("status") == "LOCKED_CONTENT_PENDING_EXACT_HEAD_ATTESTATION", "parity content lock status drift", errors)
    require(lock.get("science_authorized") is False, "parity content lock may not authorize science", errors)
    require(lock.get("pre_science_only") is True, "parity content lock is not pre-science only", errors)
    require(lock.get("r13_scientific_training_started_before_lock") is False, "parity lock claims R13 science already started", errors)
    require(lock.get("protected_surfaces_opened_before_lock") is False, "parity lock claims protected surfaces opened", errors)
    require(float(lock.get("parent_required_max_abs_difference", -1)) == OLD_TOL, "parity lock old threshold mismatch", errors)
    require(float(lock.get("superseding_required_max_abs_difference", -1)) == NEW_TOL, "parity lock new threshold mismatch", errors)
    require(float(lock.get("exact_pretrained_diagnostic", {}).get("observed_max_abs_difference", -1)) == OBSERVED, "parity lock diagnostic value mismatch", errors)

    observed_blobs = {}
    for rel, expected in lock.get("files_git_blob_sha", {}).items():
        path = ROOT / rel
        require(path.is_file(), f"locked amendment file missing: {rel}", errors)
        if path.is_file():
            actual = git_blob(path)
            observed_blobs[rel] = actual
            require(actual == expected, f"parity amendment git-blob drift: {rel}", errors)

    return {
        "schema_version": "1.0",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "old_contract_id": OLD_ID,
        "new_contract_id": NEW_ID,
        "old_required_max_abs_difference": OLD_TOL,
        "new_required_max_abs_difference": NEW_TOL,
        "observed_exact_pretrained_max_abs_difference": OBSERVED,
        "locked_file_blobs": observed_blobs,
        "science_authorized": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    report = validate()
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
