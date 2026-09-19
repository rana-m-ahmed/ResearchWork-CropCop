from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .hashing import require_sha256, sha256_file, sha256_json

AUTHORITY_ID = "EAAI-JE-TRACKBC-R07-DOWNSTREAM-v2"
EXECUTION_LOCK_ID = "TRACKB_R07_EXECUTION_LOCK_v2"
TRACK_A_CLOSURE_COMMIT = "604aafd51e20e70098ce4af647e90c8ff558a9e8"
TRACK_A_FINAL_AUDIT_SELF_HASH = "1c7d98fa47a12ae6eaa53e1c6e91b4e6eef2d04bb717c2d58c7ae2ebae2b51c6"
DATASET_MANIFEST_SHA256 = "bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2"
CLASS_MAP_SHA256 = "46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2"
DINO_AUDIT_SHA256 = "74b4701b8931976c9227845ead50788ae47e3596f575f2817b7352a715f53b79"
DINO_FACTORY_MANIFEST_SHA256 = "df70164ef227878353dde5430e8e0386b8853b53a2b66c20602e4cecd4dab7f1"
R07_RUN_RECORDS = {
    "S1": {
        "run_id": "JE-R07-CNXTT-CONTEXT-S1-8904b100d223-A01",
        "sha256": "0f403138ee43b1f0e464f092b51cf4f80a13233c9bc8ed4b17e86cf7446c5516",
    },
    "S2": {
        "run_id": "JE-R07-CNXTT-CONTEXT-S2-56023042e577-A01",
        "sha256": "0e48fdc0907a44042110f146ec27f796ea3d185fad1e2fc008cd1332bfd0ae15",
    },
    "S3": {
        "run_id": "JE-R07-CNXTT-CONTEXT-S3-56023042e577-A01",
        "sha256": "6ba1f3a7348f5c4ba0347621edef0e75e311b89bd38ff13ec4288cd81cf70050",
    },
}
REPLAY_TOLERANCE = 1e-6
CTC_V2_CONFIG_SHA256 = "53937a6d8e87d18b7de086ecd1c000700d946770c523e50bb85cf124048764c4"
R07_CHECKPOINTS = {
    "S1": "dc7fea2e8db91bf1fc023cb5e792b23b67659edec22e10a7c1d46b4010db3974",
    "S2": "199afb9f7043e599fbb2239fb3babcfb431324a3c3219250ae6dd0359d8bc310",
    "S3": "621c2e6cfecd23da21b4f17d2244bd068b5a3602ee5240f0dcc95ae4360beb37",
}
REQUIRED_INPUT_ROLES = {"core", "historical_compare", "gvlid_v5", "irish_potato"}
FORBIDDEN_SURFACE = "DS-V1-TEST-CONSUMED"

CANDIDATE_CONTRACTS = {
    "gvlid_grape": {
        "scope": "SCOPE-GRAPE-4",
        "doi": "10.17632/wkymf8bhcg.5",
        "version": "5",
        "mapping": {
            "Black Rot": "grape_black_rot",
            "Esca": "grape_esca",
            "Healthy": "grape_healthy",
            "Leaf Blight": "grape_leaf_blight",
        },
    },
    "irish_potato": {
        "scope": "SCOPE-POTATO-3",
        "doi": "10.5281/zenodo.8286529",
        "version": "01",
        "mapping": {
            "earlyblt": "potato_early_blight",
            "healthy": "potato_healthy",
            "lateblt": "potato_late_blight",
        },
    },
}


class TrackBError(RuntimeError):
    pass


@dataclass(frozen=True)
class CandidateGrade:
    grade: str
    claim_mode: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class InputBundle:
    role: str
    root: Path
    manifest_path: Path
    manifest: dict[str, Any]


def load_json(path: str | Path) -> dict[str, Any]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TrackBError(f"JSON object required: {path}")
    return obj


def git_blob_sha1(path: str | Path) -> str:
    data = Path(path).read_bytes()
    header = f"blob {len(data)}\0".encode("utf-8")
    return hashlib.sha1(header + data).hexdigest()


def verify_code_attestation(repo_root: str | Path, attestation_path: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    attestation = load_json(attestation_path)
    if attestation.get("attestation_id") != "TRACKB_CODE_ATTESTATION_v2":
        raise TrackBError("Track-B code attestation identity mismatch")
    if attestation.get("parent_track_a_closure_commit") != TRACK_A_CLOSURE_COMMIT:
        raise TrackBError("Track-B code attestation parent-closure mismatch")
    rows = attestation.get("files")
    if not isinstance(rows, list) or not rows:
        raise TrackBError("Track-B code attestation file list is empty")
    seen = set()
    for row in rows:
        rel = str(row.get("path", "")).strip().replace("\\", "/")
        expected = str(row.get("git_blob_sha1", "")).strip().lower()
        if not rel or rel in seen or len(expected) != 40:
            raise TrackBError(f"invalid/duplicate Track-B code-attestation row: {rel!r}")
        seen.add(rel)
        path = (root / rel).resolve()
        if root not in path.parents and path != root:
            raise TrackBError(f"attested code path escapes repository root: {rel}")
        if not path.is_file():
            raise TrackBError(f"attested code file missing: {rel}")
        actual = git_blob_sha1(path)
        if actual != expected:
            raise TrackBError(f"attested code drift: {rel}: expected {expected}, got {actual}")
    return attestation


def validate_downstream_authority(authority: dict[str, Any]) -> None:
    if authority.get("authority_id") != AUTHORITY_ID or authority.get("status") not in {"ACTIVE", "ACTIVE_PRE_EXECUTION"}:
        raise TrackBError("Track-B/C downstream amendment is missing or inactive")
    if authority.get("parent_track_a_closure_commit") != TRACK_A_CLOSURE_COMMIT:
        raise TrackBError("downstream amendment is not bound to the formal Track-A closure")
    if authority.get("parent_track_a_final_audit_self_hash") != TRACK_A_FINAL_AUDIT_SELF_HASH:
        raise TrackBError("downstream amendment Track-A final-audit binding mismatch")
    dataset = authority.get("dataset", {})
    if dataset.get("manifest_sha256") != DATASET_MANIFEST_SHA256:
        raise TrackBError("downstream amendment dataset manifest identity mismatch")
    if dataset.get("class_map_sha256") != CLASS_MAP_SHA256:
        raise TrackBError("downstream amendment class-map identity mismatch")
    if dataset.get("v1_test_status") != "CLOSED_FOR_TRACK_B":
        raise TrackBError("consumed V1 test is not explicitly closed for Track B")
    states = authority.get("track_b_classifier_states", {})
    for seed, expected in R07_CHECKPOINTS.items():
        row = states.get(f"R07-{seed}", {})
        if row.get("checkpoint_sha256") != expected:
            raise TrackBError(f"R07 {seed} checkpoint authority mismatch")
    if authority.get("audit_encoder", {}).get("checkpoint_sha256") != DINO_AUDIT_SHA256:
        raise TrackBError("DINO audit encoder identity mismatch")
    chronology = authority.get("chronology", {})
    if chronology.get("protected_external_predictions_before_v2") is not False:
        raise TrackBError("v2 cohort redesign does not predate protected external predictions")
    if chronology.get("external_metric_results_used_to_choose_v2_candidates") is not False:
        raise TrackBError("v2 cohort redesign is not explicitly pre-external/outcome-blind")
    redesign = authority.get("cohort_redesign", {})
    if (redesign.get("primary_confirmatory_candidate") or {}).get("id") != "gvlid_grape":
        raise TrackBError("v2 confirmatory cohort identity drift")
    if (redesign.get("complementary_stress_candidate") or {}).get("id") != "irish_potato":
        raise TrackBError("v2 stress cohort identity drift")
    if (redesign.get("retired_candidate") or {}).get("id") != "agrivision_bd":
        raise TrackBError("v2 retired-candidate chronology is missing")
    if redesign.get("post_prediction_candidate_substitution_forbidden") is not True:
        raise TrackBError("v2 candidate substitution firewall is missing")
    track_c = authority.get("track_c_deployment_representative", {})
    if track_c.get("state") != "R07-S1" or track_c.get("checkpoint_sha256") != R07_CHECKPOINTS["S1"]:
        raise TrackBError("shared downstream amendment does not freeze the independent Track-C R07-S1 representative")
    if track_c.get("track_b_results_may_change_representative") is not False:
        raise TrackBError("Track-B outcomes are not explicitly barred from changing the Track-C representative")


def validate_execution_lock(lock: dict[str, Any]) -> None:
    if lock.get("lock_id") != EXECUTION_LOCK_ID or lock.get("status") not in {"PRE_EXECUTION_LOCK", "PRE_EXECUTION_LOCK_V2"}:
        raise TrackBError("Track-B execution lock is absent or invalid")
    if lock.get("authority_id") != AUTHORITY_ID:
        raise TrackBError("execution lock/downstream authority mismatch")
    identities = lock.get("identities", {})
    expected = {
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "class_map_sha256": CLASS_MAP_SHA256,
        "dino_audit_encoder_sha256": DINO_AUDIT_SHA256,
        "dino_factory_manifest_sha256": DINO_FACTORY_MANIFEST_SHA256,
        "r07_s1_sha256": R07_CHECKPOINTS["S1"],
        "r07_s2_sha256": R07_CHECKPOINTS["S2"],
        "r07_s3_sha256": R07_CHECKPOINTS["S3"],
        "r07_s1_run_record_sha256": R07_RUN_RECORDS["S1"]["sha256"],
        "r07_s2_run_record_sha256": R07_RUN_RECORDS["S2"]["sha256"],
        "r07_s3_run_record_sha256": R07_RUN_RECORDS["S3"]["sha256"],
        "ctc_v2_config_sha256": CTC_V2_CONFIG_SHA256,
    }
    for key, value in expected.items():
        if identities.get(key) != value:
            raise TrackBError(f"execution lock identity mismatch: {key}")
    if float(identities.get("track_a_replay_tolerance", -1)) != REPLAY_TOLERANCE:
        raise TrackBError("Track-A replay tolerance drift")
    grade = lock.get("grade_rules", {})
    if int(grade.get("minimum_independent_families_per_required_mapped_class", -1)) != 50:
        raise TrackBError("external support floor drift")
    if grade.get("any_accepted_historical_link_permanently_blocks_ext_i") is not True:
        raise TrackBError("historical-overlap EXT-I guard missing")
    if grade.get("known_historical_contributor_relationship_is_ext_x") is not True:
        raise TrackBError("known-contributor EXT-X guard missing")
    if grade.get("residual_lineage_uncertainty_max_grade") != "EXT-S":
        raise TrackBError("residual-lineage EXT-S cap missing")
    if lock.get("metrics", {}).get("native_output_space") != 120:
        raise TrackBError("native output-space contract drift")
    frozen_candidates = (
        ("candidate_a", "gvlid_grape", "gvlid_v5"),
        ("candidate_b", "irish_potato", "irish_potato"),
    )
    for key, candidate_id, role in frozen_candidates:
        row = lock.get(key, {})
        contract = CANDIDATE_CONTRACTS[candidate_id]
        if row.get("id") != candidate_id or row.get("role") != role:
            raise TrackBError(f"Track-B v2 candidate identity drift: {key}")
        if row.get("scope") != contract["scope"] or row.get("doi") != contract["doi"] or str(row.get("version")) != contract["version"]:
            raise TrackBError(f"Track-B v2 candidate source identity drift: {candidate_id}")
        if row.get("mapping") != contract["mapping"]:
            raise TrackBError(f"Track-B v2 candidate mapping drift: {candidate_id}")
    automation = lock.get("automation", {})
    expected_automation = {
        "single_master_notebook": True,
        "kaggle_api_secret": "KAGGLE_API_TOKEN",
        "github_token_secret": "CROPCOP_GITHUB_TOKEN",
        "auto_download_owned_kaggle_inputs": True,
        "auto_create_or_version_private_kaggle_cache_datasets": True,
        "auto_archive_complete_restricted_evidence_to_private_kaggle": True,
        "auto_commit_public_safe_evidence_to_github": True,
        "raw_images_to_github": False,
        "checkpoints_to_github": False,
        "secrets_to_persistent_artifacts": False,
        "github_publication_requires_final_qa_terminal": True,
    }
    for key, value in expected_automation.items():
        if automation.get(key) != value:
            raise TrackBError(f"Track-B v2 automation policy drift: {key}")
    if automation.get("kaggle_dataset_owner_default") != "AUTO":
        raise TrackBError("Track-B v2 private Kaggle owner must default to AUTO")
    if automation.get("kaggle_dataset_owner_mode") != "AUTHENTICATED_TOKEN_OWNER_AUTO_DETECT":
        raise TrackBError("Track-B v2 private Kaggle owner must be inferred from the active API token")
    kaggle_policy = lock.get("kaggle", {})
    if kaggle_policy.get("internet_required_during_claim_run") is not True:
        raise TrackBError("Track-B v2 automated master requires Internet for orchestration/publication")
    if kaggle_policy.get("internet_role") != "ORCHESTRATION_AND_EVIDENCE_PUBLICATION_ONLY":
        raise TrackBError("Track-B v2 Internet role drift")
    if kaggle_policy.get("protected_inference_network_dependency") is not False:
        raise TrackBError("Track-B v2 protected inference must not have a network dependency")
    dino_pre = lock.get("candidate_generation", {}).get("dino_feature_preprocessing", {})
    if dino_pre.get("entrypoint") != "cropcop_je.data.ctc_v2_eval_transform" or dino_pre.get("ctc_v2_config_sha256") != CTC_V2_CONFIG_SHA256:
        raise TrackBError("DINO audit preprocessing contract drift")
    historical = lock.get("historical_compare", {})
    safe = historical.get("safe_postclosure_route", {})
    expected_safe = {
        "coverage_scope": "V1_TRAIN_VAL_ONLY",
        "image_count": 92744,
        "train_image_count": 76376,
        "validation_image_count": 16368,
        "v1_test_image_bytes_accessed": False,
        "maximum_evidence_grade": "EXT-S",
    }
    for key, value in expected_safe.items():
        if safe.get(key) != value:
            raise TrackBError(f"safe post-closure historical route drift: {key}")
    full = historical.get("full_ext_i_route", {})
    if (
        int(full.get("image_count", -1)) != 117546
        or full.get("authorization") != "RECOVERED_PRE_TEST_CRYPTOGRAPHIC_REPRESENTATION_ONLY"
        or full.get("raw_postclosure_rebuild_forbidden") is not True
    ):
        raise TrackBError("full EXT-I historical representation policy drift")
    if len(str(lock.get("code_attestation_sha256", ""))) != 64:
        raise TrackBError("Track-B execution lock does not bind a code attestation")


def discover_kaggle_inputs(input_root: str | Path = "/kaggle/input") -> dict[str, InputBundle]:
    root = Path(input_root)
    if not root.is_dir():
        raise TrackBError(f"Kaggle input root not found: {root}")
    found: dict[str, InputBundle] = {}
    for manifest_path in sorted(root.glob("**/TRACKB_INPUT_MANIFEST.json")):
        manifest = load_json(manifest_path)
        role = str(manifest.get("role", "")).strip()
        if not role:
            raise TrackBError(f"input manifest has no role: {manifest_path}")
        if role in found:
            raise TrackBError(f"duplicate Track-B input role {role}: {found[role].manifest_path} / {manifest_path}")
        found[role] = InputBundle(role, manifest_path.parent.resolve(), manifest_path.resolve(), manifest)
    missing = REQUIRED_INPUT_ROLES.difference(found)
    extra = set(found).difference(REQUIRED_INPUT_ROLES)
    if missing or extra:
        raise TrackBError(f"Track-B Kaggle input-role mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    assert_no_v1_test_surface(found)
    return found


def assert_no_v1_test_surface(inputs: dict[str, InputBundle]) -> None:
    for bundle in inputs.values():
        text = json.dumps(bundle.manifest, sort_keys=True)
        if FORBIDDEN_SURFACE in text:
            raise TrackBError(f"forbidden consumed V1-test surface referenced by input role {bundle.role}")
        for marker in ("v1_test", "test_consumed", "DS-V1-TEST"):
            if marker.lower() in text.lower():
                raise TrackBError(f"possible consumed-test material referenced by input role {bundle.role}: {marker}")


def resolve_bundle_file(bundle: InputBundle, key: str, *, require_hash: bool = True) -> Path:
    files = bundle.manifest.get("files", {})
    row = files.get(key)
    if not isinstance(row, dict):
        raise TrackBError(f"input role {bundle.role} does not define file key {key}")
    rel = str(row.get("path", "")).strip()
    if not rel:
        raise TrackBError(f"input role {bundle.role} file {key} has no path")
    path = (bundle.root / rel).resolve()
    if bundle.root not in path.parents and path != bundle.root:
        raise TrackBError(f"input role {bundle.role} file {key} escapes its dataset root")
    if not path.is_file():
        raise TrackBError(f"input file missing for {bundle.role}/{key}: {path}")
    if require_hash:
        expected = str(row.get("sha256", ""))
        if len(expected) != 64:
            raise TrackBError(f"input role {bundle.role} file {key} lacks SHA-256")
        require_sha256(path, expected, f"{bundle.role}/{key}")
    return path


def _checkpoint_state(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("student", "state_dict", "model", "model_state_dict"):
            state = payload.get(key)
            if isinstance(state, dict):
                return state
        if payload and all(hasattr(value, "shape") for value in payload.values()):
            return payload
    raise TrackBError("R07 checkpoint does not expose a supported state-dict payload")


def load_r07_checkpoint(path: str | Path, *, seed_label: str):
    import torch
    import torchvision
    from torchvision.models import convnext_tiny
    from .models import TorchvisionConvNeXtTinyAdapter

    seed_label = str(seed_label).upper()
    if seed_label not in R07_CHECKPOINTS:
        raise TrackBError(f"unknown R07 seed label: {seed_label}")
    require_sha256(path, R07_CHECKPOINTS[seed_label], f"R07 {seed_label} selected checkpoint")
    version = torchvision.__version__.split("+", 1)[0]
    if version != "0.27.1":
        raise TrackBError(f"torchvision version drift: expected 0.27.1, got {version}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = TorchvisionConvNeXtTinyAdapter(convnext_tiny(weights=None, num_classes=120))
    model.load_state_dict(_checkpoint_state(payload), strict=True)
    model.eval()
    return model, payload


def assign_candidate_grade(
    *,
    source_identity_ok: bool,
    mapping_ok: bool,
    family_support: dict[str, int],
    required_labels: Iterable[str],
    historical_surface_complete: bool,
    accepted_historical_link_count: int,
    unresolved_lineage: bool,
    known_historical_contributor_relationship: bool = False,
) -> CandidateGrade:
    required_labels = tuple(required_labels)
    reasons: list[str] = []
    if not source_identity_ok:
        return CandidateGrade("EXT-X", "NO_PERFORMANCE", ("SOURCE_IDENTITY",))
    if not mapping_ok:
        return CandidateGrade("EXT-X", "NO_PERFORMANCE", ("MAPPING",))
    if known_historical_contributor_relationship:
        return CandidateGrade("EXT-X", "NO_PERFORMANCE", ("KNOWN_HISTORICAL_CONTRIBUTOR_RELATIONSHIP",))
    low = [label for label in required_labels if int(family_support.get(label, 0)) < 50]
    if low:
        return CandidateGrade("EXT-X", "NO_PERFORMANCE", tuple(f"SUPPORT<{50}:{label}" for label in low))
    if not historical_surface_complete:
        reasons.append("HISTORICAL_COMPARISON_INCOMPLETE")
    if int(accepted_historical_link_count) > 0:
        reasons.append("ACCEPTED_HISTORICAL_LINKS")
    if unresolved_lineage:
        reasons.append("UNRESOLVED_LINEAGE")
    if reasons:
        return CandidateGrade("EXT-S", "CROSS_DATASET_STRESS", tuple(reasons))
    return CandidateGrade("EXT-I", "AUDIT_BOUNDED_SOURCE_INDEPENDENT", ())


def build_candidate_seal(payload: dict[str, Any]) -> dict[str, Any]:
    seal = dict(payload)
    seal.pop("seal_sha256", None)
    seal["seal_sha256"] = sha256_json(seal)
    return seal


def verify_candidate_seal(seal: dict[str, Any]) -> None:
    expected = str(seal.get("seal_sha256", ""))
    clean = dict(seal)
    clean.pop("seal_sha256", None)
    if len(expected) != 64 or sha256_json(clean) != expected:
        raise TrackBError("candidate seal self-hash mismatch")
    candidate_id = str(seal.get("candidate_id", ""))
    contract = CANDIDATE_CONTRACTS.get(candidate_id)
    if contract is None:
        raise TrackBError("candidate seal candidate identity invalid")
    grade = str(seal.get("grade", ""))
    if grade not in {"EXT-I", "EXT-S", "EXT-X"}:
        raise TrackBError("candidate seal grade invalid")
    if seal.get("scope") != contract["scope"] or seal.get("source_doi") != contract["doi"] or str(seal.get("source_version")) != contract["version"]:
        raise TrackBError("candidate seal source/scope identity invalid")
    mapping = seal.get("mapping")
    if mapping != contract["mapping"] or seal.get("mapping_sha256") != sha256_json(mapping):
        raise TrackBError("candidate seal mapping identity invalid")
    if not str(seal.get("sealed_at_utc", "")).strip():
        raise TrackBError("candidate seal timestamp missing")
    if int(seal.get("prediction_count_at_seal", -1)) != 0:
        raise TrackBError("candidate seal was not prediction-blind")
    pre = seal.get("preprocessing", {})
    if pre.get("entrypoint") != "cropcop_je.data.ctc_v2_eval_transform" or pre.get("config_sha256") != CTC_V2_CONFIG_SHA256:
        raise TrackBError("candidate seal preprocessing binding invalid")
    if int(seal.get("bootstrap_seed", -1)) != 409883112 or int(seal.get("bootstrap_replicates", -1)) != 5000:
        raise TrackBError("candidate seal bootstrap contract drift")
    if int(seal.get("external_family_order_seed", -1)) != 1936263114:
        raise TrackBError("candidate seal external-family seed drift")
    for field in (
        "downstream_authority_sha256", "execution_lock_sha256", "source_manifest_sha256", "source_metadata_record_sha256", "mapping_sha256", "family_graph_sha256",
        "representative_manifest_sha256", "exclusion_ledger_sha256", "decode_failure_ledger_sha256",
        "historical_compare_input_manifest_sha256", "accepted_within_edges_sha256", "accepted_historical_edges_sha256",
        "historical_comparison_summary_sha256", "within_comparison_summary_sha256",
    ):
        if len(str(seal.get(field, ""))) != 64:
            raise TrackBError(f"candidate seal missing SHA-256 binding: {field}")
    for field in ("source_identity_ok", "mapping_ok", "unresolved_lineage", "known_historical_contributor_relationship", "historical_surface_complete"):
        if not isinstance(seal.get(field), bool):
            raise TrackBError(f"candidate seal missing boolean grading input: {field}")
    support = seal.get("family_support")
    required_labels = tuple(contract["mapping"].keys())
    if not isinstance(support, dict) or set(support) != set(required_labels):
        raise TrackBError("candidate seal family-support keys differ from frozen required source labels")
    support = {str(k): int(v) for k, v in support.items()}
    if any(value < 0 for value in support.values()):
        raise TrackBError("candidate seal family support contains a negative count")
    recomputed = assign_candidate_grade(
        source_identity_ok=seal["source_identity_ok"],
        mapping_ok=seal["mapping_ok"],
        family_support=support,
        required_labels=required_labels,
        historical_surface_complete=seal["historical_surface_complete"],
        accepted_historical_link_count=int(seal.get("accepted_historical_link_count", -1)),
        unresolved_lineage=seal["unresolved_lineage"],
        known_historical_contributor_relationship=seal["known_historical_contributor_relationship"],
    )
    if grade != recomputed.grade or seal.get("claim_mode") != recomputed.claim_mode or list(seal.get("grade_reasons", [])) != list(recomputed.reasons):
        raise TrackBError("candidate seal grade does not recompute from its sealed prediction-blind inputs")
    authorized = {str(k).upper(): str(v) for k, v in seal.get("authorized_r07_checkpoint_sha256", {}).items()}
    if grade in {"EXT-I", "EXT-S"}:
        if authorized != R07_CHECKPOINTS:
            raise TrackBError("claim-producing candidate seal does not bind all three R07 checkpoints")
        if any(int(support[label]) < 50 for label in required_labels):
            raise TrackBError("claim-producing candidate seal violates the 50-family support floor")
    elif authorized:
        raise TrackBError("EXT-X candidate seal must not authorize protected R07 inference")


def validate_same_prediction_surface(seed_rows: dict[str, list[dict[str, Any]]]) -> list[str]:
    if set(seed_rows) != {"S1", "S2", "S3"}:
        raise TrackBError("external inference requires exactly S1/S2/S3")
    reference = [(str(r["stable_row_id"]), int(r["target_class_index"])) for r in seed_rows["S1"]]
    if len(reference) != len({x[0] for x in reference}):
        raise TrackBError("prediction surface contains duplicate row IDs")
    for seed in ("S2", "S3"):
        observed = [(str(r["stable_row_id"]), int(r["target_class_index"])) for r in seed_rows[seed]]
        if observed != reference:
            raise TrackBError(f"{seed} prediction rows/targets differ from S1")
    return [row_id for row_id, _ in reference]


def mapped_scope_metrics(rows: Iterable[dict[str, Any]], mapped_class_indices: Iterable[int]) -> dict[str, Any]:
    rows = list(rows)
    labels = tuple(sorted({int(x) for x in mapped_class_indices}))
    if not labels:
        raise TrackBError("mapped metric scope is empty")
    label_set = set(labels)
    per: dict[int, dict[str, float | int]] = {}
    correct = 0
    outside = 0
    for row in rows:
        target = int(row["target_class_index"])
        pred = int(row["predicted_class_index"])
        if target not in label_set:
            raise TrackBError(f"row target {target} lies outside the frozen mapped ground-truth scope")
        if not 0 <= pred < 120:
            raise TrackBError(f"native prediction {pred} lies outside the 120-way classifier")
        correct += int(pred == target)
        outside += int(pred not in label_set)
    if not rows:
        raise TrackBError("cannot score an empty candidate")
    f1s = []
    recalls = []
    for cls in labels:
        tp = sum(int(int(r["target_class_index"]) == cls and int(r["predicted_class_index"]) == cls) for r in rows)
        fp = sum(int(int(r["target_class_index"]) != cls and int(r["predicted_class_index"]) == cls) for r in rows)
        fn = sum(int(int(r["target_class_index"]) == cls and int(r["predicted_class_index"]) != cls) for r in rows)
        support = sum(int(int(r["target_class_index"]) == cls) for r in rows)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per[cls] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1s.append(f1)
        recalls.append(recall)
    confusion = {}
    for target in labels:
        counts: dict[str, int] = {}
        for row in rows:
            if int(row["target_class_index"]) != target:
                continue
            pred = str(int(row["predicted_class_index"]))
            counts[pred] = counts.get(pred, 0) + 1
        confusion[str(target)] = dict(sorted(counts.items(), key=lambda item: int(item[0])))
    return {
        "row_count": len(rows),
        "mapped_class_indices": list(labels),
        "accuracy": correct / len(rows),
        "macro_f1": sum(f1s) / len(f1s),
        "balanced_accuracy": sum(recalls) / len(recalls),
        "out_of_mapped_subset_prediction_rate": outside / len(rows),
        "per_class": {str(k): v for k, v in per.items()},
        "confusion_matrix_native120_nonzero": confusion,
    }


def three_seed_summary(seed_metrics: dict[str, dict[str, Any]]) -> dict[str, float]:
    if set(seed_metrics) != {"S1", "S2", "S3"}:
        raise TrackBError("three-seed summary requires exactly S1/S2/S3")
    macro = [float(seed_metrics[s]["macro_f1"]) for s in ("S1", "S2", "S3")]
    return {
        "mean_macro_f1": sum(macro) / 3.0,
        "sample_sd_macro_f1": statistics.stdev(macro),
        "minimum_seed_macro_f1": min(macro),
        "maximum_seed_macro_f1": max(macro),
    }


def validate_replay(observed: dict[str, float], expected: dict[str, float], *, tolerance: float = REPLAY_TOLERANCE) -> dict[str, Any]:
    names = ("validation_accuracy", "validation_balanced_accuracy", "validation_macro_f1", "validation_nll")
    diffs = {name: abs(float(observed[name]) - float(expected[name])) for name in names}
    maximum = max(diffs.values(), default=0.0)
    return {"status": "PASS" if maximum <= tolerance else "FAIL", "tolerance": tolerance, "differences": diffs, "max_abs_diff": maximum}


def file_record(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
