#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from lib.repo_contract import EXPECTED, FORBIDDEN_NAMES, FORBIDDEN_SUFFIXES, SECRET_PATTERNS

ROOT = Path(__file__).resolve().parents[1]

def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)

def as_int(value):
    return int(float(value))

def rows_by(path: str, key: str):
    rows = list(csv.DictReader((ROOT / path).open(encoding="utf-8", newline="")))
    return rows, {r[key]: r for r in rows}

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json-report", default="")
    args = parser.parse_args()
    errors: list[str] = []

    registry = json.loads((ROOT / "metrics/metric_registry.json").read_text())
    dataset_fingerprints = json.loads((ROOT / "data_card/dataset_fingerprints.json").read_text())
    lineage = json.loads((ROOT / "data_card/lineage/lineage.json").read_text())
    certificate = json.loads((ROOT / "evidence/public/dataset_lineage/final_dataset_certificate.json").read_text())
    fingerprint_status = json.loads((ROOT / "data_card/audit/fingerprint_verification_status.json").read_text())
    confidence_defs = json.loads((ROOT / "data_card/provenance/provenance_confidence_definitions.json").read_text())
    review_recon = json.loads((ROOT / "data_card/audit/model_readiness_manual_review_reconstruction.json").read_text())

    results = list(csv.DictReader((ROOT / "metrics/main_results.csv").open()))
    claims = list(csv.DictReader((ROOT / "evidence/public/claim_evidence_matrix.csv").open()))
    source_registry, source_registry_by = rows_by("data_card/lineage/source_registry.csv", "registry_key")
    ontology_history, ontology_history_by = rows_by("data_card/lineage/ontology_history.csv", "ontology_state")
    ontology_transitions, ontology_transition_by = rows_by("data_card/lineage/ontology_transition_registry.csv", "transition_id")
    provenance = list(csv.DictReader((ROOT / "data_card/provenance/provenance_coverage.csv").open()))
    source_counts, source_counts_by = rows_by("data_card/provenance/final_source_counts.csv", "ingestion_registry_key")
    source_split_counts, source_split_by = rows_by("data_card/provenance/final_source_split_counts.csv", "ingestion_registry_key")
    complement, complement_by = rows_by("data_card/provenance/historical_test_complement.csv", "ingestion_registry_key")
    exclusions, exclusions_by = rows_by("data_card/provenance/external_evaluation_exclusion_registry.csv", "registry_key")
    duplicate_history, duplicate_by = rows_by("data_card/audit/duplicate_edge_history.csv", "edge_category")
    accounting = list(csv.DictReader((ROOT / "data_card/audit/final_accounting.csv").open()))
    manual_summary = list(csv.DictReader((ROOT / "data_card/audit/manual_review_summary.csv").open()))
    evidence_registry = list(csv.DictReader((ROOT / "evidence/public/dataset_lineage/evidence_registry.csv").open()))

    # Existing model/runtime contract.
    for key in ("images", "classes", "train", "val", "test"):
        check(registry["dataset"][key] == EXPECTED[key], f"dataset.{key} mismatch", errors)
    check(registry["pte"]["bytes"] == EXPECTED["pte_bytes"], "PTE byte count mismatch", errors)
    check(registry["pte"]["sha256"] == EXPECTED["pte_sha256"], "PTE SHA-256 mismatch", errors)
    check({r["state"] for r in results} == {"reference", "mobile_float", "converted_int8", "pte_runtime"}, "model-state table incomplete", errors)
    blocked = {r["claim_id"] for r in claims if r["status"] == "blocked"}
    check({"C14", "C15", "C16", "C42", "C45", "C48"}.issubset(blocked), "blocked claim ledger is incomplete", errors)

    # Final benchmark identity across independent public records.
    check(certificate["final_images"] == EXPECTED["images"], "final certificate image count mismatch", errors)
    check(certificate["final_classes"] == EXPECTED["classes"], "final certificate class count mismatch", errors)
    check(certificate["split_counts"] == {"test": EXPECTED["test"], "train": EXPECTED["train"], "val": EXPECTED["val"]}, "final certificate split mismatch", errors)
    check(certificate["manual_review_deleted"] == EXPECTED["manual_review_deletions"], "manual-review certificate mismatch", errors)
    final_stage = lineage["stages"][-1]
    check(final_stage["images"] == EXPECTED["images"] and final_stage["classes"] == EXPECTED["classes"], "canonical lineage final state mismatch", errors)
    check(final_stage["unique_sha256"] == EXPECTED["images"], "final unique SHA-256 count mismatch", errors)
    check(final_stage["leakage_groups_crossing_splits"] == 0, "final leakage-group crossing is non-zero", errors)
    check(dataset_fingerprints["images"] == EXPECTED["images"] and dataset_fingerprints["classes"] == EXPECTED["classes"], "dataset_fingerprints identity mismatch", errors)
    check(dataset_fingerprints["train"] == EXPECTED["train"] and dataset_fingerprints["validation"] == EXPECTED["val"] and dataset_fingerprints["test"] == EXPECTED["test"], "dataset_fingerprints split mismatch", errors)

    # Fingerprint consistency plus explicit non-recomputation boundary.
    check(certificate["final_manifest_fingerprint"] == EXPECTED["final_manifest_fingerprint"], "final certificate manifest fingerprint mismatch", errors)
    check(certificate["build_fingerprint"] == EXPECTED["final_build_fingerprint"], "final certificate build fingerprint mismatch", errors)
    check(dataset_fingerprints["semantic_fingerprint"] == EXPECTED["final_manifest_fingerprint"], "dataset_fingerprints manifest identity mismatch", errors)
    check(dataset_fingerprints["build_fingerprint"] == EXPECTED["final_build_fingerprint"], "dataset_fingerprints build identity mismatch", errors)
    check(lineage["fingerprints"]["final_manifest"] == EXPECTED["final_manifest_fingerprint"], "lineage manifest fingerprint mismatch", errors)
    check(lineage["fingerprints"]["final_build"] == EXPECTED["final_build_fingerprint"], "lineage build fingerprint mismatch", errors)
    fp = {r["id"]: r for r in fingerprint_status["fingerprints"]}
    for fid, expected in (("final_manifest_fingerprint", EXPECTED["final_manifest_fingerprint"]), ("final_build_fingerprint", EXPECTED["final_build_fingerprint"])):
        check(fp[fid]["value"] == expected, f"{fid} status record mismatch", errors)
        check(fp[fid]["public_validator_status"] == "cross_file_consistency_only", f"{fid} overstates public recomputation", errors)
        check(fp[fid]["independent_public_recomputation"] is False, f"{fid} incorrectly claims public recomputation", errors)

    # Duplicate-edge terminology and arithmetic.
    check(as_int(duplicate_by["historical_confirmed_total"]["historical_v4_count"]) == EXPECTED["historical_v4_edges"], "historical V4 edge count mismatch", errors)
    check(as_int(duplicate_by["historical_confirmed_total"]["corrected_v5_count"]) == EXPECTED["corrected_v5_edges"], "corrected V5 edge count mismatch", errors)
    check(as_int(duplicate_by["exact_sha256"]["corrected_v5_count"]) == EXPECTED["exact_edges"], "exact edge count mismatch", errors)
    check(as_int(duplicate_by["strong_hash_near_duplicate"]["corrected_v5_count"]) == EXPECTED["strong_hash_edges"], "strong-hash edge count mismatch", errors)
    check(as_int(duplicate_by["feature_route_near_duplicate"]["historical_v4_count"]) == EXPECTED["feature_edges_reopened"], "feature edge reopened count mismatch", errors)
    check(as_int(duplicate_by["feature_route_near_duplicate"]["corrected_v5_count"]) == EXPECTED["feature_edges_retained"], "feature edge retained count mismatch", errors)
    check(as_int(duplicate_by["feature_route_near_duplicate"]["rejected_after_fix"]) == EXPECTED["feature_edges_rejected"], "feature edge rejection count mismatch", errors)
    check(as_int(duplicate_by["confirmed_cross_label_subset"]["corrected_v5_count"]) == EXPECTED["cross_label_subset"], "cross-label subset count mismatch", errors)
    check(EXPECTED["exact_edges"] + EXPECTED["strong_hash_edges"] + EXPECTED["feature_edges_retained"] == EXPECTED["corrected_v5_edges"], "corrected edge composition does not sum", errors)

    # Benchmark row accounting.
    check(accounting[0]["transition"] == "v4_audited_candidate_universe" and as_int(accounting[0]["ending_rows"]) == EXPECTED["audited_images"], "V4 accounting start mismatch", errors)
    check(any(r["transition"] == "v5_group_safe_benchmark" and as_int(r["ending_rows"]) == EXPECTED["v5_images"] for r in accounting), "V5 accounting state missing", errors)
    check(accounting[-1]["transition"] == "final_frozen_benchmark" and as_int(accounting[-1]["ending_rows"]) == EXPECTED["images"], "final accounting mismatch", errors)
    for r in accounting:
        check(as_int(r["starting_rows"]) + as_int(r["delta_rows"]) == as_int(r["ending_rows"]), f"accounting arithmetic mismatch: {r['transition']}", errors)

    # Source-registry and historical-complement reconstruction.
    check(len(source_registry) == EXPECTED["registered_sources"], "source registry row count mismatch", errors)
    check(len(source_registry_by) == EXPECTED["registered_sources"], "source registry keys are not unique", errors)
    check(sum(as_int(r["rows_in_recovered_registry"]) for r in source_registry) == EXPECTED["recovered_registry_rows"], "recovered source registry total mismatch", errors)
    check(sum(as_int(r["rows_in_recovered_early_manifest"]) for r in source_registry) == EXPECTED["early_manifest_rows"], "early manifest source total mismatch", errors)
    check(sum(as_int(r["rows_in_v4_audited_universe"]) for r in source_registry) == EXPECTED["audited_images"], "source registry does not reconcile to V4 universe", errors)
    check(EXPECTED["recovered_registry_rows"] - EXPECTED["audited_images"] == EXPECTED["registry_to_v4_attrition"], "registry-to-V4 attrition arithmetic mismatch", errors)
    registry_attrition = {k: as_int(r["rows_in_recovered_registry"]) - as_int(r["rows_in_v4_audited_universe"]) for k, r in source_registry_by.items()}
    check({k:v for k,v in registry_attrition.items() if v != 0} == {"rice": EXPECTED["registry_to_v4_attrition"]}, "registry-to-V4 attrition is not confined to 16 rice rows", errors)

    check(set(complement_by) == set(source_registry_by), "historical complement registry keys do not match source registry", errors)
    for key, r in complement_by.items():
        check(as_int(r["rows_in_recovered_early_manifest"]) == as_int(source_registry_by[key]["rows_in_recovered_early_manifest"]), f"complement early-manifest mismatch for {key}", errors)
        check(as_int(r["rows_in_v4_audited_universe"]) == as_int(source_registry_by[key]["rows_in_v4_audited_universe"]), f"complement V4 mismatch for {key}", errors)
        check(as_int(r["manifest_to_v4_delta"]) == as_int(r["rows_in_v4_audited_universe"]) - as_int(r["rows_in_recovered_early_manifest"]), f"complement delta arithmetic mismatch for {key}", errors)
    complement_nonzero = {k: as_int(r["manifest_to_v4_delta"]) for k,r in complement_by.items() if as_int(r["manifest_to_v4_delta"]) != 0}
    check(sum(complement_nonzero.values()) == EXPECTED["manifest_to_v4_complement"], "historical manifest-to-V4 complement total mismatch", errors)
    check(complement_nonzero == {"plantcity_pk": EXPECTED["plantcity_manifest_to_v4_delta"], "bangladesh": EXPECTED["bangladesh_manifest_to_v4_delta"]}, "historical complement must be 26,017 PlantCity + 1 Bangladesh", errors)
    cstage = next(x for x in lineage["stages"] if x["stage"] == "C_early_cleaning_historical_split")
    check(cstage["manifest_to_v4_test_complement_rows"] == EXPECTED["manifest_to_v4_complement"], "lineage complement total mismatch", errors)
    check(cstage["manifest_to_v4_test_complement_breakdown"] == {"plantcity_pk": 26017, "bangladesh": 1}, "lineage complement breakdown mismatch", errors)
    check("rice_neck_blast" in cstage["complement_note"] and "not" in cstage["complement_note"].lower(), "lineage does not explicitly separate rice_neck_blast from complement", errors)

    # Ontology-history contract.
    check(as_int(ontology_history_by["historical_label_map"]["value"]) == 249, "historical label-map size mismatch", errors)
    check(as_int(ontology_history_by["historical_class_index"]["value"]) == 80, "historical class-index size mismatch", errors)
    check(as_int(ontology_history_by["v4_candidate_ontology"]["value"]) == 121, "V4 candidate ontology size mismatch", errors)
    check(as_int(ontology_history_by["v5_operational_ontology"]["value"]) == EXPECTED["classes"], "V5 operational ontology size mismatch", errors)
    check(set(ontology_transition_by) == {"OT01","OT02","OT03","OT04","OT05","OT06","OT07"}, "ontology transition registry incomplete", errors)
    check("Smut" in ontology_transition_by["OT02"]["details"] and "Healthy" in ontology_transition_by["OT02"]["details"], "historical case-sensitive mapping ambiguities are not documented", errors)
    check("tomato_leaf_bacterial_spot" in ontology_transition_by["OT03"]["details"], "tomato merge missing from ontology registry", errors)
    check("17 nutrient-deficiency" in ontology_transition_by["OT04"]["details"], "nutrient collapse missing from ontology registry", errors)
    check("rice_neck_blast" in ontology_transition_by["OT07"]["details"], "121-to-120 transition missing rice_neck_blast", errors)

    # Provenance coverage and confidence semantics.
    check(sum(as_int(r["final_rows"]) for r in provenance) == EXPECTED["source_family_rows"], "provenance coverage rows do not sum to final dataset", errors)
    exact_rows = sum(as_int(r["final_rows"]) for r in provenance if r["original_source_path_status"] == "available")
    source_only_rows = sum(as_int(r["final_rows"]) for r in provenance if r["original_source_path_status"] == "unknown")
    check(exact_rows == EXPECTED["exact_source_path_rows"], "exact source-path coverage mismatch", errors)
    check(source_only_rows == EXPECTED["source_family_only_rows"], "source-family-only coverage mismatch", errors)
    complement_rows = [r for r in provenance if r["original_source_path_status"] == "unknown"]
    check(len(complement_rows) == 1 and complement_rows[0]["ingestion_registry_key"] == "plantcity_pk", "source-family-only provenance is not confined to PlantCity", errors)
    if complement_rows:
        check(complement_rows[0]["provenance_method"] == "registry_complement_reconstruction", "PlantCity complement method mismatch", errors)
        check(complement_rows[0]["provenance_confidence"] == "reconstructed_unique_complement", "PlantCity complement confidence is overstated or stale", errors)
    level_ids = {x["id"] for x in confidence_defs["levels"]}
    check(level_ids == {"verified_exact","reconstructed_unique_complement","unknown"}, "provenance confidence definitions incomplete", errors)
    check(confidence_defs["supersession"]["prior_public_label"] == "verified_exact_source_family", "prior overstated confidence label is not recorded", errors)
    check(lineage["provenance_coverage"]["source_family_only_confidence"] == "reconstructed_unique_complement", "lineage complement confidence mismatch", errors)

    # Source-by-source final reconciliation across four public derivatives.
    prov_by_source = defaultdict(int)
    for r in provenance:
        prov_by_source[r["ingestion_registry_key"]] += as_int(r["final_rows"])
    final_keys = {k for k,r in source_registry_by.items() if as_int(r["rows_in_final_frozen_dataset"]) > 0}
    check(final_keys == set(source_counts_by) == set(source_split_by) == set(prov_by_source), "final source-family key sets disagree across registries", errors)
    for key in final_keys:
        reg_final = as_int(source_registry_by[key]["rows_in_final_frozen_dataset"])
        count_final = as_int(source_counts_by[key]["final_rows"])
        split_final = sum(as_int(source_split_by[key][s]) for s in ("train","val","test"))
        prov_final = prov_by_source[key]
        check(reg_final == count_final == split_final == prov_final, f"cross-file final source count mismatch for {key}", errors)
    check(sum(as_int(r["train"]) for r in source_split_counts) == EXPECTED["train"], "source-by-split train total mismatch", errors)
    check(sum(as_int(r["val"]) for r in source_split_counts) == EXPECTED["val"], "source-by-split val total mismatch", errors)
    check(sum(as_int(r["test"]) for r in source_split_counts) == EXPECTED["test"], "source-by-split test total mismatch", errors)

    # External-evaluation exclusion lock.
    check(set(exclusions_by) == set(source_registry_by), "external-evaluation exclusion registry does not cover all 15 historical source families", errors)
    check(all(r["external_evaluation_status"] == "exclude_pending_independence_proof" for r in exclusions), "a historical ingestion source is not conservatively excluded from external validation", errors)
    for key in final_keys:
        check(as_int(exclusions_by[key]["final_rows"]) == as_int(source_registry_by[key]["rows_in_final_frozen_dataset"]), f"external exclusion final-row count mismatch for {key}", errors)

    # Manual review and model-readiness reconstruction.
    manual_map = {(r["dimension"], r["value"]): as_int(r["deleted_rows"]) for r in manual_summary}
    check(manual_map.get(("action", "DELETE_FROM_DATASET")) == EXPECTED["manual_review_deletions"], "manual delete-only count mismatch", errors)
    check(manual_map.get(("status", "FLAGGED")) == EXPECTED["manual_review_deletions"], "manual flagged count mismatch", errors)
    check({s: manual_map.get(("split", s), 0) for s in ("train", "val", "test")} == {"train": 29, "val": 8, "test": 6}, "manual deletion split counts mismatch", errors)
    check(sum(v for (d, _), v in manual_map.items() if d == "label") == EXPECTED["manual_review_deletions"], "manual deletion label counts mismatch", errors)
    check(review_recon["residual_cross_label_similarity_pairs"] == 122, "residual pair count mismatch", errors)
    check(review_recon["manual_review_queue_rows"] == 180 and review_recon["unique_pair_endpoints"] == 180, "manual review queue reconstruction mismatch", errors)
    check(review_recon["pair_endpoint_set_equals_manual_review_queue"] is True, "manual review queue endpoint equality not established", errors)
    check(review_recon["priority_score_formula_matches_all_180_rows"] is True, "priority score reconstruction mismatch", errors)
    check(review_recon["selector_source_code_status"] == "not supplied in the recovered evidence bundle", "manual-review generator provenance is overstated", errors)

    # Evidence registry integrity and derivative existence.
    required_evidence_cols = {"evidence_id", "stage", "artifact_name", "sha256", "file_size_bytes", "visibility", "claims_supported", "public_derivative"}
    check(bool(evidence_registry) and required_evidence_cols.issubset(evidence_registry[0].keys()), "evidence registry schema incomplete", errors)
    check(len({r["evidence_id"] for r in evidence_registry}) == len(evidence_registry), "evidence IDs are not unique", errors)
    check(all(len(r["sha256"]) == 64 for r in evidence_registry), "evidence registry contains invalid SHA-256 field", errors)
    for r in evidence_registry:
        if r["public_derivative"]:
            check((ROOT / r["public_derivative"]).exists(), f"evidence registry public derivative missing: {r['evidence_id']} -> {r['public_derivative']}", errors)
    ep01 = next(r for r in evidence_registry if r["evidence_id"] == "EV-P01")
    check(ep01["status"] == "current-with-interpretation-supersession", "restricted provenance ledger supersession status missing", errors)

    # Repository safety/hygiene scan.
    public_dataset_prefixes = (Path("data_card/lineage"), Path("data_card/provenance"), Path("data_card/audit"), Path("evidence/public/dataset_lineage"))
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        check(not any(part in FORBIDDEN_NAMES for part in rel.parts), f"restricted path committed: {rel}", errors)
        check(path.suffix.lower() not in FORBIDDEN_SUFFIXES, f"restricted binary committed: {rel}", errors)
        if path.stat().st_size > 25 * 1024 * 1024:
            errors.append(f"file exceeds 25 MiB public threshold: {rel}")
        if path.suffix.lower() in {".md", ".tex", ".bib", ".json", ".csv", ".py", ".sh", ".yml", ".yaml", ".cff", ".txt"} or path.name == "Makefile":
            text = path.read_text(encoding="utf-8", errors="replace")
            if rel != Path("scripts/validate_repository.py"):
                check("/mnt/data/" not in text and "/home/oai/" not in text, f"private absolute path in {rel}", errors)
            if any(rel == p or p in rel.parents for p in public_dataset_prefixes):
                check("/kaggle/input/" not in text and "/kaggle/working/" not in text, f"private/runtime Kaggle path in public dataset evidence: {rel}", errors)
            for pattern in SECRET_PATTERNS:
                check(pattern.search(text) is None, f"possible secret in {rel}", errors)

    # Canonical documentation and dangerous stale wording.
    required_docs = (
        "docs/DATASET_LINEAGE.md", "docs/DATASET_PROVENANCE.md", "docs/DATASET_AUDIT_HISTORY.md",
        "00_EAAI_DATASET_EVIDENCE_QA_GATE.md"
    )
    for doc in required_docs:
        check((ROOT / doc).exists(), f"missing canonical dataset document: {doc}", errors)
    lineage_text = (ROOT / "docs/DATASET_LINEAGE.md").read_text(encoding="utf-8")
    provenance_text = (ROOT / "docs/DATASET_PROVENANCE.md").read_text(encoding="utf-8")
    limitations_text = (ROOT / "docs/LIMITATIONS.md").read_text(encoding="utf-8")
    check("plus one `rice_neck_blast` row" not in lineage_text and "single non-PlantCity row in the V4 test complement is `rice_neck_blast`" not in provenance_text, "stale rice_neck_blast complement interpretation remains", errors)
    check("Source-level provenance and redistribution rights are incomplete" not in limitations_text, "stale provenance limitation wording remains", errors)
    check("reconstructed_unique_complement" in provenance_text, "provenance document lacks bounded complement confidence wording", errors)

    report = {"status": "PASS" if not errors else "FAIL", "errors": errors, "warnings": []}
    if args.json_report:
        Path(args.json_report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1

if __name__ == "__main__":
    raise SystemExit(main())
