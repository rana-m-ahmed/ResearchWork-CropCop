#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from lib.repo_contract import EXPECTED, FORBIDDEN_NAMES, FORBIDDEN_SUFFIXES, SECRET_PATTERNS

ROOT = Path(__file__).resolve().parents[1]

def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)

def as_int(value):
    return int(float(value))

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json-report", default="")
    args = parser.parse_args()
    errors: list[str] = []

    registry = json.loads((ROOT / "metrics/metric_registry.json").read_text())
    results = list(csv.DictReader((ROOT / "metrics/main_results.csv").open()))
    claims = list(csv.DictReader((ROOT / "evidence/public/claim_evidence_matrix.csv").open()))
    lineage = json.loads((ROOT / "data_card/lineage/lineage.json").read_text())
    certificate = json.loads((ROOT / "evidence/public/dataset_lineage/final_dataset_certificate.json").read_text())
    source_registry = list(csv.DictReader((ROOT / "data_card/lineage/source_registry.csv").open()))
    ontology_history = list(csv.DictReader((ROOT / "data_card/lineage/ontology_history.csv").open()))
    provenance = list(csv.DictReader((ROOT / "data_card/provenance/provenance_coverage.csv").open()))
    source_counts = list(csv.DictReader((ROOT / "data_card/provenance/final_source_counts.csv").open()))
    source_split_counts = list(csv.DictReader((ROOT / "data_card/provenance/final_source_split_counts.csv").open()))
    duplicate_history = list(csv.DictReader((ROOT / "data_card/audit/duplicate_edge_history.csv").open()))
    accounting = list(csv.DictReader((ROOT / "data_card/audit/final_accounting.csv").open()))
    manual_summary = list(csv.DictReader((ROOT / "data_card/audit/manual_review_summary.csv").open()))
    review_recon = json.loads((ROOT / "data_card/audit/model_readiness_manual_review_reconstruction.json").read_text())
    evidence_registry = list(csv.DictReader((ROOT / "evidence/public/dataset_lineage/evidence_registry.csv").open()))

    for key in ("images", "classes", "train", "val", "test"):
        check(registry["dataset"][key] == EXPECTED[key], f"dataset.{key} mismatch", errors)
    check(registry["pte"]["bytes"] == EXPECTED["pte_bytes"], "PTE byte count mismatch", errors)
    check(registry["pte"]["sha256"] == EXPECTED["pte_sha256"], "PTE SHA-256 mismatch", errors)
    check({r["state"] for r in results} == {"reference", "mobile_float", "converted_int8", "pte_runtime"}, "model-state table incomplete", errors)
    blocked = {r["claim_id"] for r in claims if r["status"] == "blocked"}
    check({"C14", "C15", "C16", "C42", "C45"}.issubset(blocked), "blocked claim ledger is incomplete", errors)

    # Final certificate and canonical lineage identity.
    check(certificate["final_images"] == EXPECTED["images"], "final certificate image count mismatch", errors)
    check(certificate["final_classes"] == EXPECTED["classes"], "final certificate class count mismatch", errors)
    check(certificate["split_counts"] == {"test": EXPECTED["test"], "train": EXPECTED["train"], "val": EXPECTED["val"]}, "final certificate split mismatch", errors)
    check(certificate["manual_review_deleted"] == EXPECTED["manual_review_deletions"], "manual-review certificate mismatch", errors)
    check(certificate["final_manifest_fingerprint"] == EXPECTED["final_manifest_fingerprint"], "final manifest fingerprint mismatch", errors)
    check(certificate["build_fingerprint"] == EXPECTED["final_build_fingerprint"], "final build fingerprint mismatch", errors)
    final_stage = lineage["stages"][-1]
    check(final_stage["images"] == EXPECTED["images"] and final_stage["classes"] == EXPECTED["classes"], "canonical lineage final state mismatch", errors)
    check(final_stage["leakage_groups_crossing_splits"] == 0, "final leakage-group crossing is non-zero", errors)

    # Duplicate-edge terminology and arithmetic.
    rows = {r["edge_category"]: r for r in duplicate_history}
    check(as_int(rows["historical_confirmed_total"]["historical_v4_count"]) == EXPECTED["historical_v4_edges"], "historical V4 edge count mismatch", errors)
    check(as_int(rows["historical_confirmed_total"]["corrected_v5_count"]) == EXPECTED["corrected_v5_edges"], "corrected V5 edge count mismatch", errors)
    check(as_int(rows["exact_sha256"]["corrected_v5_count"]) == EXPECTED["exact_edges"], "exact edge count mismatch", errors)
    check(as_int(rows["strong_hash_near_duplicate"]["corrected_v5_count"]) == EXPECTED["strong_hash_edges"], "strong-hash edge count mismatch", errors)
    check(as_int(rows["feature_route_near_duplicate"]["historical_v4_count"]) == EXPECTED["feature_edges_reopened"], "feature edge reopened count mismatch", errors)
    check(as_int(rows["feature_route_near_duplicate"]["corrected_v5_count"]) == EXPECTED["feature_edges_retained"], "feature edge retained count mismatch", errors)
    check(as_int(rows["feature_route_near_duplicate"]["rejected_after_fix"]) == EXPECTED["feature_edges_rejected"], "feature edge rejection count mismatch", errors)
    check(as_int(rows["confirmed_cross_label_subset"]["corrected_v5_count"]) == EXPECTED["cross_label_subset"], "cross-label subset count mismatch", errors)
    check(EXPECTED["exact_edges"] + EXPECTED["strong_hash_edges"] + EXPECTED["feature_edges_retained"] == EXPECTED["corrected_v5_edges"], "corrected edge composition does not sum", errors)

    # Final row accounting.
    check(accounting[0]["transition"] == "v4_audited_candidate_universe" and as_int(accounting[0]["ending_rows"]) == EXPECTED["audited_images"], "V4 accounting start mismatch", errors)
    check(any(r["transition"] == "v5_group_safe_benchmark" and as_int(r["ending_rows"]) == EXPECTED["v5_images"] for r in accounting), "V5 accounting state missing", errors)
    check(accounting[-1]["transition"] == "final_frozen_benchmark" and as_int(accounting[-1]["ending_rows"]) == EXPECTED["images"], "final accounting mismatch", errors)
    for r in accounting:
        check(as_int(r["starting_rows"]) + as_int(r["delta_rows"]) == as_int(r["ending_rows"]), f"accounting arithmetic mismatch: {r['transition']}", errors)

    # Provenance/source registry contracts.
    check(len(source_registry) == EXPECTED["registered_sources"], "source registry row count mismatch", errors)
    ontology = {r["ontology_state"]: r for r in ontology_history}
    check(as_int(ontology["historical_label_map"]["value"]) == 249, "historical label-map size mismatch", errors)
    check(as_int(ontology["historical_class_index"]["value"]) == 80, "historical class-index size mismatch", errors)
    check(as_int(ontology["v4_candidate_ontology"]["value"]) == 121, "V4 candidate ontology size mismatch", errors)
    check(as_int(ontology["v5_operational_ontology"]["value"]) == EXPECTED["classes"], "V5 operational ontology size mismatch", errors)
    check(len({r["registry_key"] for r in source_registry}) == EXPECTED["registered_sources"], "source registry keys are not unique", errors)
    check(sum(as_int(r["rows_in_v4_audited_universe"]) for r in source_registry) == EXPECTED["audited_images"], "source registry does not reconcile to V4 universe", errors)
    check(sum(1 for r in source_registry if as_int(r["rows_in_final_frozen_dataset"]) > 0) == EXPECTED["final_source_families"], "final source-family count mismatch", errors)
    check(sum(as_int(r["final_rows"]) for r in provenance) == EXPECTED["source_family_rows"], "provenance coverage rows do not sum to final dataset", errors)
    exact_rows = sum(as_int(r["final_rows"]) for r in provenance if r["original_source_path_status"] == "available")
    source_only_rows = sum(as_int(r["final_rows"]) for r in provenance if r["original_source_path_status"] == "unknown")
    check(exact_rows == EXPECTED["exact_source_path_rows"], "exact source-path coverage mismatch", errors)
    check(source_only_rows == EXPECTED["source_family_only_rows"], "source-family-only coverage mismatch", errors)
    check({r["ingestion_registry_key"] for r in provenance if r["original_source_path_status"] == "unknown"} == {"plantcity_pk"}, "source-family-only provenance is not confined to PlantCity", errors)
    check(sum(as_int(r["final_rows"]) for r in source_counts) == EXPECTED["images"], "source counts do not sum to final images", errors)
    check(len(source_counts) == EXPECTED["final_source_families"], "final source count table family count mismatch", errors)
    check(sum(as_int(r["train"]) for r in source_split_counts) == EXPECTED["train"], "source-by-split train total mismatch", errors)
    check(sum(as_int(r["val"]) for r in source_split_counts) == EXPECTED["val"], "source-by-split val total mismatch", errors)
    check(sum(as_int(r["test"]) for r in source_split_counts) == EXPECTED["test"], "source-by-split test total mismatch", errors)

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

    # Evidence registry schema/identity requirements.
    required_evidence_cols = {"evidence_id", "stage", "artifact_name", "sha256", "file_size_bytes", "visibility", "claims_supported"}
    check(bool(evidence_registry) and required_evidence_cols.issubset(evidence_registry[0].keys()), "evidence registry schema incomplete", errors)
    check(len({r["evidence_id"] for r in evidence_registry}) == len(evidence_registry), "evidence IDs are not unique", errors)
    check(all(len(r["sha256"]) == 64 for r in evidence_registry), "evidence registry contains invalid SHA-256 field", errors)

    # Repository safety scan.
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

    # Canonical documentation must remain present.
    for doc in ("docs/DATASET_LINEAGE.md", "docs/DATASET_PROVENANCE.md", "docs/DATASET_AUDIT_HISTORY.md"):
        check((ROOT / doc).exists(), f"missing canonical dataset document: {doc}", errors)

    report = {"status": "PASS" if not errors else "FAIL", "errors": errors, "warnings": []}
    if args.json_report:
        Path(args.json_report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1

if __name__ == "__main__":
    raise SystemExit(main())
