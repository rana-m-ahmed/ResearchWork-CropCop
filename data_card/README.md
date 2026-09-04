# Dataset card assets

This directory records the frozen CropCop Final v1 identity, reconstructed source provenance, ontology/lineage history, audit accounting, split counts, and cryptographic fingerprints. The image corpus is not redistributed.

## Canonical dataset evidence

- `lineage/lineage.json` — machine-readable chronology and fingerprints.
- `lineage/source_registry.csv` — recovered 15-source historical registry with survival/final-row accounting.
- `lineage/ontology_history.csv` — machine-readable ontology-state history and evidence authority.
- `provenance/provenance_coverage.csv` — final provenance method/coverage table.
- `provenance/provenance_confidence_definitions.json` — epistemic definitions for direct versus complement-derived source attribution.
- `provenance/historical_test_complement.csv` — explicit 91,528→117,546 source-count reconciliation.
- `provenance/external_evaluation_exclusion_registry.csv` — historical contributors blocked from being treated as independent external cohorts without new independence proof.
- `lineage/ontology_transition_registry.csv` — machine-readable raw-label → harmonization → candidate → final ontology transitions.
- `audit/fingerprint_verification_status.json` — states which cryptographic identities are publicly recomputable versus cross-file consistency locks.
- `provenance/final_source_counts.csv` and `final_source_split_counts.csv` — source-family composition of the final frozen benchmark.
- `audit/duplicate_edge_history.csv` — V4 historical versus V5 corrected duplicate semantics.
- `audit/final_accounting.csv` — 117,546 → 109,150 → 109,107 row accounting.
- `audit/manual_review_summary.csv` — aggregate final manual-review deletion accounting by split/label/reason.
- `audit/model_readiness_manual_review_reconstruction.json` — recovered semantics of the 180-row review queue.

See `docs/DATASET_LINEAGE.md`, `docs/DATASET_PROVENANCE.md`, and `docs/DATASET_AUDIT_HISTORY.md` for the narrative evidence chain.
