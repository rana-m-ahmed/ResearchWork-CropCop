# Provenance status

The recovered audit/manifests materially improve the provenance boundary. All **109,107 final rows have a reconstructed source-family attribution within the recovered source universe**.

- **84,146 rows (77.1%)**: deterministic one-to-one historical-manifest join; confidence `verified_exact`; exact historical `src_path` available.
- **24,961 rows (22.9%)**: PlantCity Pakistan source family assigned by exhaustive unique-complement reconciliation; confidence `reconstructed_unique_complement`; exact historical `src_path` unavailable.

The historical manifest-to-V4 delta is **26,018 rows = 26,017 PlantCity + 1 Bangladesh**. This is distinct from the separate **16-row rice attrition** between the 117,562-row recovered upstream registry and the 117,546-row V4 audited universe.

Complement-derived attribution is conditional on completeness of the recovered source universe, absence of an untracked later source-family introduction, and complete accounting of all other final source families. It must not be described as direct per-image provenance.

This is source-family reconstruction, not an image-rights audit. Complete original URLs, licences, annotation authority, geography, cultivar, and severity do not exist for every image, so redistribution rights remain unresolved at image level.

See `docs/DATASET_PROVENANCE.md`, `data_card/provenance/provenance_confidence_definitions.json`, and `data_card/provenance/`.
