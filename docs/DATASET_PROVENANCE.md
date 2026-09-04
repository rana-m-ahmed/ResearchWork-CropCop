# CropCop dataset provenance

## What is now recoverable

The final V5/freeze manifests collapse the source field to `main`, but the earlier recovered registry and manifest preserve enough identity to reconstruct the original ingestion source family conservatively.

For the **109,107 final rows**:

- **84,146 rows (77.1%)** join exactly to the recovered historical manifest and therefore recover the exact original source path, source-family key, domain tag, weight metadata, and historical declared split.
- **24,961 rows (22.9%)** do not have a surviving per-image `src_path` join. Exhaustive recovered-registry/final-accounting reconciliation assigns them to **PlantCity Pakistan by unique complement**. This is a reconstructed source-family attribution, not a direct per-image source-row match; the exact original source path is unavailable.
- Final **source-family attribution coverage is 109,107 / 109,107 (100%)**.
- Exact original-source-path coverage is **84,146 / 109,107 (77.1%)**.

This distinction is intentional. The repository does not convert source-family certainty into false per-image path certainty.

## Reconstruction method

1. Parse final `record_key` into historical declared split, label, and merged filename.
2. Join one-to-one against the recovered `dataset_manifest.csv` using historical split + label + merged filename.
3. Preserve exact joins as `verified_manifest_join` with confidence `verified_exact`.
4. Reconcile the unmatched historical-test complement against `final_cropcop_registry.csv`, source counts, label counts, and the V4 candidate universe.
5. The manifest-to-V4 delta is **26,018 = 26,017 PlantCity + 1 Bangladesh**. The Bangladesh final rows are all accounted for by exact manifest joins; `rice_neck_blast` is a separate V5 quarantine and is not this complement row.
6. Classify the remaining 24,961 final PlantCity rows as method `registry_complement_reconstruction`, confidence `reconstructed_unique_complement`, with original source path `unknown`.

The complement attribution is conditional on completeness of the recovered source universe, absence of an untracked later source-family introduction, and unambiguous accounting of all other final source families. Those conditions are machine-checked where the public evidence permits. Confidence definitions are frozen in `data_card/provenance/provenance_confidence_definitions.json`.

## Final source-family composition

| Source family | Final rows |
| --- | ---: |
| plantvillage | 39,973 |
| plantcity_pk | 33,872 |
| sugarcane | 13,671 |
| plant100k | 7,903 |
| abcgmp | 6,833 |
| bangladesh | 4,092 |
| plantdoc | 2,559 |
| coleaf | 119 |
| corn | 67 |
| rice | 18 |

The exact split-by-source table is in `data_card/provenance/final_source_split_counts.csv`.

## Holdout interpretation

PlantCity was historically marked as a holdout source, but later V5 reconstruction pools the audited universe and performs a new group-safe split. Therefore **PlantCity must not be described as an independent final external holdout**. It is a contributing source family in the internal benchmark.

## Licensing boundary

Source-family reconstruction does **not** establish image-level redistribution rights. Original URL/licence/annotation authority is not complete for every image. The repository therefore publishes the source registry and aggregate provenance coverage, but keeps the row-level provenance ledger and raw source paths outside public Git.

The restricted row-level ledger has 109,107 rows and SHA-256 `fe596ead93eff17a5b976b69e347ef614dd38e6d90efd41ffa0d128c54fc7181`. Its original complement-confidence label is preserved as historical reconstruction output; the public interpretation supersedes `verified_exact_source_family` with `reconstructed_unique_complement` rather than rewriting restricted bytes.
