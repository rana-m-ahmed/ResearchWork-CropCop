# CropCop Track B — EAAI External Validation

## Active path

The supported Track-B experiment is now:

- `journal_extension/track_b_r07/TRACKB_EAAI_PROTOCOL_v2.json`
- `journal_extension/scripts/run_trackb_eaai_external_validation.py`
- `journal_extension/kaggle/TrackB_EAAI_External_Validation.ipynb` (generated after the behavior commit is QA-pinned)

Track B is a **cross-source external evaluation** of the frozen CropCop R07
classifier on seven mapped classes from two public datasets. It is not a claim
of external validation across all 120 classes.

### External sources

- GVLiD v5 — DOI `10.17632/wkymf8bhcg.5`, 3,477 images, four grape classes.
- Irish Potato — DOI `10.5281/zenodo.8286529`, 58,709 images, three potato classes.

The already completed Notebook-00 materialization is reused. Notebook 00 must
not be rerun merely to execute the simplified Track-B experiment.

## Scientific contract

The active protocol freezes, before external inference:

- R07 S1/S2/S3 checkpoint identities;
- the seven source-to-CropCop class mappings;
- CTC-v2 deterministic evaluation preprocessing;
- the native 120-class output space;
- no external retraining, fine-tuning or calibration;
- no mapped-subset logit renormalization;
- out-of-mapped-scope predictions count as errors;
- mapped-scope Macro-F1 as the primary metric;
- accuracy, balanced accuracy, per-class metrics, confusion matrices and
  out-of-mapped-scope rate as secondary metrics;
- paired, class-stratified 5,000-replicate percentile bootstrap for primary
  Macro-F1 uncertainty;
- the V1 test surface remains closed.

## Strict-decode data quality

Before any model is loaded, every external file is hashed and strictly decoded with Pillow. Files that cannot be decoded are never repaired or silently ignored: they are written to `decode_invalid_manifest.csv` with path, label, raw SHA-256 and decode error, and are excluded from all metric cohorts. Published source counts remain frozen; metric denominators use the strictly decodable subset. The v2 amendment was frozen after a prediction-free v1 audit failure and before any external model prediction or metric was observed.

## Leakage screening and cohorts

The historical comparison package contains the frozen V1 **train + validation**
surface only (92,744 rows). The active leakage audit uses exact raw SHA-256.

Three cohorts are reported:

1. **full_published** — all valid mapped source images;
2. **leakage_clean_primary** — excludes exact raw-SHA matches to V1 train+val;
3. **exact_deduplicated_sensitivity** — leakage-clean plus one deterministic
   representative for each identical external raw-SHA group.

No DINO, ORB, homography or geometric-family graph is required by the active
protocol.

## Why the prior path was retired

Before any protected external performance result was observed, the prior
qualification design repeatedly exceeded the Kaggle execution budget. The
Irish Potato audit generated multi-million-pair geometric verification work
that was not proportionate to the paper claim.

The amendment therefore preserves the model, mappings, metric definitions and
V1-test closure while replacing the expensive near-duplicate qualification
machinery with exact-content leakage screening.

## Legacy / forensic only

The following mechanisms are **not supported for new Track-B runs**:

- v5 qualification -> claim orchestration;
- EXT-I / EXT-S / EXT-X runtime grading;
- DINO top-k duplicate candidate generation;
- ORB/BFMatcher/homography pair verification;
- qualification-dataset publication;
- claim leases and Kaggle attempt ledgers;
- Kaggle-owner-bound execution;
- staged v6 qualification/checkpoint execution.

Their history is retained in Git for provenance. They are not dependencies of
the EAAI external-validation runner.

## Execution

The new Kaggle notebook requires the two already-materialized Track-B datasets:

- infrastructure bundle containing `core` and `historical_compare`;
- external bundle containing `gvlid_v5` and `irish_potato`.

No Kaggle API token or account-specific owner value is part of the scientific
protocol.

A single supported run performs:

1. source/protocol/checkpoint validation;
2. exact SHA-256 leakage screening;
3. S1/S2/S3 native-120 inference on both external datasets;
4. full, leakage-clean and exact-deduplicated metric computation;
5. paired bootstrap confidence intervals;
6. paper-table and per-image evidence export;
7. final hash manifest.

Scientific performance is never used as an execution gate. Low external
performance is a result to report, not a reason to fail or redesign the run.

## Definition of done

Track B is complete when the final runner produces
`PASS_TRACKB_EAAI_EXTERNAL_VALIDATION` and the output package contains:

- protocol and environment records;
- source authority and leakage audit;
- per-image predictions for S1/S2/S3;
- full/clean/deduplicated metrics;
- primary confusion matrices;
- paired bootstrap intervals;
- `paper_table.csv`;
- `summary.json`;
- `TRACKB_FINAL_MANIFEST.json`.

No further Track-B infrastructure should be added unless required by a
reviewer or by a concrete reproducibility defect.
