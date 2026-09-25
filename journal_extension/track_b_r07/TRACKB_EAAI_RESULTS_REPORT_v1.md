# CropCop Track B — Final EAAI Results Report

## Status

**Track B computational execution is complete and evidence-locked.**

- Protocol: `TRACKB_EAAI_EXTERNAL_VALIDATION_v2`
- Frozen behavior commit: `7d9c09dfc6ec6bad640ee8af429abd1f6ab688d1`
- Final manifest: `90f4acd8858a6a7550775ea15c8ba87108c85e788086711bfaeacaf1fbd8bafa`
- Evidence ZIP SHA-256: `22a6c865ead6f28319a9dc8a4168f3ff7fb60108339710dcbd99e1aa047981a3`
- External predictions: **186,546**
- V1 test accessed: **no**
- New training/fine-tuning/calibration: **no**

## Data quality and leakage

No exact raw-SHA overlap was found between either external dataset and the frozen V1 train+validation surface.

| Dataset | Published files | Strictly decodable | Invalid | Exact V1 train+val overlap |
| --- | ---: | ---: | ---: | ---: |
| GVLiD v5 | 3,477 | 3,477 | 0 | 0 |
| Irish Potato | 58,709 | 58,705 | 4 | 0 |

The four invalid Irish Potato files were excluded before any model loading. Two were `earlyblt` and two were `healthy`; the four paths represent three unique raw payloads because the two invalid early-blight files share the same raw SHA-256.

## Primary external results

The primary cohort is the strictly decodable, exact-leakage-clean cohort. Predictions were made in the **native 120-class output space**; out-of-mapped-scope predictions remained errors.

| Dataset | N | Macro-F1 mean ± SD | 95% paired-bootstrap CI for 3-seed mean | Accuracy mean ± SD | Balanced accuracy mean ± SD | OOS prediction rate mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GVLiD v5 | 3,477 | 0.3315 ± 0.0162 | [0.3241, 0.3387] | 0.3867 ± 0.0043 | 0.3487 ± 0.0048 | 0.3637 ± 0.0504 |
| Irish Potato | 58,705 | 0.4160 ± 0.0326 | [0.4129, 0.4192] | 0.2936 ± 0.0314 | 0.2953 ± 0.0324 | 0.6582 ± 0.0162 |

These results show **substantial cross-source domain shift**. They should not be framed as strong external validation of all 120 classes.

## Class-level failure modes

### GVLiD

Across seeds, `grape_healthy` transfers relatively well (F1 about 0.77–0.82), and `grape_black_rot` is moderate (about 0.48–0.54). In contrast:

- `grape_esca` is effectively not recognized (F1 ≈ 0.00).
- `grape_leaf_blight` is also nearly absent (F1 ≈ 0.00–0.06).
- About 32–42% of predictions, depending on seed, leave the four-class grape scope.

The problem is therefore not simply random error; it is strongly class-specific.

### Irish Potato

Transfer is strongest for `potato_late_blight` (F1 ≈ 0.55–0.69) and moderate for `potato_early_blight` (≈ 0.36–0.53). `potato_healthy` has extremely high precision but very low recall (≈ 0.07–0.12), indicating that the model predicts this external healthy class only rarely.

The dominant limitation is out-of-scope routing: roughly **64–67%** of primary Irish Potato predictions leave the three-class potato scope.

## Exact-duplicate sensitivity

Both external sources contain substantial exact repetition.

| Dataset | Valid N | Exact-deduplicated N | Duplicate groups | Rows in duplicate groups |
| --- | ---: | ---: | ---: | ---: |
| GVLiD v5 | 3,477 | 2,409 | 446 | 1,514 |
| Irish Potato | 58,705 | 38,554 | 9,632 | 29,783 |

After exact deduplication:

- GVLiD mean Macro-F1 falls from **0.3315 → 0.2507** (−0.0808; about −24.4% relative).
- Irish Potato mean Macro-F1 falls from **0.4160 → 0.3646** (−0.0514; about −12.4% relative).
- Irish Potato accuracy falls from **0.2936 → 0.2341**, while OOS rate rises from **0.6582 → 0.7177**.

This means repeated exact images materially inflate the apparent external performance, especially for the potato dataset. The deduplicated sensitivity analysis should therefore be reported, not hidden.

## Paper-facing interpretation

A defensible statement is:

> Cross-source evaluation on GVLiD v5 and the Irish Potato dataset, covering seven classes represented in CropCop's native 120-class output space, found no exact overlap with the frozen training/validation surface but revealed substantial domain shift. Mean Macro-F1 was 0.331 on GVLiD and 0.416 on Irish Potato, with high out-of-scope prediction rates and pronounced class-specific failures. Exact-content deduplication further reduced performance, showing that repeated images in the external sources can inflate aggregate estimates.

The result should be framed as **external robustness evidence and a clearly quantified limitation**, not as a broad claim of field-ready generalization.

## Track-B closure decision

**No additional Track-B notebook or Kaggle run is justified.**

The remaining work is manuscript integration:

1. report the primary leakage-clean results;
2. include the exact-deduplicated sensitivity result;
3. disclose four malformed Irish Potato files and their pre-inference exclusion;
4. state explicitly that the evaluation covers seven mapped classes, not all 120;
5. discuss class-specific domain shift and high OOS routing as limitations;
6. retain the final evidence ZIP and manifest hashes as the reproducibility record.

Track B is therefore **computationally closed**.
