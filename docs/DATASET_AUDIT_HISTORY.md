# CropCop dataset audit history

This document separates historical audit facts from corrected journal-extension interpretation.

## V4 historical duplicate graph

V4 reports **8,672 confirmed relationships**:

- 6,196 exact SHA-256 edges;
- 445 strong-hash near-duplicate edges;
- 2,031 feature-route near-duplicate edges.

The V4 preflight also records **3,233 historical cross-split relationships** (2,876 exact + 357 near), 9,721 borderline pairs, 34 images in cross-label conflict families, and 20 ambiguous duplicate chains. The precheck fails rather than certifying the inherited split.

## V5 correction of the feature route

The 2,031 feature-route edges are re-verified under corrected feature/geometric semantics. V5 retains **1,932** and rejects **99**. The corrected trusted graph is therefore **8,573 edges**.

The 17 confirmed cross-label edges are included within the 8,573 trusted graph. They are **not** `8,573 + 17`.

For manuscript/repository terminology:

- `8,672` = **historical V4 confirmed-edge set before feature-route correction**;
- `8,573` = **corrected V5 trusted graph**;
- `17` = **cross-label subset of the corrected trusted graph**.

## V5 exclusion accounting

Starting from 117,546 audited rows:

- 8,355 direct-cover duplicate removals;
- 34 cross-label duplicate-family quarantine rows;
- 6 hard-quality quarantine rows;
- 1 under-supported-class quarantine row;
- leaves **109,150** rows.

The V5 split is 76,405 / 16,376 / 16,369 with zero audited leakage-group crossings and zero trusted-edge crossings.

## Model-readiness QA and review queue

The model-readiness QA does not silently mutate labels. It produces diagnostic probes and review queues. The recovered 180-image manual-review set is exactly the unique endpoint set of the 122 residual cross-label embedding-similarity pairs. The observed priority score formula reproduces every row:

`4 + 3*embedding_review_flag + soft_label_review_flag + (cleanlab_flag AND low_given_label_probability) + flag_visual_label_outlier_v5`

This formula is reconstructed from the supplied tables; the original generator source code is not present.

## Final manual review

Exactly **43** reviewed images are deleted. All 43 decisions are `DELETE_FROM_DATASET`, all are marked `FLAGGED`, and all use the same severe black-screen/distortion reason. There is no relabeling and no resplitting.

The certified transition is **109,150 → 109,107**. Final split counts are **76,376 train / 16,368 validation / 16,363 test**.

## Historical evidence policy

Historical files are not rewritten to make their counts look current. The repository records:

`historical artifact → identified issue → correction artifact → final interpretation`.

This is why both 8,672 and 8,573 remain discoverable, with different meanings.
