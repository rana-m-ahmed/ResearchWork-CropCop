# CropCop Track B — R07 External Validation Infrastructure

This namespace implements the audited Track-B v2.1-QA design as a lean, fail-closed Kaggle workflow.

## Scientific boundary

Track B evaluates the already-selected R07 ConvNeXt-Tiny family on two prospectively fixed public external candidates. It does not train, tune, select, replace, or rank new model states. The consumed V1 test is forbidden. Both external candidates are fully audited and sealed before any external R07 forward pass.

The classifier states are exactly R07-S1/S2/S3. The historical DINOv3 ConvNeXt-Tiny checkpoint is an audit-only feature encoder. The historical MobileNetV4/PTE lineage remains preprint context and is not rerun here.

## Kaggle runtime

The claim-producing notebook targets Kaggle **T4x2** but intentionally uses `cuda:0` only. There is no DDP/multi-GPU training and no dependency on a second GPU. This avoids multi-process failure modes while retaining GPU acceleration for DINO feature extraction, nearest-neighbor search, R07 replay, and external inference.

Kaggle currently provides a 12-hour CPU/GPU notebook limit and 20 GB auto-saved `/kaggle/working` storage. The P100 option was retired on 2026-09-15, so this workflow does not depend on it. Use Kaggle's Dependency Manager to install the exact `requirements-trackb.lock.txt` stack before a clean **Save & Run All**. The claim run itself should not need Internet.

## Four immutable Kaggle input datasets

The notebook discovers inputs by scanning `/kaggle/input/**/TRACKB_INPUT_MANIFEST.json`; Kaggle dataset slugs therefore do not need to be hard-coded.

### 1. `role = core`

Required manifest keys:

```json
{
  "schema_version": "1.0",
  "role": "core",
  "v1_validation_root": "relative/path/to/validation/image/root",
  "dino_factory_source_root": "relative/path/to/frozen/repository/source/root",
  "files": {
    "downstream_authority": {"path": "...json", "sha256": "..."},
    "execution_lock": {"path": "...json", "sha256": "..."},
    "code_attestation": {"path": "TRACKB_CODE_ATTESTATION_v1.json", "sha256": "..."},
    "class_map": {"path": "class_to_idx.json", "sha256": "46f7..."},
    "v1_manifest": {"path": "final_manifest.csv", "sha256": "bdb8..."},
    "r07_s1": {"path": "...ckpt", "sha256": "dc7f..."},
    "r07_s2": {"path": "...ckpt", "sha256": "199a..."},
    "r07_s3": {"path": "...ckpt", "sha256": "621c..."},
    "r07_s1_run_record": {"path": "...json", "sha256": "..."},
    "r07_s2_run_record": {"path": "...json", "sha256": "..."},
    "r07_s3_run_record": {"path": "...json", "sha256": "..."},
    "dino_checkpoint": {"path": "...", "sha256": "74b4..."},
    "dino_factory_manifest": {"path": "TEACHER_FACTORY_BUNDLE.json", "sha256": "..."}
  }
}
```

`v1_validation_root` is the directory **above** the frozen `val/` subtree because the certified manifest paths already begin with `val/`. The core package may contain only that `val/` subtree beneath the root; a sibling `test/`, `v1_test/`, `test_consumed/`, or `DS-V1-TEST-CONSUMED/` directory is a hard packaging failure. `TRACKB_CODE_ATTESTATION_v1.json` binds the load-bearing Track-B and inherited Track-A source files by Git-blob identity, and its SHA-256 is itself frozen in the execution lock. Use `prepare_trackb_core_input.py` to verify these identities and catch the common `val/val/...` packaging mistake before upload.

The execution lock also binds the three authoritative replay records and the DINO factory manifest:

- R07 S1 run record: `0f403138ee43b1f0e464f092b51cf4f80a13233c9bc8ed4b17e86cf7446c5516`
- R07 S2 run record: `0e48fdc0907a44042110f146ec27f796ea3d185fad1e2fc008cd1332bfd0ae15`
- R07 S3 run record: `6ba1f3a7348f5c4ba0347621edef0e75e311b89bd38ff13ec4288cd81cf70050`
- DINO factory manifest: `df70164ef227878353dde5430e8e0386b8853b53a2b66c20602e4cecd4dab7f1`

For operator use, prefer `trackb_build_core_package.ipynb`. It accepts the already-existing Final-V1, R07-S1/S2/S3, and Secondary-G1-v2 Kaggle datasets, discovers checkpoint/run-record/factory files by authoritative SHA-256, copies only the 16,368-image validation surface, runs `prepare_trackb_core_input.py`, and emits a publishable immutable `role = core` directory. The source dataset locators are frozen in that notebook so checkpoint filenames never need to be guessed manually.

### 2. `role = historical_compare`

The **safe executable post-closure route** is a development-surface comparison package:

```json
{
  "schema_version": "1.1",
  "role": "historical_compare",
  "coverage_scope": "V1_TRAIN_VAL_ONLY",
  "image_count": 92744,
  "expected_full_historical_image_count": 117546,
  "ext_i_eligible": false,
  "maximum_evidence_grade": "EXT-S",
  "v1_test_image_bytes_accessed": false,
  "dino_audit_encoder_sha256": "74b4701b8931976c9227845ead50788ae47e3596f575f2817b7352a715f53b79",
  "code_attestation_sha256": "...",
  "features_l2_normalized": true,
  "files": {
    "historical_manifest": {"path": "historical_manifest.csv", "sha256": "..."},
    "dino_features": {"path": "dino_features.npy", "sha256": "..."},
    "orb_offsets": {"path": "orb_offsets.npy", "sha256": "..."},
    "orb_xy": {"path": "orb_xy.npy", "sha256": "..."},
    "orb_desc": {"path": "orb_desc.npy", "sha256": "..."},
    "orb_shapes": {"path": "orb_shapes.npy", "sha256": "..."}
  }
}
```

`historical_manifest.csv` columns:

```text
hist_id,raw_sha256,phash64,dhash64,width,height
```

The runnable builder deliberately uses only the frozen Final-V1 **train + validation** development surface: 76,376 + 16,368 = **92,744** images. It never opens consumed V1-test image bytes. Because this comparison surface is partial relative to the historical 117,546-image audited universe, it permanently caps candidate evidence at `EXT-S`. That is an intentional scientific boundary, not a runtime limitation.

A candidate may reach `EXT-I` only if a complete 117,546-image comparison representation that was created before V1-test closure is genuinely recovered, cryptographically verified, and attached directly as `role = historical_compare`. Do **not** regenerate such a representation now by reopening raw V1-test image bytes. The deprecated `prepare_trackb_historical_source_input.py` fails closed for this reason.

Build the safe package once with `build_trackb_historical_compare.py` (or `trackb_build_historical_compare.ipynb`), publish its output as an immutable private Kaggle Dataset, then attach that exact dataset version to the final Track-B claim notebook. The DINO matrix is exactly `(image_count, 768)`, finite, L2-normalized, and bound to the frozen audit encoder plus CTC-v2 deterministic evaluation preprocessing.

### 3. `role = irish_potato`

```json
{
  "schema_version": "1.0",
  "role": "irish_potato",
  "doi": "10.5281/zenodo.8286529",
  "version": "01",
  "data_root": "relative/path/to/extracted/originals",
  "unresolved_lineage": false,
  "files": {
    "source_metadata_record": {"path": "SOURCE_METADATA.json", "sha256": "..."}
  }
}
```

`SOURCE_METADATA.json` must contain the observed DOI, version, source URL, retrieval timestamp, non-empty observed license/access text, `known_historical_contributor_relationship` (boolean), and `lineage_review_status` (`PASS_NO_KNOWN_RELATIONSHIP` or `RESIDUAL_UNCERTAINTY`). The runner requires exactly 58,709 images with source supports 17,772 `earlyblt`, 20,438 `healthy`, and 20,499 `lateblt`. Use `prepare_trackb_candidate_input.py` before publishing the Kaggle dataset.

### 4. `role = agrivision_v2`

```json
{
  "schema_version": "1.0",
  "role": "agrivision_v2",
  "doi": "10.17632/8t6k37ztxc.2",
  "version": "2",
  "data_root": "relative/path/to/package/root",
  "unresolved_lineage": false,
  "files": {
    "source_metadata_record": {"path": "SOURCE_METADATA.json", "sha256": "..."},
    "mapping_semantic_record": {"path": "TOMATO_MOSAIC_MAPPING.json", "sha256": "..."}
  }
}
```

The package must contain exactly one `Original_Images` directory with 5,266 images. `TOMATO_MOSAIC_MAPPING.json` must be frozen before execution and contain:

```json
{
  "status": "PASS",
  "mapping": "Tomato Mosaic -> tomato_mosaic_virus"
}
```

If that semantic record is absent or fails, Candidate B becomes `EXT-X`; the notebook does not invent a narrower post-hoc mapping. `SOURCE_METADATA.json` uses the same source/lineage fields required for Candidate A. `prepare_trackb_candidate_input.py` verifies the 5,266-original count and mapped source supports before the dataset is published.


## Lean input-preparation utilities

The final claim notebook remains one clean end-to-end notebook. Input preparation is intentionally separated because source acquisition and historical comparison indexing are infrastructure, not protected model evaluation. The executable post-closure index is train+validation-only; a full 117,546 representation is accepted only if it already exists from before V1-test closure. The repository contains a small, explicit preparation layer:

- `trackb_build_core_package.ipynb` + `build_trackb_core_package.py` — assemble the immutable core package from the exact existing Kaggle datasets using hash discovery and validation-only copying;
- `prepare_trackb_core_input.py` — hashes/binds the assembled repository, code attestation, R07 S1/S2/S3 checkpoints and authoritative run records, validation-only replay surface, class map, DINO encoder/factory and locks;
- `prepare_trackb_candidate_input.py` — verifies source metadata/counts/semantic gate and creates Candidate A/B input manifests;
- `prepare_trackb_historical_source_input.py` — deliberately disabled after V1-test closure so the full raw 117,546-image surface cannot be reconstructed by reopening consumed test images;
- `build_trackb_historical_compare.py` — produces the safe 92,744-image train+validation SHA/pHash/dHash/DINO/ORB comparison package, with an automatic `EXT-S` ceiling.

No database, workflow service, GitHub token, distributed training layer, or resume state machine is required. A technical failure of the final claim notebook is retried by rerunning the unchanged frozen notebook from the beginning.

## Final notebook stages

1. authority/environment and exact R07 validation replay;
2. source verification for both candidates;
3. prediction-blind within-candidate family and historical-overlap audit; decoded failures are preserved before seal, and the geometric audit persists accepted relations plus deterministic rejection-funnel counts rather than gigabytes of low-value rejected-pair rows;
4. deterministic `EXT-I/EXT-S/EXT-X` grade + immutable seal for both candidates;
5. prediction firewall;
6. S1/S2/S3 native 120-way inference only for `EXT-I`/`EXT-S` candidates, with mapped-class precision/recall/F1 plus sparse native-120 confusion evidence and explicit out-of-mapped prediction rate;
7. fixed 5,000-replicate family bootstrap using shared resample indices;
8. independent re-read/recompute QA and `TRACK_B_CLOSED`.

A bad metric is a valid scientific result. It never changes the candidate, mapping, grade, row set, model seed, or retry policy.

## Output

The notebook writes only to `/kaggle/working/trackb_r07` and emits:

- `TRACKB_PREDICTION_FIREWALL.json`
- candidate audit/seal evidence
- S1/S2/S3 row-level predictions for claim-producing candidates
- seedwise metrics and three-seed summary
- frozen bootstrap output
- `TRACKB_FINAL_QA.json`
- `TRACKB_FINAL_CLOSURE.json`
- `TRACKB_COMPLETE_EVIDENCE.zip`
- `TRACKB_PUBLIC_EVIDENCE.zip`
- `TRACKB_PACKAGE_MANIFEST.json`

No GitHub credential is required and the notebook never pushes to the repository.
