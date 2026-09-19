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

### 2. `role = historical_compare`

Required identity:

```json
{
  "schema_version": "1.0",
  "role": "historical_compare",
  "image_count": 117546,
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

For the preferred `EXT-I` route, the six files cover the full 117,546-image V4 audited candidate universe. The runner also accepts a cryptographically bound **partial** comparison package so Track B can still complete as `EXT-S`; any count below 117,546 permanently caps that candidate at stress-test evidence. The DINO matrix is exactly `(image_count, 768)`, finite, L2-normalized, and bound to the frozen audit encoder plus `CTC-v2` deterministic evaluation preprocessing (`53937a6d8e87d18b7de086ecd1c000700d946770c523e50bb85cf124048764c4`). This is the prospective Track-B implementation of the 03R “deterministic evaluation preprocessing” clause; it is not presented as a reconstruction of an undocumented historical feature transform.

Build this package once with `build_trackb_historical_compare.py` (or the generated `trackb_build_historical_compare.ipynb`), publish the output as an immutable private Kaggle Dataset, and attach that fixed dataset version to the final Track-B claim notebook. The builder first verifies the same code attestation frozen for the claim run and writes that attestation SHA into both its build certificate and output manifest. The builder writes ORB arrays in bounded chunks rather than retaining all descriptors in RAM.

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

The final claim notebook remains one clean end-to-end notebook. Input preparation is intentionally separated because source acquisition and the 117,546-image historical feature index are infrastructure, not protected model evaluation. The repository contains only four small preparation utilities:

- `prepare_trackb_core_input.py` — hashes/binds the frozen repository, code attestation, R07 S1/S2/S3, validation-only replay surface, class map, DINO encoder and locks;
- `prepare_trackb_candidate_input.py` — verifies source metadata/counts/semantic gate and creates Candidate A/B input manifests;
- `prepare_trackb_historical_source_input.py` — binds the raw 117,546-image V4 source manifest for one-time indexing;
- `build_trackb_historical_compare.py` — produces the compact SHA/pHash/dHash/DINO/ORB comparison package.

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
