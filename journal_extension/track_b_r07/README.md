# CropCop Track B — R07 External Validation v3

Track B evaluates the frozen Track-A-selected R07 ConvNeXt-Tiny family on prospectively fixed external cohorts. It does not train, tune, reselect, or replace model states. The consumed V1 test remains closed.

## Current pre-inference status

The v3 remediation implementation has passed repository-level QA, but **protected external inference is not yet authorized**. Real Kaggle qualification Q1/Q2/Q3 must pass first. See `reports/TRACKB_REMEDIATION_REGISTER_v1.*` and `reports/TRACKB_EXECUTION_READINESS_v3.*`. The older v2 readiness reports are retained only as historical records.

## Operator entry point

Use only:

`journal_extension/kaggle/trackb_r07_master.ipynb`

One-time Kaggle setup:

- enable Internet;
- select T4 x2;
- add secret `KAGGLE_API_TOKEN`;
- add secret `CROPCOP_GITHUB_TOKEN`.

No Track-B input dataset needs to be manually uploaded or attached. The supported qualification notebook clones the frozen Track-B source at exact commit `d2796cb69daec26a94113e5909719eb353b94832`, loads only `KAGGLE_API_TOKEN`, applies `requirements-trackb.lock.txt` directly to Kaggle's active Python interpreter **before any scientific package import**, and launches `run_trackb_r07_master.py` in qualification mode from a fresh subprocess. Qualification intentionally skips GitHub write credentials and Git push preflight because it produces no GitHub publication and no protected R07 external predictions.

For the later protected claim/publication run, `CROPCOP_GITHUB_TOKEN` must be a valid GitHub PAT with repository write permission; that claim path performs the non-mutating Git push preflight before final publication.

Do not create a Python `venv` on Kaggle for Track B: the Kaggle system interpreter may not provide a working `ensurepip` path. Do not manually install Torch either; the master notebook owns the one-time exact-lock repair.

The older `trackb_r07_end_to_end.ipynb` is retained only as an inspectable internal claim-engine/recovery surface. It is not the supported operator workflow.

## Kaggle runtime lock

Track B reuses the already-qualified Track-A clean-session package-repair pattern rather than a virtual environment. The scientific stack is repaired once from `requirements-trackb.lock.txt`, then all scientific imports/execution occur only in fresh subprocesses.

Frozen execution packages include:

- Python 3.12.13
- torch 2.12.1
- torchvision 0.27.1
- timm 1.0.26
- NumPy 2.5.2
- Pillow 12.3.0
- transformers 5.0.0
- huggingface-hub 1.30.0
- safetensors 0.8.0
- opencv-python-headless 4.13.0.92
- Kaggle CLI 2.2.4

OpenCV 4.13.0.92 is a pre-results technical compatibility re-lock from 4.12.0.88 because the older wheel conflicts with the frozen NumPy 2.5.2 dependency on Python 3.12. No protected external prediction preceded this re-lock.

## Frozen scientific cohorts

### External grape cohort — GVLiD v5

- Candidate ID: `gvlid_grape`
- Input role: `gvlid_v5`
- DOI: `10.17632/wkymf8bhcg.5`
- Version: 5
- Expected images: 3,477
- Scope: `SCOPE-GRAPE-4`

Frozen mapping:

| Source label | CropCop label |
|---|---|
| Black Rot | `grape_black_rot` |
| Esca | `grape_esca` |
| Healthy | `grape_healthy` |
| Leaf Blight | `grape_leaf_blight` |

The source publication/package has a one-image arithmetic inconsistency between its stated total and one displayed class-count table. Track B therefore treats the exact acquired version-5 bytes as authority: total identity must reconcile to 3,477 and observed class supports are enumerated and frozen before any R07 prediction. The cohort contains both in-situ vineyard and ex-situ/controlled-background imagery; Track B does not describe every representative as a biologically independent field leaf.

### Complementary stress cohort — Irish Potato Version 01

- Candidate ID: `irish_potato`
- Input role: `irish_potato`
- DOI: `10.5281/zenodo.8286529`
- Version: `01`
- Expected images: 58,709
- Scope: `SCOPE-POTATO-3`

Frozen mapping:

| Source label | CropCop label |
|---|---|
| `earlyblt` | `potato_early_blight` |
| `healthy` | `potato_healthy` |
| `lateblt` | `potato_late_blight` |

Expected source support remains 17,772 / 20,438 / 20,499 respectively.

Agri-Vision Bangladesh is retired from Track-B v2 **before protected external inference**. It is not a fallback candidate if either frozen v2 cohort performs poorly.

## Why the two-cohort design is locked this way

Candidate selection used only pre-external evidence: ontology compatibility, acquisition/source independence, field realism, public provenance, support, frozen R07 internal class reliability, and complementarity between a stronger confirmatory scope and a harder stress scope. External R07 performance cannot alter the cohort pair, mappings, model family, or seed set.

Track B evaluates exactly R07-S1/S2/S3. The historical DINOv3 ConvNeXt-Tiny checkpoint is an audit-only feature encoder. Historical MobileNetV4/PTE remains contextual preprint evidence and is not the Track-B classifier.

## Automated master workflow

The master controller performs these stages in order:

1. load `KAGGLE_API_TOKEN` and verify the exact frozen source checkout for qualification;
2. repair any stock Kaggle package drift to the exact Track-B lock and verify the repaired stack/CUDA from a fresh subprocess;
3. auto-detect the Kaggle owner authenticated by `KAGGLE_API_TOKEN` and verify access to every frozen Final-V1, R07-S1/S2/S3, and DINO source dataset before large downloads begin;
4. automatically download those frozen assets;
5. build the immutable `core` package containing only the allowed validation replay surface and frozen model/audit identities;
6. build or reuse an exact private `historical_compare` cache over V1 train+validation only;
7. delete large temporary source downloads to control working-disk pressure;
8. acquire GVLiD v5 and Irish Potato from their authoritative public repositories;
9. validate/package both external candidates;
10. run prediction-blind exact/pHash/dHash/DINO/ORB family and historical-overlap audits;
11. freeze each candidate grade and immutable seal;
12. write `TRACKB_PREINFERENCE_QUALIFICATION.json` and stable `qualification_science_sha256`;
13. independently verify the pre-inference qualification and assert zero protected external predictions;
14. stop. The supported qualification notebook does **not** run S1/S2/S3 external inference or publish claim results.

The later protected claim run is a separate exact-SHA execution authorized only after Q1/Q2/Q3 review and requires the reviewed `qualification_science_sha256`.

The controller automatically garbage-collects large source downloads between stages rather than requiring the operator to create and reattach multiple intermediate datasets.

## Publication durability

GitHub publication is an operational dissemination layer, not part of the scientific
estimator. The notebook proves Git write access before compute. After terminal local `TRACK_B_CLOSED` + QA PASS, the complete restricted evidence is published to private Kaggle and fully round-trip byte-verified before GitHub publication. Both private Kaggle publication and GitHub public-safe publication receive four attempts with bounded retry delays.

If GitHub becomes unavailable after scientific closure, Track B does **not** discard
or reinterpret results. The automation receipt records
`TRACK_B_CLOSED_PRIVATE_EVIDENCE_ARCHIVED_GITHUB_PUBLICATION_FAILED`, and the
private evidence remains the durable recovery authority. Credential failures should
normally be caught before dependency repair by the early dry-run preflight.

## Historical comparison boundary

The executable post-closure historical comparison uses only frozen Final-V1 train + validation:

- train: 76,376
- validation: 16,368
- total: 92,744
- consumed V1-test image bytes accessed: **false**

This route has a permanent maximum grade of `EXT-S` because it is not the complete 117,546-image historical audited universe.

`EXT-I` is possible only if a complete pre-test cryptographic comparison representation of all 117,546 historical images is genuinely recovered and independently verified. The repository must not recreate that surface now by reopening consumed V1-test images.

## Prediction firewall

For each candidate, the following must be frozen before R07 inference:

- DOI/version/source metadata;
- raw-image manifest;
- frozen source→CropCop mapping;
- decode-failure ledger;
- family graph and deterministic family representatives;
- exact/near historical-overlap evidence;
- family support;
- terminal `EXT-I`, `EXT-S`, or `EXT-X` grade;
- all three R07 checkpoint hashes;
- class-map and preprocessing identity;
- bootstrap configuration;
- immutable seal hash.

`EXT-X` produces no claim-making protected classifier inference.

## Evidence publication boundary

GitHub receives only allow-listed text evidence after terminal QA. The publication layer rejects model/checkpoint archives, raw images, secrets, credentials, private runtime paths, and row-level logits.

Complete restricted evidence is persisted to a private Kaggle dataset under the owner authenticated by the API token. The master notebook does not hard-code a personal Kaggle owner.

## Claim boundary

- `EXT-I`: only audit-bounded source-independent wording for that candidate's frozen mapped scope.
- `EXT-S`: public cross-dataset / external-domain stress-test wording only.
- `EXT-X`: no claim-producing performance result.

No Track-B outcome establishes universal 120-class field generalization, agronomic treatment readiness, or device/runtime performance.

## Core frozen identities

- dataset manifest SHA-256: `bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2`
- class-map SHA-256: `46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2`
- DINO audit checkpoint SHA-256: `74b4701b8931976c9227845ead50788ae47e3596f575f2817b7352a715f53b79`
- R07-S1: `dc7fea2e8db91bf1fc023cb5e792b23b67659edec22e10a7c1d46b4010db3974`
- R07-S2: `199afb9f7043e599fbb2239fb3babcfb431324a3c3219250ae6dd0359d8bc310`
- R07-S3: `621c2e6cfecd23da21b4f17d2244bd068b5a3602ee5240f0dcc95ae4360beb37`

Current authority:

- `EAAI-JE-TRACKBC-R07-DOWNSTREAM-v3`
- `TRACKB_R07_EXECUTION_LOCK_v3`
- `TRACKB_CODE_ATTESTATION_v3`
- `TRACKB_EXTERNAL_LINEAGE_REVIEW_v1`
