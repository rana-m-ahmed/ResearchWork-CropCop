# Stage 01A-G1P-v2 — Final Source Closure Report

- **Artifact:** `01A_G1P_V2_FINAL_SOURCE_CLOSURE_REPORT.md`
- **Date:** 2026-09-06
- **Repository:** `rana-m-ahmed/ResearchWork-CropCop`
- **PR:** #8
- **Status:** **REPOSITORY REMEDIATION CLOSED — REAL G1 INPUT READINESS NOT YET EXECUTED**

## 1. Authority stack

### Stage-03R scientific authority

- ID: `EAAI-JE-SDL-v2.1-QA`
- SHA-256: `aab17b65b0873dcb1ecedb061eb02ff60ccb09f8b830184f5e2231a600278f74`

### Stage-04 base execution architecture

- ID: `EAAI-JE-REA-v2.2-LEAN`
- SHA-256: `a1bc1f8bb2d7a133a46c3f59f83dd2ba8ca3bf0fa8d77ae323553276931bc079`

### Stage-04A additive MGPU amendment

- ID: `EAAI-JE-MGPU-A1`
- SHA-256: `3f08f2dbe7e83143c7e6f1fcf6732c7d88086295a600a74b323b89b46caed3f6`

No scientific authority was changed by Stage-01A-G1P-v2.

## 2. Superseded historical execution source

Previous canonical execution source:

`fe88e426b4698977d65efe9702f1d48cf5ff96a3`

Previous dependency lock:

`767859a3cb1b37c0e50f753aa8c7a5c400424417cba3eaa59f0ab617ff18f701`

That source remains historically important because real clean-Batch:

- Smoke A passed;
- fresh Smoke B passed;
- dual-GPU smoke passed;
- independent audits passed.

Those runs remain valid evidence only for the exact historical source/dependency pair above.

They do **not** qualify the new G1P-v2 source.

### Why `fe88e...` was superseded

The final remediation independently reproduced and then repaired a real G1→G2 launch-contract defect:

`validate_g1_barrier.py` required `--dual-gpu-smoke-evidence`, while the downstream `run_lane.validate_g1(...)` consumer did not supply it.

The remediation also closed pre-science reproducibility gaps that necessarily changed execution source:

- exact historical DINO teacher reconstruction;
- canonical EMA-state enforcement;
- hash-bound historical class-order lineage;
- exact Transformers compatibility dependency;
- canonical teacher-state evidence;
- synthetic direct-vs-adapter parity;
- shared G1 input resolution;
- deterministic official MobileNetV4 preparation/provenance;
- self-contained private G1 bundle;
- G1 seal schema v2 with Smoke-B + dual-smoke chronology;
- shared G1 barrier API;
- parent fail-early full-G1 validation;
- deterministic safe package transport;
- private Kaggle target preflight;
- publication round-trip verification;
- publication-only repair without pair regeneration;
- single-root downstream G1 package recovery;
- non-qualifying real-input readiness entrypoint;
- CPU-defined G1 pair initialization.

No model-quality result was used to choose any remediation.

## 3. Frozen Stage-01A-G1P-v2 execution source

# `b89144d8826b6c61c3be7a91cac08681b1b4a99c`

This is the immutable Stage-01A-G1P-v2 execution source.

It contains the complete execution remediation plus:

`01A_G1P_V2_POST_REFACTOR_SCIENCE_DIFF.md`

It does **not** contain the later canonical wrapper rebinding.

No execution implementation file was modified after this SHA was frozen.

### Exact-head execution-source CI

- GitHub Actions run: **#180**
- Run ID: `34010784369`
- exact expected SHA: `b89144d8826b6c61c3be7a91cac08681b1b4a99c`
- exact observed checkout SHA: same
- conclusion: **SUCCESS**

Run #180 verified:

- Python compile PASS;
- repository validator PASS;
- JE static validator PASS;
- science-diff PASS;
- Stage-04A amendment hash PASS;
- canonical notebook code compile PASS;
- canonical notebook generation consistency PASS;
- forbidden tracked model/checkpoint scan PASS;
- live-looking secret scan PASS;
- active operator-doc validator PASS;
- QA1 Wave A: **18 / 18 PASS**;
- QA1 Wave B: **13 / 13 PASS**;
- QA1 Wave C: **8 / 8 PASS**;
- MGPU suite: **66 / 66 PASS**;
- complete CPU-safe suite: **257 / 257 PASS**;
- exact Python 3.12.13 / Transformers 5.0.0 compatibility job PASS.

## 4. Final execution dependency lock

The final execution dependency-lock SHA is:

`6ea5fb51a0cc39c7940e4aaeb136d0f05214f5c558edd3c47b4cb61c6b516f37`

The pre-results compatibility addition is:

`transformers==5.0.0`

The exact compatibility matrix passed with:

- Python 3.12.13;
- torch 2.12.1;
- torchvision 0.27.1;
- timm 1.0.26;
- numpy 2.5.2;
- Pillow 12.3.0;
- safetensors 0.8.0;
- kaggle 2.2.4;
- huggingface-hub 1.30.0;
- transformers 5.0.0.

Direct offline `DINOv3ConvNextConfig` / `DINOv3ConvNextModel` construction and synthetic strict-load compatibility passed.

## 5. Historical teacher closure

Historical forensics established without contradiction:

- dataset locator: `cropcop-model-rfdv`;
- exact relative checkpoint:
  `cropcop_runs/stage1_dino_tiny_ce_256/checkpoints/best_macro_f1.pt`;
- teacher SHA:
  `74b4701b8931976c9227845ead50788ae47e3596f575f2817b7352a715f53b79`;
- model:
  `facebook/dinov3-convnext-tiny-pretrain-lvd1689m`;
- experiment: `stage1_dino_tiny_ce_256`;
- resolution: 256;
- canonical historical state: **EMA**;
- classifier key: `head.weight`;
- classifier shape: `[120,768]`;
- feature width: 768;
- classes: 120;
- historical index semantics: `class_map_index`;
- output-order transform: `none`.

The frozen factory does not use `from_pretrained()`, does not substitute a different teacher, does not fall back to raw checkpoint state, and does not permute classes.

The actual RFDV checkpoint bytes must still be reverified during the real non-qualifying readiness job.

## 6. G1 seal and package closure

G1 schema v2 binds:

- Stage-03R authority;
- exact execution source;
- exact dependency lock;
- frozen manifest/class map;
- frozen-V1 identity evidence;
- exact official MobileNetV4 artifact bytes/tensor identity/provenance;
- exact S1/S2/S3 pair initializations;
- exact copied DINO teacher bytes;
- canonical teacher state = EMA;
- teacher factory bundle/manifest;
- teacher class-order evidence;
- teacher canonical-state evidence;
- teacher adapter-parity evidence;
- canonical Smoke-B evidence digest;
- canonical dual-GPU-smoke evidence digest;
- seal self-hash.

A schema-v1 G1 seal is invalid.

The deterministic private transport is:

- `G1_PACKAGE.tar`;
- `G1_PACKAGE_MANIFEST.json`.

It enforces:

- stable ordering;
- normalized metadata;
- regular files only;
- no absolute paths;
- no `..`;
- no symlinks;
- member SHA/bytes;
- package SHA/bytes;
- safe extraction;
- exact post-extraction member verification.

## 7. Private G1 publication closure

Before G1 publication the source requires:

- valid `owner/dataset` slug;
- authenticated owner match;
- authenticated account membership/edit evidence;
- authoritative Kaggle metadata proving private target;
- rejection of a public target;
- rejection of a missing target unless:
  `CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET=1`.

Explicit creation defaults private and never uses a public-create flag.

Successful publication requires:

1. version upload;
2. processing-status PASS;
3. fresh download;
4. package SHA verification;
5. safe extraction;
6. member verification;
7. seal verification;
8. S1/S2/S3 pair verification;
9. teacher verification;
10. MNV4 verification.

It emits:

- `G1_PUBLICATION_RECEIPT.json`;
- `G1_TERMINAL_EVIDENCE.json`.

Publication repair validates and reuses an existing sealed local G1. It does not regenerate pair initialization and does not rewrite the seal.

## 8. Downstream G1 mount closure

G2/principal now consume one explicit input:

`CROPCOP_G1_INPUT_ROOT=/kaggle/input/<sealed-g1-dataset>`

The parent:

1. locates exact package + manifest;
2. verifies them;
3. safe-extracts outside Git;
4. verifies every member;
5. binds package → G1 seal → source → dependency;
6. runs the complete shared G1 barrier;
7. only after that performs GPU inventory / child launch.

Children revalidate G1 before calibration or principal execution.

After G1:

- RFDV is not a downstream dependency;
- MNV4 is not reacquired;
- teacher bytes are not reacquired;
- sealed private copies are used.

## 9. Non-qualifying real G1 input readiness

The repository now provides:

`journal_extension/scripts/validate_g1_inputs.py`

This is **not** a canonical qualification phase and has not been executed as part of repository closure.

The real readiness job must attach:

- `CropCop-Model-RFDV`;
- frozen Final-V1 source.

It verifies:

- exact mounted teacher SHA;
- real checkpoint structure;
- real offline factory strict load;
- complete EMA overlay;
- canonical floating-tensor equality to EMA;
- finite state;
- synthetic adapter parity;
- historical class-order source hashes;
- frozen manifest/class-map hashes;
- official MNV4 acquisition/provenance/tensor identity;
- private target preflight;
- deterministic package-transport dry-run using the real verified bytes.

It emits:

`G1_INPUT_READINESS.json`

and explicitly records:

- qualifying phase = false;
- G1 seal created = false;
- pair initializations created = false;
- G1 publication performed = false;
- model training performed = false;
- optimizer steps = 0;
- V1 validation evaluated = false;
- V1 test accessed = false;
- protected external surface accessed = false;
- scientific metric computed = false.

## 10. CPU G1

Actual G1 is designed for clean Kaggle Batch execution with no accelerator.

Pair initialization is CPU-defined under the locked PyTorch version using CPU RNG isolation and the frozen S1/S2/S3 seeds.

G1 terminal evidence records:

`accelerator_required=false`.

G2/principal remain T4×2 phases.

## 11. Final wrapper

Final wrapper SHA:

# `d3b1234bc6a7ce763e804a851d22dd13a86c590c`

This is **not** the execution source.

The wrapper hard-binds:

`AUTHORIZED_SOURCE_SHA = "b89144d8826b6c61c3be7a91cac08681b1b4a99c"`

The canonical notebook remains:

- nbformat 4;
- one thin orchestration code cell plus markdown;
- real multiline Python source;
- exact six-phase operator vocabulary;
- no legacy calibration/principal aliases;
- notebook-global monotonic clock established before clone/install/bootstrap/launch;
- generated from `generate_canonical_notebook.py`, not hand-patched.

The wrapper now exposes:

### CPU G1

- `CROPCOP_RFDV_ROOT`;
- `CROPCOP_FINAL_V1_ROOT`;
- `CROPCOP_G1_PRIVATE_DATASET_SLUG`.

### G2 / principal

- `CROPCOP_G1_INPUT_ROOT`.

It preserves exact Smoke-A/Smoke-B/dual-smoke handoff roots and production secrets/durability requirements.

### Source → wrapper diff

The exact compare from execution source `b89144d8...` to wrapper `d3b1234b...` is eight files only:

- `journal_extension/README.md`;
- `journal_extension/kaggle/README_SMOKE.md`;
- `journal_extension/kaggle/canonical_lane.ipynb`;
- `journal_extension/kaggle/generate_canonical_notebook.py`;
- `journal_extension/kaggle/smoke_inputs.example.json`;
- `journal_extension/scripts/validate_operator_docs.py`;
- `journal_extension/src/cropcop_je/validate.py`;
- `tests/test_je_smoke_sr.py`.

No execution implementation or protected scientific file changed after source freeze.

## 12. Final wrapper exact-head CI

- GitHub Actions run: **#189**
- Run ID: `34010998548`
- exact expected wrapper SHA:
  `d3b1234bc6a7ce763e804a851d22dd13a86c590c`
- exact observed checkout SHA: same
- workflow conclusion: **SUCCESS**

Run #189 verified:

- Python compile PASS;
- repository validator PASS;
- JE static validator PASS;
- science-diff PASS;
- Stage-04A amendment hash PASS;
- canonical notebook code compile PASS;
- **canonical notebook generation consistency PASS**;
- forbidden-artifact scan PASS;
- secret scan PASS;
- active operator-doc validator PASS;
- QA1 Wave A: **18 / 18 PASS**;
- QA1 Wave B: **13 / 13 PASS**;
- QA1 Wave C: **8 / 8 PASS**;
- MGPU suite: **66 / 66 PASS**;
- complete CPU-safe suite: **257 / 257 PASS**;
- exact Transformers compatibility job PASS.

## 13. Science-diff verdict

`01A_G1P_V2_POST_REFACTOR_SCIENCE_DIFF.md`

records:

# **PASS — NO SCIENTIFIC DRIFT**

Protected scientific Git blobs remain byte-identical for:

- `train.py`;
- `data.py`;
- `models.py`;
- CTC-v2;
- all six R04/R05 configs;
- experiment registry.

Frozen scientific semantics remain unchanged.

## 14. Claims explicitly not made

This repository closure does **not** claim:

- real G1 input readiness PASS;
- final-source Smoke A PASS;
- final-source Smoke B PASS;
- final-source dual-GPU smoke PASS;
- G1 PASS;
- G2 PASS;
- R04 launched;
- R05 launched;
- V1 continuity PASS;
- external inference;
- device evaluation;
- any new scientific result.

No V1-test access or protected external-surface access was performed during remediation.

## 15. Repository gate

# **PASS — 01A-G1P-v2 CLOSED; REAL G1 INPUT READINESS MAY BEGIN**

G1 itself is **not** authorized yet.

The next human action is the non-qualifying CPU G1 input-readiness job. It must return:

- `G1_INPUT_READINESS.json`;
- complete console log.

Only an independent PASS audit of that real readiness evidence may authorize final-source Smoke A/B → dual-GPU-smoke qualification.

