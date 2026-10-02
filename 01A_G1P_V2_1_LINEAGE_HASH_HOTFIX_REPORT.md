# CropCop Stage-01A-G1P-v2.1 — Historical Lineage Hash Hotfix Report

- **Artifact:** `01A_G1P_V2_1_LINEAGE_HASH_HOTFIX_REPORT.md`
- **Date:** 2026-09-06
- **Repository:** `rana-m-ahmed/ResearchWork-CropCop`
- **PR:** #8 / `je-stage01a-core-20260905`
- **Final verdict:** **PASS — G1P-v2.1 LINEAGE HASH HOTFIX CLOSED; REAL G1 INPUT READINESS MUST BE RE-RUN**

## 1. Triggering real Kaggle failure

The real non-qualifying CPU G1 readiness run against the prior frozen G1P-v2 source:

`b89144d8826b6c61c3be7a91cac08681b1b4a99c`

reached:

`verify_teacher_class_order.py`

and failed before teacher model loading while validating historical evidence-source bytes.

Observed first mismatch:

- evidence file:
  `historical_v7_teacher_semantics_excerpt.txt`
- stale manifest SHA:
  `56cd4a81a6bc6e07157dc50ddda26d1b0f350b50633796428766280dfa6f2367`
- live checked-in SHA:
  `011c4cbfd6075d410113a283caf241d748dee84b8efdb542dac9cdb6d463ae91`

The same audit also found a stale SHA for:

`historical_final_qa_teacher_loader_excerpt.txt`

- stale manifest SHA:
  `45bb0e4636bd7bae8d5df9bc624e5c5a4cb1f28a10c314a8116fa13a150807d9`
- independently recomputed live checked-in SHA:
  `28d5346bdcdd082a09a12d0342fe6cd72951aeb324582eb7a2e29b4232b3eeaa`

This was an evidence-manifest integrity defect only.

It was **not** a teacher-checkpoint, class-map, Final-V1 mount, Kaggle-path, CRLF/LF, EMA-policy, class-order, or model-construction defect.

## 2. Live evidence-byte recomputation

At pre-hotfix live PR head:

`f80677fa146c005b387e773534374730072cfc89`

the exact checked-in evidence files were recomputed from live Git content.

### v7 excerpt

- path:
  `journal_extension/evidence/historical/teacher_stage1/historical_v7_teacher_semantics_excerpt.txt`
- bytes: **3,677**
- line ending at EOF: LF
- CRLF at EOF: false
- SHA-256:
  `011c4cbfd6075d410113a283caf241d748dee84b8efdb542dac9cdb6d463ae91`
- Git blob:
  `984da2d6440a5f6bdfbb3069a52e2bbee721d3cc`

### final-QA excerpt

- path:
  `journal_extension/evidence/historical/teacher_stage1/historical_final_qa_teacher_loader_excerpt.txt`
- bytes: **5,388**
- line ending at EOF: LF
- CRLF at EOF: false
- SHA-256:
  `28d5346bdcdd082a09a12d0342fe6cd72951aeb324582eb7a2e29b4232b3eeaa`
- Git blob:
  `6d857136469fce77269b5427af4ef30f321ce62d`

The second candidate hash from the diagnostic investigation was therefore independently confirmed from the live bytes before editing the manifest.

## 3. Historical-source fidelity reverification

The original archived sources were re-read from the user's CropCop Library.

### Historical v7 trainer

Archive notebook:

`CropCop_Definitive_Training_QA_Production_v7_1_DDP_MAXIMUM_ROBUST.ipynb`

Notebook SHA-256:

`e116528e7307a29fc2f59a6973e64597bc9b074b35be48bab3959de49dca5bcf`

The notebook's embedded Base85/zlib trainer was materialized exactly as the notebook does.

Recovered file:

`cropcop_ddp_trainer_v7.py`

- bytes: **60,393**
- full SHA-256:
  `a1910fb38f3f7114b8ec607b5ed0408ddaf6abda39b8c29e8819e79b2259b83c`

This exactly matches the historical full hash already stored in the lineage manifest.

The three committed quoted v7 source blocks were compared byte-for-byte against the recovered trainer, excluding only the source-owned evidence annotation header and `HISTORICAL EXCERPT BOUNDARY` separators.

Exact source-block hashes:

1. DINO classifier block:
   `2aa5e5aa104e1dfa75b3ca5981b8c8e8b1c1f86f1ef5b1e3119aff57ea2e62da`
2. model-spec/build-model block:
   `864e7d96c76656840b34a3fc5eb3777f13e7046387c35da4ccaf4e58c7dd2015`
3. EMA block:
   `b6cf1feade13d938fbeda27f9113dfa8292dc1a89069a7ac710170d461b88aa5`

All three blocks are exact.

### Certified final QA engine

Archive:

`CropCop_Final_Model_QA_Evidence_Bundle (1).zip`

Recovered member:

`artifacts/cropcop_final_qa_engine_v1.py`

- bytes: **20,518**
- full SHA-256:
  `9b82b14303a6abb90ac64914b04f44a6d63ee56b12bc4c5a58957356a2ba8786`

This exactly matches the historical full hash already stored in the lineage manifest.

The two committed quoted final-QA source blocks were compared byte-for-byte against the recovered engine, excluding only source-owned evidence annotations.

Exact source-block hashes:

1. manifest/class-map loader block:
   `39bb53128c63cf8f65a1ed76950ff64fe2c4bb4a3103b181bf2cd31f9c3c120f`
2. DINO/checkpoint/EMA loader block:
   `24024ce298ae661426aa0493ddd44fa2578625a020cd3a08eb21c4b6c2714a00`

Both blocks are exact.

## 4. Excerpt files were not changed

Because the committed historical source blocks were faithful, neither evidence excerpt was rewritten.

The frozen v2.1 source still has the same excerpt Git blobs:

- v7 excerpt:
  `984da2d6440a5f6bdfbb3069a52e2bbee721d3cc`
- final-QA excerpt:
  `6d857136469fce77269b5427af4ef30f321ce62d`

Therefore the hotfix corrects stale manifest metadata rather than manufacturing new historical evidence.

## 5. Exact manifest repair

Only two values inside:

`journal_extension/evidence/historical/teacher_stage1/teacher_lineage_manifest.json`

were changed.

### Change 1

From:

`56cd4a81a6bc6e07157dc50ddda26d1b0f350b50633796428766280dfa6f2367`

to:

`011c4cbfd6075d410113a283caf241d748dee84b8efdb542dac9cdb6d463ae91`

### Change 2

From:

`45bb0e4636bd7bae8d5df9bc624e5c5a4cb1f28a10c314a8116fa13a150807d9`

to:

`28d5346bdcdd082a09a12d0342fe6cd72951aeb324582eb7a2e29b4232b3eeaa`

No other lineage semantic field was changed.

In particular, unchanged:

- `teacher_checkpoint_sha256`;
- `teacher_model_id`;
- `experiment_id`;
- `model_key`;
- `resolution`;
- `canonical_state`;
- `manifest_sha256`;
- `class_map_sha256`;
- `semantic_manifest_fingerprint`;
- `historical_index_semantics`;
- `classifier_weight_key`;
- `classifier_shape`;
- `feature_dimension`;
- `not_inferred_from_shape_only`;
- `output_order_transform`;
- `historical_source_full_hashes`.

## 6. Regression and static validation added

The prior CI could validate the lineage JSON schema/semantics without recomputing every historical evidence file's actual bytes.

G1P-v2.1 adds executable regression coverage in:

`tests/test_je_g1p_v2.py`

The regression:

1. loads `teacher_lineage_manifest.json`;
2. iterates every `historical_evidence_sources[*]`;
3. resolves the path beneath the historical evidence root;
4. rejects path escape;
5. requires the source file to exist;
6. computes `sha256_file(path)`;
7. requires equality with the manifest SHA.

A second ordering regression verifies that:

`require_sha256(p, expected, ...)`

remains before:

`load_exact_teacher(...)`

inside `verify_teacher_class_order.py`.

The JE static validator was also hardened with the same independent evidence-source path/existence/SHA check, so exact-head CI now catches this class of defect before Kaggle.

No ignore flag, bypass, or weakening of `require_sha256()` was introduced.

## 7. Previously failing path is closed

The production verifier itself remains fail-closed and unchanged:

`verify_teacher_class_order.py`

still validates every historical evidence source via `require_sha256()` before loading the teacher.

With the corrected manifest, the independent repository hash regression and static validator both pass all historical evidence-source byte checks.

A real RFDV teacher is intentionally not stored in CI; therefore the real readiness job must still be re-run to exercise the full checkpoint/model path under the new source.

## 8. Scope proof

From the prior live PR head `f80677fa...` to the v2.1 implementation source, only three files changed:

- `teacher_lineage_manifest.json`;
- `journal_extension/src/cropcop_je/validate.py`;
- `tests/test_je_g1p_v2.py`.

The manifest diff is exactly two deletions + two additions corresponding to the two stale SHA values.

No historical excerpt changed.

No changes were made to:

- teacher checkpoint identity;
- teacher factory semantics;
- class-order semantics;
- `train.py`;
- `data.py`;
- `models.py`;
- R04/R05 configs;
- seeds;
- dependency versions;
- CTC-v2;
- Stage-03R;
- Stage-04;
- Stage-04A.

No G1, G2, R04, R05, V1-continuity, external-inference, or device-evaluation run was performed.

## 9. Dependency lock

Dependency-lock SHA remains exactly:

`6ea5fb51a0cc39c7940e4aaeb136d0f05214f5c558edd3c47b4cb61c6b516f37`

No package version changed.

## 10. New frozen execution source

The old G1P-v2 source is now historical:

`b89144d8826b6c61c3be7a91cac08681b1b4a99c`

New frozen Stage-01A-G1P-v2.1 execution source:

# `3c71331494b3e031bbbbc3f08d27cd2605c31097`

No execution/scientific implementation was modified after this source was frozen.

### Exact-head source CI

- Actions run: **#193**
- Run ID: `34015594366`
- expected checkout:
  `3c71331494b3e031bbbbc3f08d27cd2605c31097`
- observed checkout: same
- conclusion: **SUCCESS**

Source CI reported:

- repository validator PASS;
- JE static validator PASS;
- science-diff PASS;
- Stage-04A hash PASS;
- active operator-doc validator PASS;
- QA1 Wave A: **18 / 18 PASS**;
- QA1 Wave B: **13 / 13 PASS**;
- QA1 Wave C: **8 / 8 PASS**;
- MGPU tests: **66 / 66 PASS**;
- full CPU-safe suite: **259 / 259 PASS**;
- exact Transformers 5.0.0 compatibility job PASS.

Science verdict:

# **PASS — NO SCIENTIFIC DRIFT**

## 11. Final wrapper SHA

After source freeze, wrapper/source-bound surfaces only were updated.

Final wrapper SHA:

# `fc8c0376e8ba272379022a6e15667479f9e8eb44`

The canonical generator/notebook now bind:

`AUTHORIZED_SOURCE_SHA = "3c71331494b3e031bbbbc3f08d27cd2605c31097"`

Source → wrapper diff is restricted to eight wrapper/docs/validation-test surfaces:

- `journal_extension/README.md`;
- `journal_extension/kaggle/README_SMOKE.md`;
- `journal_extension/kaggle/canonical_lane.ipynb`;
- `journal_extension/kaggle/generate_canonical_notebook.py`;
- `journal_extension/kaggle/smoke_inputs.example.json`;
- `journal_extension/scripts/validate_operator_docs.py`;
- `journal_extension/src/cropcop_je/validate.py` — source-binding literal only;
- `tests/test_je_smoke_sr.py` — wrapper-binding expectation only.

No execution implementation or protected scientific file changed after source freeze.

### Exact-head wrapper CI

- Actions run: **#201**
- Run ID: `34015782557`
- exact wrapper head:
  `fc8c0376e8ba272379022a6e15667479f9e8eb44`
- conclusion: **SUCCESS**

Wrapper CI reported:

- repository validator PASS;
- JE static validator PASS;
- science-diff PASS;
- Stage-04A hash PASS;
- canonical notebook compile PASS;
- canonical notebook/generator consistency PASS;
- forbidden artifact scan PASS;
- secret scan PASS;
- active operator-doc validation PASS;
- QA1 Wave A: **18 / 18 PASS**;
- QA1 Wave B: **13 / 13 PASS**;
- QA1 Wave C: **8 / 8 PASS**;
- MGPU tests: **66 / 66 PASS**;
- full CPU-safe suite: **259 / 259 PASS**;
- exact Transformers compatibility job PASS.

## 12. Readiness status

The real CPU readiness run that discovered this defect is **not** converted into a PASS.

It was source-bound to the superseded source and terminated on a legitimate evidence-integrity gate.

No readiness bypass was added.

The real non-qualifying CPU G1 readiness notebook must be re-run from a clean session against:

`3c71331494b3e031bbbbc3f08d27cd2605c31097`

using the same correct mounts:

- `/kaggle/input/datasets/ranamuhammadahmed6/cropcop-model-rfdv`
- `/kaggle/input/datasets/ranamuhammadahmed6/cropcop-finalized-v8-11-2026-1/CropCop_Final_v1`

The evidence SHA check must remain enabled.

Return for audit:

1. `G1_INPUT_READINESS.json`;
2. complete console log.

Only a new independent PASS audit can authorize subsequent final-source qualification.

## 13. Final repository verdict

# **PASS — G1P-v2.1 LINEAGE HASH HOTFIX CLOSED; REAL G1 INPUT READINESS MUST BE RE-RUN**

