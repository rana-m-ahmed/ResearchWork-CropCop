# Stage 01A-G1P-v2.1 — Real CPU G1 Input Readiness Audit

- **Artifact:** `01A_G1P_V2_1_REAL_INPUT_READINESS_AUDIT.md`
- **Audit date:** 2026-09-06
- **Execution source under audit:** `3c71331494b3e031bbbbc3f08d27cd2605c31097`
- **Dependency lock:** `6ea5fb51a0cc39c7940e4aaeb136d0f05214f5c558edd3c47b4cb61c6b516f37`
- **Executed notebook SHA-256:** `3a74659e587e2a83dd19c78bf54c431bd303ba06eb239e4c7881b0fb64314abf`
- **Returned G1_INPUT_READINESS.json SHA-256:** `087590383d6ef2ab47e462f0ffa34df7483350fd65450d0fea9adfa8914f30c1`
- **Notebook-reported status:** `PASS_WITH_PRIVATE_TARGET_DEFERRED`

## 1. Evidence consistency

The returned `G1_INPUT_READINESS.json` is byte-for-byte identical to the JSON object emitted by the executed notebook.

No Python notebook output has `output_type=error`.

The notebook checked out the exact immutable source:

`3c71331494b3e031bbbbc3f08d27cd2605c31097`

and reported exact-source checkout PASS before readiness checks.

The locked dependency identity remained:

`6ea5fb51a0cc39c7940e4aaeb136d0f05214f5c558edd3c47b4cb61c6b516f37`

The pip resolver warning about unrelated preinstalled Kaggle packages was non-terminal; the install command completed successfully and the repository dependency validator subsequently passed.

## 2. Real input resolution

The readiness notebook resolved only the explicit allowed Kaggle roots:

- RFDV:
  `/kaggle/input/datasets/ranamuhammadahmed6/cropcop-model-rfdv`
- frozen Final-V1:
  `/kaggle/input/datasets/ranamuhammadahmed6/cropcop-finalized-v8-11-2026-1/CropCop_Final_v1`

The source-owned resolver bound:

- manifest SHA:
  `bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2`
- class-map SHA:
  `46f7811726c19c42bd7213b2d8178b19a5a182a1b763f60a94ee2c0e5f6688d2`
- teacher checkpoint SHA:
  `74b4701b8931976c9227845ead50788ae47e3596f575f2817b7352a715f53b79`
- teacher factory:
  `historical_dino_tiny:build_teacher`

## 3. Historical lineage hotfix proof

The exact code path that previously failed, `verify_teacher_class_order.py`, now passed against the real mounted teacher and source-owned lineage evidence.

It verified:

- `historical_v7_teacher_semantics_excerpt.txt`
  SHA-256:
  `011c4cbfd6075d410113a283caf241d748dee84b8efdb542dac9cdb6d463ae91`
- `historical_final_qa_teacher_loader_excerpt.txt`
  SHA-256:
  `28d5346bdcdd082a09a12d0342fe6cd72951aeb324582eb7a2e29b4232b3eeaa`

Class-order record status:

`PASS`

The verified historical semantics remained:

- `historical_index_semantics = class_map_index`
- `classifier_weight_key = head.weight`
- classifier shape = `[120,768]`
- output transform = `none`
- proof not inferred from shape alone = true.

This closes the real failure that motivated the v2.1 hotfix.

## 4. Real teacher / EMA readiness

The actual RFDV teacher was loaded successfully.

Observed:

- checkpoint bytes: `446918619`
- checkpoint SHA:
  `74b4701b8931976c9227845ead50788ae47e3596f575f2817b7352a715f53b79`
- strict real checkpoint factory load: PASS
- canonical state: EMA
- complete EMA coverage: PASS
- floating state keys: 182
- EMA shadow keys: 182
- canonical floating tensors equal EMA: true
- finite state: true
- classifier key: `head.weight`
- classifier shape: `[120,768]`
- feature dimension: 768
- teacher output width: 120
- protected data accessed by canonical-state check: false
- scientific metric computed: false.

## 5. Adapter parity

Synthetic direct-vs-adapter parity passed.

- direct-vs-adapter parity: true
- max absolute difference: `0.0`
- mean absolute difference: `0.0`
- feature shape: `[1,768]`
- logits shape: `[1,120]`
- protected data accessed: false
- scientific metric computed: false.

## 6. Official MobileNetV4 readiness

The official source-owned MobileNetV4 preparation path passed.

- model:
  `mobilenetv4_conv_medium.e500_r256_in1k`
- timm:
  `1.0.26`
- serialized candidate SHA:
  `35ca23dc46c0075d9acdcab06d30e5395c4d97e422d8cb5e5e627823aeb7c1fa`
- canonical tensor identity SHA:
  `abfaaee39cc76681f4b0df9a96c5784a37b5630a7c69b3536e055381ff486c9f`
- exact official timm tensor match: true.

The unauthenticated Hugging Face Hub warning is not an integrity failure; the official artifact acquisition and exact tensor-provenance verification completed successfully.

## 7. Deterministic transport dry-run

The readiness transport dry-run passed.

- status: PASS
- transport: deterministic uncompressed tar
- members: 9
- package bytes: `486113280`
- package SHA:
  `4c65157d9008d4d8ae5229d926a44094819e9cd13478b56a8597687ae27210f1`
- safe extraction verified: true
- pair initializations created: false
- production G1 seal created: false.

The dry-run included the real verified teacher and MNV4 bytes.

## 8. Non-qualification / scientific firewall

The returned readiness evidence records:

- qualifying phase: false
- G1 seal created: false
- pair initializations created: false
- G1 publication performed: false
- private dataset version created: false
- model training performed: false
- optimizer steps performed: 0
- V1 validation evaluated: false
- V1 test accessed: false
- external protected surface accessed: false
- scientific metric computed: false.

Therefore this is readiness evidence only, not a scientific or qualification run.

## 9. Private target status

Private target status is deliberately:

`DEFERRED`

for future slug:

`ranamuhammadahmed6/cropcop-g1-sealed`

The readiness notebook did not create or publish a private dataset version.

This does **not** block Smoke A/B or dual-GPU smoke because those qualification phases do not require the future G1 package target.

It does mean **G1 itself is not yet authorized solely by this readiness audit**. Before/during actual G1, the frozen source must execute its authenticated private-target creation/preflight path, using explicit authorization such as:

`CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET=1`

when creation is intended, and it must prove the target is private before publication.

## 10. Independent verdict

# **PASS — REAL G1 ARTIFACT/MODEL INPUT READINESS COMPLETE; FINAL-SOURCE SMOKE MAY BEGIN**

This verdict authorizes the final-source technical qualification sequence only.

It does not authorize a G1 PASS claim and does not claim private G1 publication readiness.

Required chronology now becomes:

```text
smoke-write
→ fresh smoke-restore
→ independent audit
→ dual-gpu-smoke
→ independent audit
→ private-target preflight/create at G1
→ g1
```

The next human action is a clean Kaggle Batch `smoke-write` run bound to execution source:

`3c71331494b3e031bbbbc3f08d27cd2605c31097`

with only `CROPCOP_GITHUB_TOKEN` required.
