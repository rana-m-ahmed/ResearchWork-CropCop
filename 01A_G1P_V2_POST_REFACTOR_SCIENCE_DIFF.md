# Stage 01A-G1P-v2 — Post-Refactor Science-Diff and Full QA

- **Artifact:** `01A_G1P_V2_POST_REFACTOR_SCIENCE_DIFF.md`
- **Date:** 2026-09-06
- **Implementation candidate verified:** `e6c2977d03dc75a1b1f5c33f83e29973a8d60c06`
- **Exact-head GitHub Actions:** #179
- **Run ID:** `34010661763`
- **Conclusion:** SUCCESS
- **Verdict:** **PASS — NO SCIENTIFIC DRIFT**

## 1. Authority preservation

Scientific authority remains:

- ID: `EAAI-JE-SDL-v2.1-QA`
- SHA-256: `aab17b65b0873dcb1ecedb061eb02ff60ccb09f8b830184f5e2231a600278f74`

Stage-04 base execution architecture remains:

- ID: `EAAI-JE-REA-v2.2-LEAN`
- SHA-256: `a1bc1f8bb2d7a133a46c3f59f83dd2ba8ca3bf0fa8d77ae323553276931bc079`

Stage-04A additive execution amendment remains:

- ID: `EAAI-JE-MGPU-A1`
- SHA-256: `3f08f2dbe7e83143c7e6f1fcf6732c7d88086295a600a74b323b89b46caed3f6`

No scientific authority was replaced or re-locked by G1P-v2.

## 2. Protected scientific Git blobs

The current exact implementation candidate was independently compared against the original Stage-01A-MGPU science sentinel.

| Protected object | Frozen Git blob | Current Git blob | Result |
|---|---|---|---|
| `journal_extension/src/cropcop_je/train.py` | `a259f7c825942e7f3031b20e29f33fc55f1d9d7f` | same | BYTE-IDENTICAL |
| `journal_extension/src/cropcop_je/data.py` | `9871ece382e0345bcab2e832d3116611855c3de7` | same | BYTE-IDENTICAL |
| `journal_extension/src/cropcop_je/models.py` | `c7db1155de1621e751c4bdbad7e9c4096ab06746` | same | BYTE-IDENTICAL |
| CTC-v2 | `517d6fcc38e3a8e480fb1edcd97bac87907c5b5c` | same | BYTE-IDENTICAL |
| R04 S1 | `4f6e985c08210f425b0c651c8bc10e61ad3db5f8` | same | BYTE-IDENTICAL |
| R04 S2 | `9d6f85ac6656588397ef96e76e694cf09e2203d1` | same | BYTE-IDENTICAL |
| R04 S3 | `d08f78619c22ae47115865bed2da52ac67c74625` | same | BYTE-IDENTICAL |
| R05 S1 | `2607f803d7d0ee9ded564dd52f9d91d4d0ef35a3` | same | BYTE-IDENTICAL |
| R05 S2 | `79cb7d5ecbd4674a7ce70718dae07df5de2a1b9b` | same | BYTE-IDENTICAL |
| R05 S3 | `6df147239c652d9c64368208e240b63c9ad5f61f` | same | BYTE-IDENTICAL |
| experiment registry | `ac83f61eba78a24e0a40e387a817a27bcdef2bed` | same | BYTE-IDENTICAL |

Therefore the principal model computation, configs, seeds, objectives and registry have zero byte changes.

## 3. Scientific semantics preserved

The executable sentinel remains green for:

- 30 epochs;
- AdamW scientific parameters;
- frozen LR/schedule semantics;
- micro-batch 16;
- accumulation 4;
- effective batch 64;
- direct objective;
- teacher objective;
- validation-only checkpoint selection;
- S1/S2/S3 seeds;
- same-seed direct/teacher pairing;
- exact MobileNetV4 student identity;
- exact historical DINOv3 teacher checkpoint identity;
- 120-way class semantics;
- V1-test firewall;
- external protected-surface firewall.

No DDP/DataParallel/FSDP or execution-envelope identity has been introduced into principal scientific checkpoint identity.

## 4. Execution dependency amendment

The only dependency change is the pre-results compatibility addition:

`transformers==5.0.0`

The new deterministic dependency-lock SHA is:

`6ea5fb51a0cc39c7940e4aaeb136d0f05214f5c558edd3c47b4cb61c6b516f37`

Selection was based only on:

- archived historical environment evidence;
- historical v7 allowed range;
- official versioned DINOv3 ConvNeXt API;
- exact package resolver compatibility;
- offline direct construction;
- synthetic strict-load geometry.

No model-quality result informed the version choice.

## 5. Historical teacher closure

G1P-v2 now contains a public-safe, source-owned historical teacher factory that:

- reconstructs DINOv3 ConvNeXt-Tiny offline;
- strict-loads raw `checkpoint["model"]`;
- requires exact complete EMA shadow coverage over all floating state entries;
- copies EMA exactly;
- verifies finite canonical state;
- preserves `head.weight`;
- verifies classifier shape `[120,768]`;
- exposes historical feature/head adapter semantics;
- forbids pretrained-network model acquisition;
- forbids raw-state fallback;
- forbids class permutation.

Historical source evidence is hash-bound in the repository lineage package.

## 6. G1 sealing and chronology hardening

The upgraded G1 v2 seal now binds:

- scientific authority;
- exact execution source;
- dependency lock;
- manifest/class-map identity;
- frozen-V1 identity evidence;
- exact official MobileNetV4 artifact/tensor/provenance;
- S1/S2/S3 pair initializations;
- exact copied DINO teacher;
- canonical teacher state = EMA;
- teacher factory bundle/manifest;
- class-order evidence;
- canonical-state evidence;
- adapter-parity evidence;
- terminal Smoke-B digest;
- terminal dual-GPU-smoke digest;
- seal self-hash.

A schema-v1 G1 seal is no longer accepted.

## 7. G1→G2 interface and parent fail-early repair

The previously reproduced defect where:

`run_lane.validate_g1()`

omitted:

`--dual-gpu-smoke-evidence`

has been eliminated through a shared G1 barrier contract.

Creation-time, lane-side and parent-side validation now consume the same barrier input contract.

For G2/principal the parent now:

1. requires one explicit `CROPCOP_G1_INPUT_ROOT`;
2. locates exact `G1_PACKAGE.tar` + `G1_PACKAGE_MANIFEST.json`;
3. verifies package hash/bytes/manifest;
4. safe-extracts outside Git;
5. verifies every extracted member hash/bytes;
6. binds package→seal→source→dependency;
7. runs the complete G1 barrier;
8. only then proceeds to GPU inventory or child launch.

Children revalidate G1 again before calibration or principal execution.

RFDV and external MNV4 acquisition are not required downstream after G1.

## 8. Private package / publication durability

G1P-v2 now provides:

- deterministic uncompressed tar transport;
- stable member ordering;
- normalized tar metadata;
- no absolute paths;
- no `..`;
- no symlinks;
- member SHA/bytes;
- package SHA/bytes;
- safe extraction;
- post-extract member verification;
- authenticated private-target preflight;
- owner match;
- authenticated account dataset membership;
- authoritative private metadata check;
- explicit creation only behind `CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET=1`;
- no public-create mode;
- Kaggle processing-status polling;
- fresh published-version download;
- package/member/seal/pair/teacher/MNV4 round-trip verification;
- `G1_PUBLICATION_RECEIPT.json`;
- `G1_TERMINAL_EVIDENCE.json`;
- publication-only repair that never regenerates pair initialization and never rewrites a valid seal.

## 9. Non-qualifying real-input readiness

A dedicated CPU-safe entrypoint now exists:

`journal_extension/scripts/validate_g1_inputs.py`

It verifies the real mounted:

- RFDV teacher exact path/SHA;
- real checkpoint structure;
- real offline strict factory load;
- complete EMA overlay;
- canonical tensor equality;
- synthetic adapter parity;
- historical class-order source hashes;
- frozen V1 manifest/class-map identities;
- official MNV4 acquisition/tensor identity;
- private target preflight;
- deterministic transport construction/extraction using the real verified teacher/MNV4 bytes.

It emits:

`G1_INPUT_READINESS.json`

with explicit false flags for:

- qualifying phase;
- G1 seal creation;
- pair initialization creation;
- G1 publication;
- model training;
- optimizer steps;
- V1 validation evaluation;
- V1 test access;
- protected external access;
- scientific metric computation.

## 10. CPU G1 support

Actual G1 remains a clean Kaggle Batch phase but does not require CUDA.

Pair initialization is CPU-defined under the exact locked PyTorch version using:

`torch.random.fork_rng(devices=[])`

and the frozen seed.

Terminal G1 evidence records:

`accelerator_required=false`

G2/principal remain T4×2 execution-envelope phases.

## 11. Full test matrix

Actions #179 exact head:

`e6c2977d03dc75a1b1f5c33f83e29973a8d60c06`

reported:

- Python compile PASS;
- repository validator PASS;
- JE static validator PASS;
- science-diff PASS;
- Stage-04A hash PASS;
- canonical notebook compile PASS;
- notebook generation consistency PASS;
- forbidden model/checkpoint scan PASS;
- secret scan PASS;
- active operator-doc validator PASS;
- QA1 Wave A: PASS;
- QA1 Wave B: PASS;
- QA1 Wave C: PASS;
- MGPU suite: **66 / 66 PASS**;
- complete CPU-safe suite: **257 / 257 PASS**;
- exact Python 3.12.13 + Transformers 5.0.0 compatibility job: PASS.

The G1P-v2 tests explicitly cover:

- pre-fix G1→G2 regression;
- shared barrier contract;
- schema-v2 dual-smoke binding;
- deterministic package equality;
- traversal rejection;
- corruption detection;
- package mount verification;
- parent validation before GPU inventory;
- child revalidation before calibration/principal;
- downstream sealed teacher/MNV4 use;
- RFDV absence downstream;
- readiness non-qualification;
- no-network teacher construction;
- publication repair immutability;
- private target owner/public/missing/status failures;
- CPU-defined G1 initialization.

## 12. No protected scientific execution

During G1P-v2 repository remediation:

- G1 was not run;
- G2 was not run;
- R04 was not run;
- R05 was not run;
- V1 continuity was not run;
- external inference was not run;
- device evaluation was not run;
- V1 test was not opened;
- no new scientific metric was created.

The previous real Smoke A/B and real dual-GPU smoke remain valid historical evidence only for source:

`fe88e426b4698977d65efe9702f1d48cf5ff96a3`

and dependency lock:

`767859a3cb1b37c0e50f753aa8c7a5c400424417cba3eaa59f0ab617ff18f701`

They do not qualify the new source/dependency pair.

## 13. Wave-22 verdict

# **PASS — NO SCIENTIFIC DRIFT**

The next operation is the source/wrapper two-SHA freeze. No execution implementation change is authorized after the source candidate is frozen unless a new exact blocker invalidates it.
