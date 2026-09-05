# CropCop → EAAI Journal Extension

## Stage 01A-P — Pre-G1/G2 Launch Integrity Report

- **Stage:** 01A-P — Pre-G1/G2 Launch Integrity, Global G1 Seal and Kaggle Session Closure
- **Execution date/time:** 2026-09-05T16:36:17+05:00 (Asia/Karachi)
- **Repository:** `rana-m-ahmed/ResearchWork-CropCop`
- **Branch:** `je-stage01a-core-20260905`
- **PR:** #8
- **Current `main` HEAD:** `32190dd86293caa82170df3feea505e3c7443b4b`
- **Prior audited Stage-01A-H head:** `ac39ebd1ba1ad87de6d17ddfa027cd03a5bb7d59`
- **Stage-01A-P implementation commit:** `78089ec8e87f33c05b521b03d25cbaed923058f6`
- **Final documentation/PR head:** not self-embedded; exact final branch SHA is attested by PR #8 and its exact-head CI after publication
- **Scientific authority:** `EAAI-JE-SDL-v2.1-QA`
- **Scientific authority SHA-256:** `aab17b65b0873dcb1ecedb061eb02ff60ccb09f8b830184f5e2231a600278f74`
- **Execution authority:** `EAAI-JE-REA-v2.2-LEAN`
- **Status at report commit:** `IMPLEMENTED_AWAITING_EXACT_HEAD_CI_AND_REAL_KAGGLE_SMOKE`
- **Historical V1 mutation statement:** no historical V1 artifact was modified; V1 test was not accessed.
- **Protected-external statement:** no protected external prediction was accessed or produced.
- **Scientific-result statement:** no G1/G2 empirical measurement and no R04/R05 result was produced.

## 1. Independent-audit findings and fixes

| Severity | Finding | Stage-01A-P closure |
|---|---|---|
| S0 | Each lane could independently establish MobileNetV4 evidence/pair initialization | One global G1 creation path generates S1/S2/S3 once; production lanes only validate/consume the sealed bundle |
| S0 | Arbitrary architecture-compatible pretrained bytes could pass strict loading | G1 compares every candidate tensor against `timm==1.0.26`'s own official pretrained object and seals the candidate byte hash plus pretrained_cfg provenance |
| S0 | Teacher checkpoint hash did not freeze reconstruction semantics | Complete trusted factory/helper source manifest with per-file hashes and one bundle hash; R05 requires exactly that bundle |
| S0 | Teacher class order could be asserted from a supplied class-map hash/shape | G1 requires hashed historical lineage evidence establishing class-map-index semantics, 120-way classifier identity and no factory output reorder; shape alone is rejected |
| S0 | Missing pair artifacts could be regenerated lane-locally after sealing | G1 creation refuses an existing seal or any existing pair material; lane runner contains no pair-generation path |
| S0 | G2 did not prove one coherent G1 identity across accounts | Every calibration summary binds G1 seal, global MNV4 pretrained SHA and dependency lock; teacher calibration also binds teacher/factory; G2 rejects drift |
| S0 | Principal identity did not bind G1/G2 | Run record/checkpoint/resume identity now includes `g1_seal_sha256` and `g2_barrier_sha256` plus dependency/factory bundle hashes |
| S0 | Each child subprocess could receive a fresh 12-hour budget | One notebook-level monotonic start is established before clone/install and propagated to all subprocesses |
| S1 | Calibration could fall through into principal science in one Saved Version | Explicit `smoke/g1/calibration/principal` phases; calibration always exits; principal requires one explicit authorized experiment |
| S1 | Canonical notebook assumed an already prepared repository/environment | Thin clean-session bootstrap now retrieves secrets, clones privately via askpass, checks out exact detached SHA, installs exact lock and validates source |
| S1 | HEAD equality did not prove executed source was clean | Prelaunch and runner reject staged, unstaged and untracked files and require mutable roots outside the repository |
| S1 | Non-core execution dependencies were not frozen | Pre-results dependency lock covers NumPy, Pillow, safetensors, Kaggle client and huggingface-hub in addition to inherited core versions |
| S1 | No real clean-session infrastructure proof existed | Added synthetic/unprotected CUDA smoke: source/stack/secrets → short training → atomic checkpoint → private sync → clean restore → verification/resume → public-safe evidence |
| S1 | Durable run locators could collide | Production Kaggle locator template must contain the run ID and resolve uniquely across calibration/principal identities |
| S1 | Filesystem durable store had a target/previous rename crash window | Previous generation is retained and recognized as a valid restore candidate if active target is absent/invalid |
| S1 | Existing evidence branches could not reliably be fetched in a clean private clone | Evidence fetch and push both use ephemeral askpass credentials; token is not placed in URLs/configuration |
| S2 | `validate_prelaunch.py` was not part of the production path | Canonical G1 validation invokes static + clean-source prelaunch before the full mounted-byte G1 barrier |

No scientific recipe field was changed by these fixes.

## 2. Global G1 seal design

The G1 sealing process is one canonical pre-results operation.

`capture_mnv4_pretrained_provenance.py` verifies that the supplied MobileNetV4 candidate represents
the official `mobilenetv4_conv_medium.e500_r256_in1k` object exposed by exact `timm==1.0.26`.
It constructs the official pretrained model through timm's locked pretrained configuration and requires
the candidate state dictionary to have identical keys, shapes, dtypes and tensor values. G1 then freezes
the candidate file's SHA-256/byte count together with the official pretrained_cfg provenance.

`seal_g1.py` requires a clean authorized source commit, the exact dependency environment, the frozen V1
manifest/class map, exact teacher bytes, a validated teacher-factory bundle, validated historical class
ordering, and a prior real-Kaggle infrastructure-smoke attestation. It creates S1/S2/S3 pair artifacts in
one operation and writes one canonical `G1_MODEL_IDENTITY_SEAL.json` with a canonical self-hash.

The sealed private bundle is versioned through a pre-created private Kaggle dataset. It is not eligible
for public Git.

## 3. Pair-init persistence

The three pair artifacts are generated only by the G1 phase:

- S1 seed 21270083 → R04-S1 + R05-S1
- S2 seed 606135704 → R04-S2 + R05-S2
- S3 seed 1153870846 → R04-S3 + R05-S3

Each pair evidence record freezes its binary SHA/bytes, source pretrained SHA, pair ID, seed and exact
authorized consumers. Later clean sessions restore/mount those exact private bytes. The production lane
runner cannot invoke `prepare_pair_init.py` or `save_pair_initialization`.

## 4. Teacher provenance and class-order evidence

Teacher evidence is deliberately split into three independent identities:

1. exact historical checkpoint bytes;
2. complete trusted factory/helper source bundle;
3. historical class-order evidence.

The public repository proves the historical checkpoint SHA, but it does **not** contain enough
historical material to independently establish teacher-logit class ordering. Therefore Stage 01A-P
does not fabricate a PASS record. At real G1, `verify_teacher_class_order.py` requires a restricted
historical lineage manifest plus actual hashed historical evidence files. It checks that the historical
lineage binds the frozen class map, identifies the exact classifier weight key, explicitly establishes
`class_map_index` semantics, and does not rely on output shape alone. It then verifies the loaded
classifier/output width is 120 and that the sealed factory performs no output reorder.

If that historical evidence cannot be supplied, G1 remains blocked.

## 5. G1 → G2 → principal chain

Calibration summaries now carry:

- exact source Git SHA;
- software-stack hash;
- G1 seal hash;
- execution-dependency-lock hash;
- global MobileNetV4 pretrained SHA;
- teacher checkpoint/factory bundle identity for teacher calibration;
- scheduling telemetry and durable-sync timing.

The G2 barrier requires all three calibration summaries and rejects any source, software, G1,
dependency or pretrained drift. Teacher calibration must bind the frozen teacher and one factory bundle.

Principal runs require both a valid G1 seal and terminal G2 barrier. Both hashes are part of the run,
checkpoint and resume identity. Resume with a changed G1 or G2 identity fails closed.

## 6. Notebook-global Kaggle wall clock

The canonical notebook sets `CROPCOP_NOTEBOOK_STARTED_MONOTONIC` before clone, package installation,
bootstrap, calibration or training. Every child constructs its budget from that same monotonic start.

Before principal execution, the runner compares the real remaining safe notebook time with the
G2-derived scheduling forecast plus checkpoint, durable-sync and additional reserve. It never grants R05
a fresh 12-hour budget after R04 or any earlier phase.

Calibration and principal Saved Versions are separate by default. Principal execution runs one explicit
state and then exits. If the full state does not fit but a useful safe segment does, the existing
checkpoint/Layer-B path handles a planned rollover. If even a useful segment plus finalization cannot fit,
the new Saved Version refuses to start the expensive phase.

## 7. Clean-session bootstrap

The canonical notebook remains thin. It:

- establishes the global clock;
- disables Python bytecode writes so repository imports cannot dirty the source tree;
- requires exact Python 3.12.13;
- retrieves Git/Kaggle credentials from Kaggle Secrets without printing values;
- clones the private repository with an ephemeral askpass helper;
- checks out the exact authorized commit detached;
- installs the exact dependency lock;
- runs source/dependency bootstrap validation;
- delegates to one explicit smoke/G1/calibration/principal repository phase.

The bootstrap rejects a dirty tree and any mutable output/evidence root inside the source repository.

## 8. Execution dependency lock

The pre-results execution lock is:

| Dependency | Version |
|---|---:|
| Python | 3.12.13 |
| torch | 2.12.1 |
| torchvision | 0.27.1 |
| timm | 1.0.26 |
| NumPy | 2.5.2 |
| Pillow | 12.3.0 |
| safetensors | 0.8.0 |
| kaggle | 2.2.4 |
| huggingface-hub | 1.30.0 |

The canonical dependency-lock SHA-256 is:

`767859a3cb1b37c0e50f753aa8c7a5c400424417cba3eaa59f0ab617ff18f701`

The non-core choices were frozen before JE empirical execution for compatibility/reproducibility, not
selected after seeing model performance. Runtime execution refuses substitutions. If exact core versions
cannot execute on the real Kaggle platform, execution stops for a technical re-lock decision.

## 9. Infrastructure smoke

The smoke test is intentionally non-scientific and accepts only synthetic/unprotected input. On the
actual Kaggle CUDA environment it must prove:

clean exact source → locked environment/secret retrieval → synthetic training → atomic checkpoint →
private durable sync → deletion of local recovery state → private restore → checkpoint verification →
resume → public-safe evidence publication.

Its machine-readable evidence is source-bound and hashed into G1. Therefore a smoke from another source
commit cannot authorize G1.

No real smoke execution is claimed in this report.

## 10. Durable topology

The production recovery backend is a private Kaggle dataset with a unique locator per run ID. The
template validator rejects a shared locator that cannot resolve uniquely. Calibration's actual durable
sync supplies observed write/sync evidence before principal science.

The filesystem backend remains available for controlled environments; it now fsyncs its transaction
boundaries and can restore the validated previous durable generation if an interrupted target/staging
rename leaves the active generation absent.

Terminal bundles retain `latest`, `previous` and `selected` checkpoint references as applicable;
later run IDs cannot overwrite another run's durable namespace.

## 11. New verification surface

A new `tests/test_je_pre_g1_integrity.py` adds targeted coverage for:

- dependency lock self-hash and non-core drift;
- global pretrained consistency across pair inits;
- factory-bundle drift and class-order evidence requirements;
- refusal to regenerate pair artifacts into an existing G1 target;
- exact clean source / dirty source / wrong HEAD / output-root placement;
- cross-lane G1/pretrained/dependency drift at G2;
- principal launch without G1/G2 and incoherent G1/G2;
- resume under changed G1/G2;
- notebook-global clock sharing and no fresh second-principal budget;
- durable locator collisions;
- filesystem previous-generation recovery;
- selected-checkpoint retention;
- sealed-pair consumption only;
- canonical phase/bootstrap ordering;
- durable sync before planned rollover exit;
- one-principal-per-Saved-Version default;
- mandatory pre-G1 real-smoke attestation.

The existing 54 tests are retained. The exact final discovered count is determined by GitHub Actions on
the final pushed head and must not be guessed in this report.

## 12. Repository publication and exact-head CI

The source implementation is frozen at:

`78089ec8e87f33c05b521b03d25cbaed923058f6`

This report is added afterward, so the final PR head necessarily differs from the implementation commit.
Embedding the final report commit's own SHA or its future Actions run ID into itself would be
self-referential. As in Stage 01A-H, the exact final branch SHA, Actions run number/ID, checkout assertion,
test count and conclusion are recorded in PR #8 metadata after CI completes on that exact head.

Old Actions run #27 is historical Stage-01A-H evidence only and is not accepted for this continuation.

## 13. Remaining real inputs / external closure work

Repository implementation cannot manufacture the following:

- real Kaggle CUDA execution under the exact dependency lock;
- the synthetic/unprotected smoke bundle and private smoke durable dataset;
- exact frozen V1 manifest/class-map bytes and V1 train/validation image root;
- exact MobileNetV4 candidate pretrained bytes to verify against official timm;
- exact historical teacher checkpoint bytes;
- trusted teacher factory/helper source bundle;
- historical training/class-map/model-construction evidence sufficient to prove teacher output ordering;
- private G1 Kaggle dataset destination;
- exact ConvNeXt-Tiny pretrained bytes required later for CAL-CNXTT;
- per-run private durable dataset destinations/credentials.

These are not replaced with invented hashes, timings, notebook IDs or PASS records.

## 14. Gate

Repository engineering alone does not authorize real G1.

The final Stage-01A-P gate additionally requires a fresh exact-head GitHub Actions PASS and a real
clean-session Kaggle infrastructure-smoke PASS bound to that same source. Because this report is written
before those external facts exist, its committed status is:

**BLOCKED — PRE-G1/G2 LAUNCH INTEGRITY DEFECTS REMAIN**

The blocker is external execution/evidence closure, not a scientific redesign request. No G1/G2 or
R04/R05 launch is authorized by this report.
