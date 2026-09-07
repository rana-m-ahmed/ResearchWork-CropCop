# CropCop EAAI Journal Extension — Lean JE Core

This namespace implements the minimum Stage-03R-bound machinery required before R04/R05 training.
It is intentionally fail-closed: restricted dataset/model bytes are supplied at execution time and
are verified before use; V1 test and sealed external surfaces are not resolvable by the training API.

## Authority

- Scientific lock: `EAAI-JE-SDL-v2.1-QA`
- Lock SHA-256: `aab17b65b0873dcb1ecedb061eb02ff60ccb09f8b830184f5e2231a600278f74`
- Repository baseline at Stage 00: `32190dd86293caa82170df3feea505e3c7443b4b`


<!-- QA1_OPERATOR_BEGIN -->
## Current Stage-01A-G1P-v2.2 operator path

**Frozen execution source:** `beabe97d046e071edacdfa1c6933eeb4edb3a588`  
**Dependency lock:** `6ea5fb51a0cc39c7940e4aaeb136d0f05214f5c558edd3c47b4cb61c6b516f37`

Repository closure is followed first by the separate, non-qualifying CPU `G1_INPUT_READINESS.json` job. After independent readiness PASS, final-source qualification proceeds:

```text
smoke-write
→ fresh smoke-restore
→ independent audit
→ dual-gpu-smoke
→ independent audit
→ g1
→ independent terminal G1 audit
→ calibration-dual
→ principal-dual
```

Canonical operator phases remain exactly:

- `smoke-write`
- `smoke-restore`
- `dual-gpu-smoke`
- `g1`
- `calibration-dual`
- `principal-dual`

Smoke A/B require only `CROPCOP_GITHUB_TOKEN`. The dual smoke is T4×2 and synthetic.

For calibration/principal, the certified Final-V1 schema is frozen to `record_key`, `portable_relpath`, `split`, and `label`; numeric targets are derived only from the locked `class_to_idx.json`. Production Kaggle durable recovery targets must be private and owner-bound before launch.

CPU G1 consumes exact `CROPCOP_RFDV_ROOT`, `CROPCOP_FINAL_V1_ROOT`, `CROPCOP_G1_PRIVATE_DATASET_SLUG`, terminal Smoke-B, and terminal dual-smoke evidence. The exact teacher path, teacher factory/lineage and official MNV4 preparation are source-controlled/resolved by the frozen implementation.

The superseded v2.1 source and its source-bound readiness/Smoke/dual-smoke/G1 evidence are historical only. Fresh v2.2 qualification starts again from readiness. Calibration-dual is blocked until independent audit of fresh terminal v2.2 G1 evidence.

If `ranamuhammadahmed6/cropcop-g1-sealed` is authoritatively present/private/owned at fresh G1 time, reuse it with `CROPCOP_G1_ALLOW_CREATE_PRIVATE_DATASET=0`; do not recreate it unnecessarily.

G2/principal use one sealed private input root, `CROPCOP_G1_INPUT_ROOT`, rather than reattaching RFDV or reacquiring MNV4/teacher bytes. The parent verifies and safe-extracts the deterministic G1 package and runs the complete G1 barrier before GPU child launch.

G1 / calibration-dual / principal-dual preserve Kaggle credentials and production durability requirements. Interactive editor runs are diagnostic only.

See `journal_extension/kaggle/README_SMOKE.md` for the active human handoff. Older Stage 01A-H/P/SR/MGPU sections below are historical implementation chronology.
<!-- QA1_OPERATOR_END -->

## Restricted inputs

Scientific jobs require explicit paths to restricted bytes. Nothing here downloads or guesses them.

Required for R04:
- full frozen V1 manifest whose SHA-256 is `bdb82211...f68e2`;
- frozen 120-class map whose SHA-256 is `46f78117...688d2`;
- V1 image root;
- exact MobileNetV4 ImageNet pretrained object from `timm==1.0.26`, hashed at G1;
- a per-seed paired student initialization produced once and reused by matching R04/R05.

R05 additionally requires:
- historical DINOv3 ConvNeXt-Tiny checkpoint with SHA-256 `74b4701b...3b79`;
- a trusted restricted-environment teacher factory/adaptor. The public JE core deliberately does not
  guess a checkpoint constructor.

## Manifest adapter

The full restricted manifest schema is not reconstructed in public Git. Execution therefore passes
the exact column names explicitly (`stable_row_id`, path, split, class index). The loader validates
the manifest hash before reading rows and authorizes only `DS-V1-TRAIN` and `DS-V1-VAL`.

## Principal flow

1. `python -m cropcop_je.validate --repo-root .`
2. Verify exact pretrained/teacher bytes with `scripts/verify_artifact.py`.
3. Create `PAIR_INIT_S1/S2/S3` with `scripts/prepare_pair_init.py`.
4. Run direct/teacher calibration with `scripts/calibrate.py`.
5. Only after G0/G1/G2 pass, launch the six principal configs.

Calibration outputs are scheduling evidence only and must never become scientific checkpoints.


## Stage 01A-H execution hardening

The original Stage-01A G0 core is preserved, but its execution machinery is now hardened for
multi-session Kaggle use before any real G1/G2 run begins.

The hardened path adds:

- one canonical thin notebook: `journal_extension/kaggle/canonical_lane.ipynb`;
- three small lane specifications for K1/K2/K3;
- runtime-only principal run-ID resolution rather than predeclared fake launched IDs;
- atomic, content-addressed checkpoint objects;
- `latest` / `previous` / `selected` recovery generations;
- SHA-256 + load validation before checkpoint publication;
- persisted best-validation state and complete validation history across resume;
- exact mid-epoch cursor and deterministic resumed data order;
- explicit epoch propagation in sampler keys so persistent workers cannot use stale augmentation epochs;
- teacher projection probing in eval/no-grad mode with RNG isolation and original train/eval modes restored;
- checkpoint scientific identity bound to source Git SHA, lane, software-stack identity, teacher-factory source,
  model/data/config hashes, seed, and paired initialization;
- append-only execution-segment ledgers;
- bounded session rollover with a finalization margin;
- Layer-A local recovery plus pluggable Layer-B durable persistence;
- scheduling-only calibration with real dataloader/checkpoint/validation telemetry;
- a persisted three-calibration G2 barrier;
- cross-lane calibration-summary recovery through public-safe evidence branches when configured;
- audited `run-evidence/<run_id>` Git publication of small text evidence only;
- compile, repository-contract, JE-static, full CPU-safe test, and exact-head CI context artifacts.

These mechanisms do not change R04/R05 science, CTC-v2, seeds, objectives, selection rules, protected
surfaces, or the historical teacher identity.

### Important execution boundary

No real G1/G2 calibration or R04/R05 scientific run is represented by this repository state.
Restricted bytes and a qualified Kaggle GPU environment must be supplied and verified in the next
execution stage. Calibration IDs and lane definitions in the repository are orchestration definitions,
not evidence that a Kaggle job has run.


## Stage 01A-P — pre-G1/G2 launch integrity

Stage 01A-P closes orchestration and identity gaps discovered after the 01A-H hardening gate. It does
not change R04/R05 science.

### Canonical Saved-Version phases

The one canonical notebook now requires an explicit phase:

1. `smoke` — synthetic/unprotected infrastructure only;
2. `g1` — official-pretrained verification, teacher-lineage verification, one-time pair creation and
   global G1 sealing;
3. `calibration` — one authorized non-scientific G2 calibration for the selected lane;
4. `principal` — exactly one explicitly selected R04/R05 state, only after terminal G1 + G2.

A calibration Saved Version always exits after calibration/G2 integration. It never falls through into
principal science. A principal Saved Version executes only `CROPCOP_PRINCIPAL_EXPERIMENT` and exits
after that state becomes terminal or reaches a planned notebook-global rollover boundary.

### Global G1

Real G1 produces one private `G1_MODEL_IDENTITY_SEAL.json`. The seal binds:

- EAAI scientific authority and exact source commit;
- frozen V1 manifest and class-map identities;
- `mobilenetv4_conv_medium.e500_r256_in1k` under exact `timm==1.0.26`;
- tensor-exact verification against timm's own official pretrained object and its pretrained_cfg;
- all three S1/S2/S3 paired student initialization hashes/bytes/consumers;
- exact historical teacher bytes;
- a complete trusted teacher-factory source-bundle hash;
- independently supplied historical class-order evidence;
- the pre-results execution dependency lock;
- the real non-scientific Kaggle infrastructure-smoke attestation.

After a G1 seal or any pair-init material exists, `seal_g1.py` refuses in-place regeneration.

### Exact execution dependency lock

Production execution is frozen before JE results to:

- Python 3.12.13
- torch 2.12.1
- torchvision 0.27.1
- timm 1.0.26
- NumPy 2.5.2
- Pillow 12.3.0
- safetensors 0.8.0
- kaggle 2.2.4
- huggingface-hub 1.30.0

The canonical notebook fails closed if Python differs and installs the exact requirements lock before
the repository execution code is invoked. The runtime validates every locked package again before G1,
G2 or principal execution.

### Source and session integrity

The notebook establishes one monotonic 12-hour clock before clone/install/bootstrap work and propagates
that start time to every subprocess. Children cannot receive a fresh wall-clock budget. Mutable output,
G1, G2 and terminal-evidence roots must remain outside the repository. The execution repository must be
at the exact authorized detached commit with a clean tracked and untracked working tree.

Private Git access uses a temporary askpass program with credentials supplied through the environment;
tokens are never embedded in clone URLs or committed Git configuration.

### Durable recovery

Scientific/calibration run locators must be unique per run ID. The Kaggle private-dataset backend is the
production path. The filesystem backend remains supported for controlled environments and now preserves
a recoverable previous generation across the target/staging rename window.

Real G1/G2 remains blocked until the real Kaggle synthetic smoke is green and the restricted model/data
and historical teacher-class-order evidence are available. Repository code never manufactures those
inputs.


## Stage 01A-SR — API-free cross-session smoke

The Stage-01A smoke path is now split into two explicit clean-session phases:

- `smoke-write`: create and verify a deterministic synthetic checkpoint, then export the complete
  recovery bundle beneath `/kaggle/working/cropcop-smoke-a-export` and stop.
- `smoke-restore`: in a fresh Saved Version, consume the exact attached Smoke-A Notebook Output from
  an explicit `/kaggle/input/...` root, verify it before any training/output write, restore/recover,
  load model/optimizer/scheduler/scaler, advance the optimizer step, write a separate Smoke-B export,
  publish audited text evidence, and stop.

This qualification path is deliberately **Kaggle-API-free**. Smoke A/B require only the Kaggle Secret
`CROPCOP_GITHUB_TOKEN`. They do not require `KAGGLE_USERNAME`, `KAGGLE_KEY`, Kaggle CLI dataset
creation/versioning/download, a private recovery dataset, or any CropCop dataset/model artifact.

The later `KagglePrivateDatasetStore` remains intact for future execution paths; Stage 01A-SR does not
redesign scientific persistence.

### Saved-Version handoff boundary

Smoke A writes:

```
/kaggle/working/cropcop-smoke-a-export/
    SMOKE_A_MANIFEST.json
    SMOKE_A_EVIDENCE.json
    checkpoint_index.json
    objects/<content-addressed-checkpoint>
```

The human operator uses **Save & Run All**, then attaches that exact Notebook Output to a fresh Smoke-B
Saved Version using **Add Input → Notebook Output Files**. Smoke B is given one explicit
`SMOKE_A_INPUT_ROOT` and searches only within that selected root. It fingerprints the attached input
before/after and never modifies it.

The machine-enforced Smoke-B sequence is:

`READ_A → VERIFY_A → RESTORE_A → RECOVER_A → LOAD_A → RESUME → CHECKPOINT_B`.

A checkpoint for B cannot be created until the state machine has reached verified resume.

### Source-SHA configuration note

The Smoke A/B state machine was first verified at `045fcf5c80366438b69a54288d9081e9b57ed973`, but the first real Kaggle
checkout exposed a repository line-ending canonicalization defect outside that state machine.
The clean execution source is now frozen and exact-head verified at
`939455c2cc8787bb295e073d706c32768820bfab` (Actions run #40, 124/124 tests PASS), including both a repository-wide
`git ls-files --eol` invariant and a brand-new exact-head Linux clone cleanliness test.
The later notebook-wrapper commit may safely hard-bind `AUTHORIZED_SOURCE_SHA` to this already
verified clean execution source without creating a commit-SHA self-reference. Real Smoke A/B now
executes repository code from exactly `939455c2cc8787bb295e073d706c32768820bfab`.


### Evidence-publication execution source

After the first real Smoke-A run reached public evidence publication, clean Kaggle exposed that the
temporary evidence worktree had no guaranteed Git author/committer identity. The publication layer was
hardened and exact-head verified at `67370145c9104edd52330b788c3b41b28f5cab87` (Actions run #43, 127/127 tests PASS). The canonical
notebook wrapper now binds to this source for the next Smoke-A retry.
