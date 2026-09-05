# CropCop EAAI Journal Extension — Lean JE Core

This namespace implements the minimum Stage-03R-bound machinery required before R04/R05 training.
It is intentionally fail-closed: restricted dataset/model bytes are supplied at execution time and
are verified before use; V1 test and sealed external surfaces are not resolvable by the training API.

## Authority

- Scientific lock: `EAAI-JE-SDL-v2.1-QA`
- Lock SHA-256: `aab17b65b0873dcb1ecedb061eb02ff60ccb09f8b830184f5e2231a600278f74`
- Repository baseline at Stage 00: `32190dd86293caa82170df3feea505e3c7443b4b`

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
