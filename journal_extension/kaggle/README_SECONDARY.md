# Track-A Secondary Kaggle Execution

This directory contains the additive dual-T4 execution path for the four fixed secondary Track-A states and the inference-only principal validation evidence backfill.

## Phases

The additive repository runners support the secondary phases. A thin secondary canonical Kaggle wrapper is generated only after the core secondary source receives exact-head CI and is frozen to that immutable source SHA. The wrapper then exposes explicit phases for source-qualified smoke, secondary G1, secondary G2, secondary science, and principal validation backfill. No phase falls through automatically into another scientific phase.

## Account allocation

Use K1 for `SEC-MECHANISM-T4X2-V1`, K2 for `SEC-CONTEXT-T4X2-V1`, and K3 initially for `VAL-BACKFILL-S3-T4X2-V1`. After K1/K2 secondary science is terminal, reuse those account-local T4 pairs for S1/S2 validation backfill.

## Required invariants

- exact immutable secondary source SHA;
- exact locked software environment;
- exactly two Tesla T4 devices for dual envelopes;
- exactly one visible CUDA device per child;
- source-qualified Smoke-B and dual-GPU smoke before secondary G1;
- one immutable shared secondary G1 bundle for K1/K2;
- terminal secondary G2 barrier with explicit MobileNetV4, EfficientNet-B0 and ConvNeXt-Tiny construction coverage before any of the four secondary scientific states;
- real save→private-dataset→fresh-local-restore→resume durability qualification during G2;
- independent durable locator per scientific child;
- parent-side fresh-vs-resume resolution: checkpoint present → `required`, pristine v1 target → `never`, ambiguous target → hard fail;
- no V1-test or protected external access before G5.

## Operator variables

Shared scientific/data variables include `CROPCOP_MANIFEST`, `CROPCOP_CLASS_MAP`, `CROPCOP_IMAGE_ROOT`, the manifest column names, `CROPCOP_SOURCE_GIT_COMMIT`, and `CROPCOP_SECONDARY_G1_INPUT_ROOT`.

Secondary science additionally requires `CROPCOP_SECONDARY_G2_BARRIER`, `CROPCOP_DURABLE_STORE_KIND=kaggle-dataset`, and a per-run `CROPCOP_DURABLE_LOCATOR_TEMPLATE` containing `{run_id}`.

`CROPCOP_SECONDARY_ENVELOPE` is `mechanism` on K1 or `context` on K2. `CROPCOP_VALIDATION_BACKFILL` is `S1`, `S2`, or `S3` on the account that owns those principal durable datasets.

`CROPCOP_NUM_WORKERS_PER_CHILD` is optional. If omitted, the secondary runner chooses a conservative value from the available CPU count. This may be adjusted without changing scientific semantics.

## Continuation rule

Scientific Saved Versions always reuse the same source-derived run ID and private durable target. The parent probes the target before launch. If `checkpoint_index.json` exists, resume is mandatory. A pristine first-version target may launch fresh. A later-version target without a checkpoint index is treated as ambiguous/corrupt and execution stops rather than restarting science. Restore exceptions are also fatal. Completed checkpoints may be rehydrated only to recognize terminal completion; no extra optimizer step is allowed.

## G2 queue

A single T4x2 G2 Saved Version runs a four-profile queue: MobileNetV4 direct + teacher first, then EfficientNet-B0 and ConvNeXt-Tiny on the first freed slots. Each profile exercises exact one-GPU isolation, frozen train/validation surfaces, checkpoint create, real Kaggle-private durable sync, deletion of the local checkpoint directory, durable restore, and resumed optimizer progress. The parent also runs selected-checkpoint verification, checkpoint-identity mismatch rejection, and a double-publication idempotency probe.
