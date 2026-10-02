# Stage 01A-MGPU-QA1 — Live Confirmation

- **Artifact:** `01A_MGPU_QA1_00_LIVE_CONFIRMATION.md`
- **Date:** 2026-09-05
- **Repository:** `rana-m-ahmed/ResearchWork-CropCop`
- **PR:** #8 / `je-stage01a-core-20260905`
- **Live PR head inspected:** `e7ad831dae327c4e81b9417e46cb47c5237354af`
- **Live main:** `32190dd86293caa82170df3feea505e3c7443b4b`
- **PR state:** OPEN / DRAFT
- **Latest exact-head CI:** run #81 / ID `33976564582` / SUCCESS
- **Pre-QA MGPU execution source:** `ba5dd4661b97d072593af4646b76552686953a2d`
- **Recorded final wrapper:** `7e67ed785c9a3a89d892f9f82d28bd8dc471bea4`
- **Status:** **QA1 FINDINGS REPRODUCED — REMEDIATION AUTHORIZED**
- **Scientific re-lock:** **NOT REQUIRED**

## 1. Live authority and integrity state

Verified live repository state:

- Stage-03R authority: `EAAI-JE-SDL-v2.1-QA`
- Stage-03R SHA-256: `aab17b65b0873dcb1ecedb061eb02ff60ccb09f8b830184f5e2231a600278f74`
- Stage-04 base SHA-256: `a1bc1f8bb2d7a133a46c3f59f83dd2ba8ca3bf0fa8d77ae323553276931bc079`
- Stage-04A authority: `EAAI-JE-MGPU-A1`
- Stage-04A SHA-256: `3f08f2dbe7e83143c7e6f1fcf6732c7d88086295a600a74b323b89b46caed3f6`
- science-diff: PASS
- live head CI #81: compile PASS, repository validator PASS, JE static PASS, science-diff PASS, amendment hash PASS, canonical notebook compile/generation consistency PASS, secret/artifact scans PASS, MGPU 66/66 PASS, complete suite 193/193 PASS.

The final `e7ad831d...` commit relative to wrapper `7e67ed78...` is documentation-only.

## 2. Active wrapper/operator state

The canonical generator/notebook correctly bind the pre-QA MGPU execution source:

`ba5dd4661b97d072593af4646b76552686953a2d`

and expose:

- `smoke-write`
- `smoke-restore`
- `dual-gpu-smoke`
- `g1`
- `calibration-dual`
- `principal-dual`

However, active operator files are stale:

- `journal_extension/kaggle/README_SMOKE.md` still directs execution at `67370145c9104edd52330b788c3b41b28f5cab87`;
- `journal_extension/kaggle/smoke_inputs.example.json` still binds the same superseded SHA;
- `journal_extension/README.md` still contains the older pre-MGPU operator phase model.

## 3. Independent finding classification

| Finding | Classification | Live reproduction |
|---|---|---|
| S0-1 — 04A chronology not machine-enforced | **CONFIRMED** | `smoke_dual_gpu.py` requires terminal Smoke-B, but `run_g1.py`, `seal_g1.py` and `run_envelope.py` do not require terminal dual-GPU-smoke evidence. Dual-smoke evidence also lacks a digest binding to the exact Smoke-B object. |
| S1-2 — graceful finalization can kill valid child after 30 s | **CONFIRMED** | `terminate_process_group(..., grace_seconds=30.0)` sends SIGTERM, waits only 30 s and then SIGKILLs. Global-stop paths call it directly and break before normal child result collection. |
| S1-3 — publication failure can be laundered through continuation | **CONFIRMED** | child result may contain `publication_status=FAIL`, but state records the child only as `PASS`; `continuation_skip_set()` skips every PASS child without evidence-completeness validation. |
| S1-4 — fully completed epoch-30 checkpoint recovery edge | **CONFIRMED** | scientific resume uses `start_epoch=payload["epoch"]`; when restored epoch equals locked epoch count, the training loop runs zero iterations while terminal return still depends on loop-produced `latest_ref`. Execution wrapper has no pre-loop completed-checkpoint terminal reconstruction. |
| S2-5 — durable access preflight unnecessarily global | **CONFIRMED** | `durable_plan()` validates locator uniqueness and authenticated access against all 3 calibration + 6 principal reserved IDs for every envelope. |
| S1-6 — active docs point to superseded source | **CONFIRMED** | active smoke README/example still name `67370145...`; broader JE README still documents old `smoke/g1/calibration/principal` phase flow. |
| S2-7 — no explicit Smoke-B → dual-smoke / dual-smoke → G1 wrapper handoff | **CONFIRMED** | wrapper has explicit `SMOKE_A_INPUT_ROOT` only. `dual-gpu-smoke` expects `CROPCOP_INFRA_SMOKE_EVIDENCE`, but canonical wrapper does not resolve exact Smoke-B evidence from an explicit input root; no explicit dual-smoke input root exists for G1/later phases. |
| S2-8 — continuation trusts state more than full prior bundle | **CONFIRMED** | continuation locates one `ENVELOPE_STATE.json` and validates state fields/run IDs only. It does not require coherent manifest/evidence/state triple or result-artifact presence before skip decisions. |

No audit finding is classified `ALREADY_FIXED` or `NOT_REPRODUCIBLE`.

## 4. Residual / optional findings

### Periodic durable sync

Confirmed residual fact:

- local periodic checkpoints occur every 250 optimizer steps;
- Layer-B durable sync occurs when the execution segment returns, not after every local periodic checkpoint.

This is recorded only. QA1 will not introduce aggressive per-checkpoint Kaggle dataset versioning.

### Physical UUID attribution

The current design uses parent numeric slot→UUID inventory plus child one-visible-T4 proof. This remains a bounded homogeneous-T4 assumption and is not a QA1 blocker.

## 5. Protected science boundary

QA1 will preserve byte-identical unless an exact blocker proves otherwise:

- `train.py`
- `data.py`
- `models.py`
- CTC-v2
- all R04/R05 configs
- experiment registry/seeds/objectives/model identities

No DDP/DataParallel/FSDP, teacher/student split, protected evaluation access, G1/G2/R04/R05 launch, or real Kaggle Smoke A/B is authorized during repository remediation.

## 6. Gate

# **GO — WAVE A REMEDIATION MAY BEGIN**

Wave A is restricted to chronology binding, dual-smoke evidence validation/binding, explicit operator handoffs, active operator documentation and corresponding tests/CI.
