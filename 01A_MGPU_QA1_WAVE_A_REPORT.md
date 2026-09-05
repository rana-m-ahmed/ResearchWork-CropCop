# Stage 01A-MGPU-QA1 — Wave A Report

- **Wave:** A — Binding Gates + Operator Handoff
- **Exact-head commit:** `a439277ba70dd3638a955231d39e2bbddaaa8e8d`
- **GitHub Actions:** run #95 / ID `33978283923`
- **Conclusion:** SUCCESS
- **Science-diff:** PASS
- **Wave-A tests:** 18 / 18 PASS
- **Existing MGPU tests:** 66 / 66 PASS
- **Complete CPU-safe suite:** 211 / 211 PASS
- **Scientific re-lock:** NOT REQUIRED

## Closed findings

### S0-1 — 04A chronology

Closed.

One canonical `validate_terminal_dual_gpu_smoke_evidence(...)` now validates the terminal technical evidence contract, including exact source/dependency/amendment, Batch execution, T4×2 inventory and child isolation, overlap, optimizer/checkpoint progress, science-diff, publication and exact Smoke-B digest binding.

`dual-gpu-smoke` now records:

`smoke_b_evidence_sha256 = sha256_json(terminal_smoke_b_object)`

and self-validates its terminal evidence after publication metadata exists.

Terminal dual-smoke evidence is now required before:

- G1 launch;
- G1 sealing;
- G1 barrier validation;
- `calibration-dual`;
- `principal-dual`.

This remains technical launch evidence and was not added to scientific checkpoint identity or the G1 scientific seal.

### S2-7 — explicit operator evidence handoff

Closed.

The canonical wrapper now exposes:

- `CROPCOP_SMOKE_B_INPUT_ROOT`;
- `CROPCOP_DUAL_GPU_SMOKE_INPUT_ROOT`.

For `dual-gpu-smoke`, exactly one `SMOKE_B_EVIDENCE.json` must exist below the explicit Smoke-B root.

For G1/later dual phases, exactly one terminal Smoke-B evidence and one `DUAL_GPU_SMOKE_EVIDENCE.json` must exist below their separate explicit roots.

No global `/kaggle/input` evidence search is used.

### S1-6 — active operator documentation

Wave-A portion closed.

The active smoke README, smoke-input example and marked current section of `journal_extension/README.md` now describe the six-phase MGPU chronology and current pre-QA source. Historical reports/chronology sections were not rewritten.

A machine validator now prevents the active files from drifting from:

- generator-authorized source;
- exact six-phase vocabulary;
- Smoke A → Smoke B → audit → dual smoke → audit → G1 chronology.

The source identifier will be updated again only after the final QA1 source freeze in Wave C.

## Exact-head verification

Run #95 proved at exact head `a439277...`:

- expected checkout SHA == actual checkout SHA;
- compile PASS;
- repository contract validator PASS;
- JE static validator PASS;
- science-diff PASS;
- Stage-04A hash PASS;
- canonical notebook code compile PASS;
- generator == committed notebook PASS;
- forbidden-artifact scan PASS;
- secret scan PASS;
- active operator-doc validator PASS;
- Wave-A 18/18 PASS;
- MGPU 66/66 PASS;
- total 211/211 PASS.

## Protected science

No protected Stage-03R file changed. In particular, `train.py`, `data.py`, `models.py`, CTC-v2, R04/R05 configs and experiment registry remain governed by the existing science sentinel.

# **WAVE A PASS — WAVE B MAY BEGIN**
