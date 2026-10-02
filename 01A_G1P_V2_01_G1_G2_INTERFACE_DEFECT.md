# Stage 01A-G1P-v2 — Wave 1 G1→G2 Interface Defect

- **Artifact:** `01A_G1P_V2_01_G1_G2_INTERFACE_DEFECT.md`
- **Date:** 2026-09-06
- **Pre-fix witness head:** `f332253a77b4742c0c455388ab1c2ec115e3a898`
- **Exact-head workflow:** Actions #138 / run ID `34009257715`
- **Purpose:** reproduce the independent audit's claimed contract failure before remediation
- **Implementation changed before reproduction:** NO

## 1. Current producer/consumer contract

`journal_extension/scripts/validate_g1_barrier.py` requires:

```text
--dual-gpu-smoke-evidence
```

as a mandatory CLI argument.

`journal_extension/kaggle/run_g1.py` correctly passes that required argument when it validates a newly created G1 bundle.

However, the downstream lane consumer:

`journal_extension/kaggle/run_lane.py::validate_g1(...)`

constructs the `validate_g1_barrier.py` invocation without:

```text
--dual-gpu-smoke-evidence <CROPCOP_DUAL_GPU_SMOKE_EVIDENCE>
```

Therefore the G1 barrier contract differs between creation-time and downstream G2/principal validation.

## 2. Executable pre-fix regression

Added:

`tests/test_je_g1p_v2_wave1_contract.py`

The regression does not scan source text for a token.

It:

1. imports the actual `run_lane.py`;
2. executes `run_lane.validate_g1(...)`;
3. intercepts the actual subprocess argv constructed for `validate_g1_barrier.py`;
4. requires the mandatory dual-smoke argument and exact environment-resolved value.

Expected pre-fix assertion:

`--dual-gpu-smoke-evidence` must occur in the actual barrier argv.

## 3. Exact reproduced failure

Actions #138 reached the complete CPU-safe suite and failed exactly this regression:

```text
AssertionError: '--dual-gpu-smoke-evidence' not found in [
  ...,
  'validate_g1_barrier.py',
  '--repo-root', ...,
  '--authorized-source-sha', ...,
  '--g1-bundle-dir', ...,
  '--manifest', ...,
  '--class-map', ...,
  '--pretrained', ...,
  '--teacher-checkpoint', ...,
  '--teacher-factory-root', ...,
  '--infra-smoke-evidence', ...,
  '--output', ...,
]
```

The run summary was:

- QA1 Wave A: PASS
- QA1 Wave B: PASS
- QA1 Wave C: PASS
- MGPU: PASS
- full suite: **237 run / 1 failure**
- failure: exactly the new G1→G2 interface regression

This independently reproduces the audit finding.

## 4. Why the old tests missed it

Existing tests covered important pieces independently:

- terminal dual-smoke validation;
- G1 creation-time dual-smoke preflight;
- `seal_g1.py` dual-smoke gating;
- envelope-level dual-smoke gating;
- G1 object/seal checks.

But no executable contract test exercised the actual downstream `run_lane.validate_g1()` barrier argv against the complete mandatory parser contract.

The system therefore had individually tested producer and validator components without a shared typed/central barrier-argument contract.

This is CLI-contract drift.

## 5. Downstream consequence

On a real G2 or principal child path, `run_lane.validate_g1()` reaches:

`validate_g1_barrier.py`

without a required parser argument.

The barrier process therefore terminates before it can return PASS.

Consequences:

- G2 cannot legitimately start through that child path;
- principal cannot legitimately start through that child path;
- no optimizer step should be trusted past this broken prelaunch contract;
- a G1 bundle that was valid at creation time is not sufficient to make the downstream lane launchable under current code.

This is a technical execution blocker, not a scientific-design defect.

## 6. Source-bound consequence

The already-qualified real Smoke A/B and dual-smoke evidence bind source:

`fe88e426b4698977d65efe9702f1d48cf5ff96a3`

Any remediation of this interface necessarily changes execution source.

Therefore those real Smoke A/B + dual-smoke runs remain valid historical evidence for `fe88e...` only and cannot qualify the future repaired source.

No G1 has yet been produced, so there is no G1 object to migrate or grandfather.

## 7. Required remediation architecture

A one-line missing CLI argument would make this symptom disappear, but it would leave the same drift class possible.

The remediation should centralize the G1 barrier contract in a shared API/object used by:

- G1 creation-time validation;
- downstream lane validation;
- parent envelope fail-early validation.

The executable regression added in this wave must remain and become green only after that shared contract is wired.

## 8. Wave-1 verdict

# **PASS — CURRENT G1→G2 DEFECT REPRODUCED**

No scientific re-lock is implicated.
