# Track-A Independent Pre-Closure Audit — 2026-09-19

## Gate

**GO AFTER REQUIRED ARCHIVAL RECOVERY**

Track-A scientific execution is complete and internally consistent, but the repository-side final closure must not be marked `TRACK_A_CLOSED` until the three original terminal account execution manifests are supplied to the frozen global-audit controller.

## 1. Scope and authority

This audit treats Track A as the final 21-state unit authorized by `EAAI-JE-TRACKA-POSTTRAINING-CLOSURE-V1` and `CROPCOP-TRACKA-POSTTRAINING-CLOSURE-V1`, not as the earlier ten-state launch matrix.

Frozen sources:
- training science: `56023042e57758591df9babb3438f191dbe10312`
- post-training analysis: `e08e471cc033902694622c12ceef03a19707b6dc`

No scientific semantics change is authorized by this branch.

## 2. Whole-campaign evidence audit

Verified directly against the connected repository:

- 21/21 expected post-training evidence branches exist.
- 12/12 direct states are present.
- 9/9 auxiliary states are present.
- 21/21 `POSTTRAINING_STATE_COMPLETION.json` records report `PASS`.
- 21/21 publication manifests report `PASS`.
- 21/21 completions report private-generation round-trip verification.
- 0/21 report V1-test access.
- 0/21 report external-surface access.
- 0/21 report post-training fitting/adaptation.
- 21/21 evidence branches descend from `e08e471...`.
- 21/21 are zero commits behind `e08e471...`.
- 0/21 modify any path outside `journal_extension/evidence/public/track_a/posttraining/`.

This is sufficient to conclude that the *scientific state execution and state-level publication layer* is complete.

## 3. Frozen direct-selector reproduction

The exact frozen selector uses mean validation macro-F1, worst-seed macro-F1, bottom-12-class mean F1, corruption degradation, model-state bytes, and parameter count. XAI has zero selector weight.

Independent reproduction from the 12 direct public evidence bundles yields:

| Family | Mean macro-F1 | Worst-seed macro-F1 | Bottom-12 mean F1 | Mean corruption degradation (pp) | FP32 state bytes |
|---|---:|---:|---:|---:|---:|
| R04 MobileNetV4 Direct | 0.959844 | 0.957995 | 0.758649 | 3.813593 | 34,625,160 |
| R06 EfficientNet-B0 Context | 0.966386 | 0.965725 | 0.791383 | 3.850706 | 16,813,528 |
| R07 ConvNeXt-Tiny Context | **0.966809** | 0.964824 | **0.801529** | **2.974573** | 111,649,632 |
| R13 ViT/D-Little Context | 0.961707 | 0.960434 | 0.763693 | 4.456849 | 88,941,024 |

R13 is Pareto-dominated by R06. The nondominated frontier is R04/R06/R07. The frozen lexicographic selector therefore resolves to:

**R07 / ConvNeXt-Tiny Context — SELECTED**

This is a dry-run reproduction only. The official selector must still be sealed by `close_tracka_v12.py` after the frozen global evidence audit passes.

## 4. Auxiliary three-seed analysis

Frozen descriptive analysis gives:

### R05 teacher minus R04 direct
- mean accuracy delta: **+0.0081 percentage points**
- mean macro-F1 delta: **−0.2470 pp**
- mean balanced-accuracy delta: **−0.2205 pp**
- mean NLL delta: **−0.01112**

Therefore the journal extension does **not** support a claim that teacher guidance consistently improves the matched MobileNetV4 model.

### R12 logits minus R12 feature
- mean accuracy delta: **+0.0387 pp**
- mean macro-F1 delta: **+0.2367 pp**
- mean balanced-accuracy delta: **+0.2733 pp**
- mean NLL delta: **−0.01107**

These are descriptive three-seed contrasts. The frozen analysis explicitly authorizes **no new hypothesis tests or multiple-comparison p-values**.

## 5. XAI boundary

All 12 XAI gates report `PASS` and remain inference-only. XAI is not used in model selection.

The XAI evidence must nevertheless be discussed cautiously. Several families show negative saliency-vs-random confidence-drop advantage, and R07 shows high degenerate-map rates (~0.58–0.67 across seeds) despite finite maps. This evidence supports a bounded diagnostic/interpretability analysis, not a claim of biologically validated or causally faithful localization.

## 6. Formal-closure blocker

The frozen script `audit_tracka_v12_posttraining_evidence.py` requires exactly three original terminal account execution manifests and compares their embedded state-completion payloads with the state publications.

Expected originals:
- `TRACKA_GPU_K1_e08e471cc033.zip/ACCOUNT_GPU_COMPLETION.json`
- `TRACKA_GPU_K2_e08e471cc033.zip/ACCOUNT_GPU_COMPLETION.json`
- `TRACKA_GPU_K3_e08e471cc033.zip/ACCOUNT_GPU_COMPLETION.json`

The v5 notebook explicitly copies `ACCOUNT_GPU_COMPLETION.json` into each final QA ZIP. Those ZIPs are not currently present in this chat/library/repository audit environment.

**Do not synthesize or reconstruct these account manifests merely to satisfy the validator.** Doing so would weaken the independent closure contract.

## 7. Exact terminal sequence after artifact recovery

1. Place the original K1/K2/K3 `ACCOUNT_GPU_COMPLETION.json` files beside the already-published global readiness gate.
2. Checkout exact `e08e471cc033902694622c12ceef03a19707b6dc`.
3. Run `audit_tracka_v12_posttraining_evidence.py`.
4. Require:
   - 21-state union,
   - 12 direct,
   - 9 auxiliary,
   - 21 private round-trips,
   - exact published completion equality,
   - publication-manifest/hash validation,
   - selector and auxiliary analysis authorization,
   - all protected surfaces closed.
5. Run `close_tracka_v12.py`.
6. Require:
   - `TRACKA_AUXILIARY_ANALYSIS.json = PASS`
   - `TRACKA_DIRECT_SELECTION.json = PASS`
   - `TRACKA_COMPREHENSIVE_CLOSURE.json = PASS`
   - `TRACKA_FINAL_CLOSURE_AUDIT.json = PASS`
   - `track_a_closed = true`
   - selection terminal status `SELECTED`
   - Track-B and Track-C handoff authorized.
7. Only then emit **TRACK_A_CLOSED** and begin downstream candidate-specific Track B/C execution.

## 8. Current conclusion

The scientific campaign itself is complete: 21 states, 21 immutable checkpoint lineages, all required direct/auxiliary post-training evidence, no protected-surface contamination, and a deterministic frozen-selector outcome.

The only remaining blocker is **archival recovery of the three original account-completion manifests**. Until that is satisfied, Track A is scientifically complete but not formally closed under its own frozen controller.
