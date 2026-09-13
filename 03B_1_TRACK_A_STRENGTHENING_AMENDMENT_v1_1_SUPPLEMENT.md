# 03B.1 — Track-A Strengthening Amendment v1.1 Supplement

**Status:** `POST-RESULTS AMENDMENT SUPPLEMENT — PRE-EXECUTION LOCK`  
**Date:** 2026-09-13  
**Parent amendment:** `EAAI-JE-TRACKA-STRENGTHENING-A1`  
**Scientific outputs observed from A1 before this supplement:** `NONE`

## Purpose

This supplement removes the remaining preprint-model preference from the journal-extension design and adds a rigorously bounded explainability study.

### Model neutrality

MobileNetV4 is the **historical preprint architecture**, not the predetermined journal-primary model.

The journal-primary architecture will be selected only after all comparative Track-A evidence is complete, using `model_selection_protocol.json`. The primary candidate pool is restricted to the direct architecture families already frozen for Track A:

- MobileNetV4 direct;
- EfficientNet-B0 direct;
- ConvNeXt-Tiny direct.

R05 and R12 remain teacher/mechanism experiments and do not enter the primary architecture pool.

Track-B model inference and Track-C journal-candidate runtime evaluation are held until the Track-A model-selection decision is sealed. Model-independent Track-B provenance work and Track-C infrastructure preparation may continue.

### Explainability

A Grad-CAM++ study is added under `xai_gradcampp_protocol.json`.

It is not a decorative figure requirement. It uses:
- prediction-blind class-stratified sampling;
- all three direct architecture families across S1/S2/S3;
- a deterministic target-layer rule;
- perturbation-based faithfulness against matched random masks;
- map-degeneracy and sanity checks;
- deterministic qualitative panels.

Because the dataset does not provide lesion masks, no lesion-localization IoU or biological-causality claim is authorized.

### Selection order

Track A must complete:
1. three-seed clean evaluation;
2. robustness;
3. classwise analysis;
4. efficiency context;
5. Grad-CAM++ explainability/faithfulness audit.

Only then may the frozen model-selection protocol be applied.

After the selection record is committed:
- Track B may open selected-model external predictions after its independent-cohort gate;
- Track C may build/validate the selected journal deployment artifact.

The historical MobileNetV4 PTE remains valuable historical evidence but cannot constrain the journal model choice.

## Anti-fishing rule

No new architecture, seed, objective, explainability method, XAI sample-selection rule, selection weight, or selection criterion may be added after this supplement merely because an observed result is inconvenient.

# `GO — SUPERSEDE MODEL-SPECIFIC B/C HANDOFF CLAUSES; KEEP SCIENCE CLOSED UNTIL A1-G1A/A1-G2`
