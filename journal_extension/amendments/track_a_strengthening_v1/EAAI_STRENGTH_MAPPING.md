# EAAI Acceptance-Strength Mapping — CropCop Journal Extension

**Status:** `PLANNING / PRE-EXECUTION MAPPING`  
**Date:** 2026-09-13  
**Scope:** Maps the journal-extension evidence plan to current *Engineering Applications of Artificial Intelligence* (EAAI) expectations and reviewer-visible plant-health AI gaps. This is a planning document, not an acceptance guarantee.

## 1. EAAI-facing thesis

The journal extension should not be framed as “MobileNetV4 with more experiments” or as a new architecture paper. Its strongest EAAI-facing contribution is an **auditable engineering study** that reconstructs a 120-class public benchmark, challenges the historical training assumptions under controlled multi-seed evidence, compares candidate architectures without preselecting a winner, stress-tests reliability, selects the journal-primary model under a frozen evidence rule, and then sends that selected model through independent external-validity and deployment/runtime tracks.

EAAI explicitly emphasizes practical engineering application, public-data reproducibility, verification/validation of AI software, safety/reliability, real-time software/hardware architecture and benchmarking. The CropCop extension should align its claims to those dimensions.

## 2. Strength map

| EAAI/reviewer dimension | Evidence before A1 | Planned evidence after A1 / B / C | Expected strength | Residual risk |
|---|---|---|---|---|
| Real engineering application | 120-class plant-health diagnosis; offline edge target | selected-model external + runtime validation | **Very strong** | must avoid medical/agronomic autonomy claims |
| Public-data reproducibility | audited reconstruction, frozen manifest/class map, public evidence | amendment evidence and claim map | **Very strong** | manuscript must explain source-family overlap clearly |
| Verification / validation | immutable checkpoints, deterministic replay, exact hashes | all new states replayed; robustness/XAI protocols | **Exceptional for application paper** | complexity of evidence must be presented clearly |
| Multi-seed model science | principal R04/R05 already 3-seed | R06/R07/R12 become 3-seed | **Very strong** | n=3 remains descriptive; avoid significance inflation |
| Architecture benchmarking | S1-only context | 3-seed MNV4/EFFB0/CNXTT + neutral selection gate | **Very strong** | no late architecture additions |
| Robustness / reliability | clean validation | 5 corruptions x 3 severities x 9 direct states | **Strong** | synthetic corruption != field validity |
| 120-class reliability | aggregate metrics | classwise tail + confusion analysis | **Strong** | no expert pathology adjudication yet |
| Explainability / trust | absent | Grad-CAM++ + quantitative perturbation faithfulness + sanity checks | **Moderate-to-strong supporting evidence** | no lesion masks; cannot claim localization accuracy |
| External generalization | unresolved | Track B independent-cohort gate and selected-model inference | **Potentially decisive** | candidate may fail independence/ontology gate |
| Edge/runtime engineering | historical MNV4 artifact | Track C re-targeted to the Track-A-selected journal model | **Potentially decisive** | selected architecture may prove expensive or unsupported |
| Efficiency evidence | historical MNV4 artifact size | params/state bytes for all candidates + real runtime for selected model | **Strong if C succeeds** | FLOPs alone must not be sold as latency |
| Model-selection neutrality | preprint centered MNV4 | model-neutral Pareto/lexicographic gate frozen before new results | **Very strong** | must preserve historical MNV4 as history, not preference |
| Negative-result integrity | teacher benefit not supported | original conclusion preserved regardless of amendment outcomes | **Very strong** | wording must remain “under frozen protocol” |
| AI novelty | not a novel architecture | auditable reconstruction + evidence-driven selection + robustness/external/runtime validation | **Moderate-to-strong engineering novelty** | abstract must make AI contribution explicit; avoid pretending architectural novelty |
| Reproducible post-results amendment | not applicable | amendment separately disclosed, hashed, fail-closed | **Very strong** | manuscript must distinguish original vs amendment evidence |

## 3. Why Grad-CAM++ is included

Grad-CAM++ is included because explainability is common in contemporary plant-disease AI and can improve the trust/debugging story, but qualitative heatmaps alone are weak evidence. The protocol therefore requires deterministic sampling, perturbation-based faithfulness against matched random masks, map-degeneracy checks and sanity checks.

Without lesion or symptom masks, the paper will **not** report IoU/localization accuracy and will not claim the model “looks at the disease lesion.” The honest claim is that Grad-CAM++ provides a reproducible post-hoc view of class-discriminative regions and that perturbation tests quantify whether highly attributed regions are more influential than matched random regions.

## 4. Highest-impact remaining dependencies

1. **Finish strengthened Track A without model preference.**
2. **Freeze and apply the model-selection gate only after every comparative Track-A study is complete.**
3. **Track B:** obtain a genuinely source-independent external cohort. A failed independence gate is preferable to contaminated “external” evidence.
4. **Track C:** quantize/export/runtime-test the selected journal-primary architecture, not automatically MobileNetV4.
5. **Synthesis:** if the scientifically strongest model is not the best deployable model, report both roles explicitly instead of rewriting Track A.

## 5. Reviewer attack-surface map

### Closed or substantially reduced
- “single lucky run” -> three seeds for principal and candidate architectures/mechanisms.
- “why MobileNetV4?” -> no architecture is preselected; journal-primary is evidence-selected.
- “aggregate accuracy hides class failures” -> 120-class tail/confusion analysis.
- “controlled validation says little about field conditions” -> corruption robustness plus Track-B external gate.
- “lightweight claim is hand-wavy” -> exact size/parameter evidence plus Track-C device runtime.
- “black-box diagnosis” -> Grad-CAM++ with quantitative faithfulness and sanity checks.
- “technical retries changed science” -> immutable run identities and recovery evidence.

### Still requiring careful manuscript framing
- n=3 does not justify broad population-level significance claims.
- synthetic corruptions do not establish external field generalization.
- Grad-CAM++ does not establish biological causality or lesion localization without annotations.
- external validation may be impossible if no provenance-clean cohort survives.
- EAAI fit depends on presenting the work as AI engineering/verification/benchmarking, not only a plant classifier.

## 6. Recommended final contribution stack

The strongest manuscript contribution stack is:

1. **Audited benchmark reconstruction and provenance control** for a 120-class public plant-health problem.
2. **Controlled model-science reconstruction** showing what did and did not survive from the preprint.
3. **Three-seed, model-neutral architecture/mechanism benchmarking** with an evidence-frozen selection gate.
4. **Reliability analysis:** corruption stress tests + 120-class tail/confusion evidence.
5. **Reproducible explainability audit:** Grad-CAM++ plus perturbation faithfulness/sanity checks.
6. **Independent external validity** on the selected model if a clean cohort passes Track B.
7. **Real deployment validation** of the selected model under Track C.
8. **End-to-end auditability:** immutable identities, checkpoint replay, public-safe evidence and explicit post-results amendment history.

This stack is stronger for EAAI than adding more architectures or tuning until one model wins.
