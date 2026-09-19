# CropCop Track-A Final Closure Report

**Status:** LOCKED / PASS  
**Frozen analysis source:** `e08e471cc033902694622c12ceef03a19707b6dc`  
**Scientific states:** 21/21  
**Direct states:** 12/12  
**Auxiliary states:** 9/9  
**Private evidence round-trips:** 21/21  
**V1 test access during closure:** 0  
**External prediction access during selection:** 0  
**Track-C candidate-result access during selection:** 0  
**New training or optimizer advance during post-training closure:** 0  

## 1. Terminal decision

**GO - TRACK A CLOSED.**  
**GO - sealed Track-B and Track-C handoff authorized.**  
No further Track-A model-science compute is authorized under the current authority.

The frozen deterministic selector returned **SELECTED** and selected **R07** as the scientific-primary family. This is the best frozen pretrained candidate system under the common CropCop downstream protocol; it is not a causal architecture-superiority claim.

## 2. Evidence integrity

The repository's own post-training global audit consumed all 21 immutable published evidence branches and verified exact state inventory, publication manifests, completion chains, direct/auxiliary evidence contracts, private durability round-trips, source lineage, and protected-surface markers before scientific selection.

The original account-level aggregation files were not persisted to Git. For closure only, K1/K2/K3 aggregation manifests were deterministically reconstructed from the sealed account-readiness partitions plus each state's immutable published `POSTTRAINING_STATE_COMPLETION.json`. The reconstructed inputs contain no new scientific measurement and are retained under `closure_inputs/`.

## 3. Frozen direct-candidate selection

| Family | Mean macro-F1 | Worst-seed macro-F1 | Bottom-12 class mean F1 | Mean corruption degradation (pp) | FP32 state bytes | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| R04 | 0.959844 | 0.957995 | 0.758649 | 3.814 | 34,625,160 | 8,588,232 |
| R06 | 0.966386 | 0.965725 | 0.791383 | 3.851 | 16,813,528 | 4,161,268 |
| R07 | 0.966809 | 0.964824 | 0.801529 | 2.975 | 111,649,632 | 27,912,408 |
| R13 | 0.961707 | 0.960434 | 0.763693 | 4.457 | 88,941,024 | 22,235,256 |

The frozen Pareto/lexicographic selector selected **R07**. XAI remained an audit surface and was not used as a weighted selector.

## 4. Matched teacher/direct result

Three-seed validation macro-F1:
- R04 direct: **0.959844 +/- 0.001639**
- R05 teacher: **0.957374 +/- 0.003955**
- Teacher minus direct mean delta: **-0.002470**

The journal extension must not claim that teacher guidance consistently improves aggregate performance.

## 5. R12 mechanism ablation

Three-seed validation macro-F1:
- logits: **0.959639 +/- 0.001947**
- feature: **0.957272 +/- 0.002412**
- logits minus feature mean delta: **0.002367**

These are descriptive mechanism results. The frozen authority does not authorize post-hoc hypothesis tests or multiple-comparison p-values.

## 6. Journal-extension review alignment achieved by Track A

Track A now supplies the model-science evidence requested by the review in the areas assigned to this track: matched direct MobileNetV4 control, three-seed major comparisons, compact mechanism ablations, representative same-split direct candidates, exact selected-checkpoint replay, long-tail/classwise evidence, corruption robustness, model-state efficiency, and bounded XAI sanity checks.

Track A intentionally does not close independent external validity or physical-device deployment. Those are protected downstream tracks.

## 7. Permanent claim boundary

The authoritative allowed/forbidden claim set is `TRACKA_CLAIM_MATRIX.json`. Track A does not authorize field-generalization claims, physical-device claims, causal architecture-superiority wording, unsupported teacher-benefit wording, new V1-test selection, post-hoc auxiliary p-values, or causal localization claims from XAI.

## 8. Next action

Proceed to **Track B (external validity)** and **Track C (runtime/device)** using only the sealed Track-A scientific primary authorized by `TRACKA_COMPREHENSIVE_CLOSURE.json`.

The global manuscript evidence freeze remains downstream of Track B/C terminal evidence.
