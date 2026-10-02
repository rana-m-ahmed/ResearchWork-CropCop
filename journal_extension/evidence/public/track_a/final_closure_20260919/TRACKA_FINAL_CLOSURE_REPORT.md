# CropCop Track-A Final Closure Report

**Gate:** `TRACK_A_CLOSED`  
**Analysis source:** `e08e471cc033902694622c12ceef03a19707b6dc`  
**Training-science source:** `56023042e57758591df9babb3438f191dbe10312`

## Terminal result

Track A is formally closed as a single **21-state** model-science unit. The frozen selector resolves to **R07 / ConvNeXt-Tiny Context** as the journal-primary family. Track-B and Track-C handoff of that sealed family is authorized.

## Verified closure conditions

- 21/21 state completions PASS (12 direct, 9 auxiliary).
- K1/K2/K3 terminal manifests PASS at 5/5, 8/8, 8/8.
- 21/21 private evidence round-trips verified.
- 21/21 publication packages byte-hash verified against terminal publication certificates.
- 0 V1-test accesses.
- 0 external predictions opened before selection.
- 0 Track-C candidate outcomes opened before selection.
- 0 post-training fitting/adaptation or optimizer advancement.
- Connected-GitHub audit: all 21 evidence branches descend from `e08e471cc033902694622c12ceef03a19707b6dc`, are zero commits behind it, and change only the approved Track-A public-evidence subtree.
- Global readiness byte SHA-256 independently matches GitHub: `db7ccc76b0e980c70d7dae44ce95a6d9f3e9872e2ac0bd5f9ce1a0d44060520c`.
- Immutable Wave-1 and Wave-2 closure bytes are bound by SHA-256 `f1a10405b2b049c763512b2562814f1fe603d9e6335f4231347e546300c89803` and `29a8f045729a36b126da8f470e91789569fc342c6dc051c9882e12661a68aee2`.

## Frozen selection

The nondominated frontier is R04/R06/R07. The pre-results frozen lexicographic selector resolves **R07**. No XAI quantity is given selector weight.

## Auxiliary conclusions

- R05 teacher minus R04 direct: mean Macro-F1 delta `-0.2470` pp.
- R12 logits minus feature: mean Macro-F1 delta `+0.2367` pp.
- No new hypothesis tests or multiple-comparison p-values are authorized.

## Claim boundary

Track A supports selection of the best frozen pretrained candidate **system under the common CropCop downstream protocol**. It does not establish a causal architecture-only superiority claim. Teacher guidance must not be described as a consistent performance improvement under the frozen three-seed evidence. XAI remains bounded diagnostic evidence, not validated biological localization.

## Handoff

`TRACK_A_CLOSED = true`  
`TRACK_B_HANDOFF_AUTHORIZED = true`  
`TRACK_C_HANDOFF_AUTHORIZED = true`  
`SEALED_PRIMARY_FAMILY = R07`

The next scientific work is Track B external/generalization evidence and Track C real-device/runtime evidence. No further Track-A compute is authorized unless a genuine validity defect is discovered and formally re-locked.
