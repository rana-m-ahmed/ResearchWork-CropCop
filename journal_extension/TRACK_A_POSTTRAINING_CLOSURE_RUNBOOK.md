# Track-A Post-Training Closure Runbook v1

**Authority:** `EAAI-JE-TRACKA-POSTTRAINING-CLOSURE-V1`  
**Training science source:** `56023042e57758591df9babb3438f191dbe10312`  
**Scope:** evidence recovery, selected-checkpoint post-training analysis, deterministic model selection, and comprehensive 21-state Track-A closure.  
**Science change authorized:** **NO**.

## Canonical notebook surfaces

Use only these generated notebooks for the campaign:

- `notebooks/tracka_posttraining_preflight.ipynb` — run once on K1/K2/K3 for continuation recovery and historical artifact availability.
- `notebooks/tracka_posttraining_execute.ipynb` — run first with `CROPCOP_ACCOUNT_PHASE=readiness`, then after global readiness with `CROPCOP_ACCOUNT_PHASE=evidence`.
- `notebooks/tracka_posttraining_global.ipynb` — run in order with `CROPCOP_GLOBAL_PHASE=placement`, then `readiness`, then `closure`.

The notebooks are generated deterministically by `generate_tracka_v12_posttraining_notebooks.py`. CI regenerates them and requires a byte-clean Git diff, so notebook code cannot silently diverge from the reviewed generator.

## 1. Non-negotiable boundary

All 21 Track-A training states are terminal. This runbook does not authorize training, adaptation, optimizer advance, new architectures, new seeds, hyperparameter changes, selector changes, V1-test access, Track-B predictions, or Track-C candidate-result access.

Authorized work is limited to:

1. recover/certify missing terminal metadata for the 11 v1.2 continuation states;
2. verify all 21 selected checkpoints and required executor inputs;
3. run the already-frozen post-training evidence executors;
4. seal the already-frozen selector and comprehensive closure.

## 2. Terminal metadata recovery

The final K1/K2/K3 account reports prove the 11 continuation runs are terminal `PASS` and bind run IDs and selected-checkpoint SHA-256 values. Per-run Git publication did not publish their `run_record.json` files.

The private generation-aware checkpoint stores retain the selected checkpoint/index. Each selected checkpoint contains the exact scientific identity and validation-selection state. Therefore the only authorized reconstruction route is:

`recover_tracka_v12_terminal_record.py`

Required outputs per state:

- `RECOVERED_TERMINAL_RUN_RECORD.json`
- `TERMINAL_RECOVERY_CERTIFICATE.json`

The recovery must cryptographically agree on run ID, selected SHA, checkpoint index, checkpoint identity, frozen config/CTC identity, seed, science source, selected epoch and selected validation metrics. It must never be represented as the original unpublished run record.

## 3. PT-0 — exact closure-analysis checkout

Use the additive post-training closure branch. Record the exact HEAD:

```bash
git rev-parse HEAD
```

That exact SHA becomes `CROPCOP_ANALYSIS_SHA` for all post-training evidence and closure artifacts.

CI must prove this branch is an additive descendant of `56023042...` and that the frozen direct/auxiliary/XAI/selector/closure implementations remain byte-identical to the training science source.

## 4. PT-1 — recover the 11 continuation records

Use the exact public account report plus the exact durable locator from:

`journal_extension/locks/track_a_posttraining_closure_authority_v1.json`

Example:

```bash
python journal_extension/scripts/recover_tracka_v12_terminal_record.py \
  --repo-root . \
  --experiment-id R13-VIT-DLITTLE-DIFF-CONTEXT-S1 \
  --run-id JE-R13-VIT-DLITTLE-DIFF-CONTEXT-S1-56023042e577-A01 \
  --account-report /input/K2/TRACKA_V12_K2_PUBLIC_REPORT.json \
  --checkpoint-root /kaggle/working/recovery/R13-S1/private_checkpoints \
  --durable-store-locator ranaabdulrehmannn/cropcop-r13-vit-dlittle-diff-con-696948b9-56023042 \
  --output-dir /kaggle/working/recovery/R13-S1
```

No training or protected inference is performed.

## 5. PT-2 — resolve the 10 historical artifact roots

Resolve the immutable historical states:

- R04 direct S1/S2/S3;
- R05 teacher S1/S2/S3;
- R06 S1;
- R07 S1;
- R12 logits S1;
- R12 feature S1.

For each, resolve the canonical terminal run record, selected checkpoint root/SHA, and any required historical initialization evidence. The exact historical lineage is already encoded in `tracka_v12_historical.py` and the immutable Wave-1/Wave-2 closures. No checkpoint substitution is allowed.

## 6. PT-3 — metric-blind placement, deterministic account inventories, and private evidence targets

First run the historical artifact-availability probe on K1/K2/K3. Then freeze placement with `freeze_tracka_v12_posttraining_placement.py`.

The placement freeze:

- pins all 11 continuation states to the account that owns their private checkpoint durability;
- assigns each historical state by the frozen account-preference order and pre-metric artifact availability only;
- assigns GPU0/GPU1 deterministically;
- records that no validation, robustness, XAI, runtime-quality, or selection outcome was opened before placement.

Next build one account inventory per K1/K2/K3 with `build_tracka_v12_posttraining_account_inventory.py`. Do not centralize private checkpoint bytes across accounts.

Each account inventory records:

- exact closure-analysis SHA;
- frozen manifest, class map and image root;
- exact G1A bundle;
- only the run records/checkpoint roots the account can actually verify;
- recovery certificates for continuation states assigned to that account;
- complete role-specific direct/XAI/auxiliary executor arguments;
- fixed account/GPU placement;
- a deterministic per-state private evidence dataset locator derived from the frozen campaign owner/template and exact analysis SHA.

Before account readiness, bootstrap or verify every assigned private evidence target:

```bash
python journal_extension/scripts/bootstrap_tracka_v12_posttraining_evidence_targets.py \
  --repo-root . \
  --account-id K1 \
  --account-inventory /kaggle/working/K1_POSTTRAINING_INVENTORY.json \
  --analysis-source-git-commit "$CROPCOP_ANALYSIS_SHA" \
  --allow-create 1 \
  --output /kaggle/working/K1_POSTTRAINING_EVIDENCE_TARGETS.json
```

Auto-creation is explicit opt-in only. Every target must settle as authoritatively private and its owner must equal the authenticated Kaggle account.

Then seal account readiness:

```bash
python journal_extension/scripts/build_tracka_v12_posttraining_account_readiness.py \
  --repo-root . \
  --inventory /kaggle/working/K1_POSTTRAINING_INVENTORY.json \
  --evidence-target-preflight /kaggle/working/K1_POSTTRAINING_EVIDENCE_TARGETS.json \
  --analysis-source-git-commit "$CROPCOP_ANALYSIS_SHA" \
  --output /kaggle/working/K1_POSTTRAINING_ACCOUNT_READINESS.json
```

Repeat for K2 and K3. Account readiness now fails before GPU work if any selected checkpoint, recovered record, historical lineage, executor path/argument, or private evidence target is incomplete or inconsistent.

## 7. PT-4 — seal the global `POSTTRAINING_READINESS_GATE`

Collect only the three public-safe account readiness certificates into one aggregation environment. The global gate must **not** require private checkpoint material.

Run:

```bash
python journal_extension/scripts/build_tracka_v12_posttraining_readiness.py \
  --repo-root . \
  --authority journal_extension/locks/track_a_posttraining_closure_authority_v1.json \
  --account-readiness /evidence/K1_POSTTRAINING_ACCOUNT_READINESS.json \
  --account-readiness /evidence/K2_POSTTRAINING_ACCOUNT_READINESS.json \
  --account-readiness /evidence/K3_POSTTRAINING_ACCOUNT_READINESS.json \
  --analysis-source-git-commit "$CROPCOP_ANALYSIS_SHA" \
  --output /evidence/POSTTRAINING_READINESS_GATE.json
```

The global gate must prove:

- exact account set K1/K2/K3;
- union is exactly 21 states = 12 direct + 9 auxiliary;
- exactly 21 unique selected checkpoint SHAs;
- all 11 continuation recoveries certified;
- all 10 historical states match immutable lineage;
- every private selected checkpoint was locally verified on an authorized account;
- every per-state evidence dataset was preflighted as owner-bound, settled and authoritatively private;
- all three gates share the same analysis source, training source, closure authority and frozen dataset identity;
- no state appears on two account gates;
- V1 test, Track-B predictions and Track-C candidate outcomes remain closed;
- no training/adaptation or optimizer advance occurred.

**No post-training analysis starts unless all three account gates and this global gate are PASS.**

## 8. PT-5 — direct evidence for all 12 candidate states

Run the frozen `run_tracka_v12_direct_evidence.py` for R04/R06/R07/R13 × S1/S2/S3.

Each state must produce:

- exact 16,368-row clean `DS-V1-VAL` replay;
- accuracy, balanced accuracy, macro-F1 and NLL;
- all 120 classwise precision/recall/F1/support;
- directed confusion summary;
- five corruption families × three severities;
- deterministic private-row evidence hashes;
- efficiency identity;
- `DIRECT_STATE_EVIDENCE_GATE.json`.

Replay must match the training-time selected validation metrics within the frozen `1e-6` tolerance.

## 9. PT-6 — XAI for all 12 direct states

Run the frozen `run_tracka_v12_xai.py`.

The protocol remains unchanged:

- 240 deterministic validation images;
- Grad-CAM++;
- R13 target `blocks.13.norm1`;
- deletion faithfulness at 10/20/30%;
- ten random controls per fraction;
- 30-image classifier-randomization sanity set;
- 30-image horizontal-flip consistency set;
- fixed 12-image qualitative panel;
- XAI has zero selector weight.

A `WARNING_NONFINITE_MAPS` outcome is reported transparently and is not a post-result reason to alter the selector.

## 10. PT-7 — auxiliary evidence

Run `run_tracka_v12_auxiliary_evidence.py` for:

- R05 S1/S2/S3;
- R12 logits S1/S2/S3;
- R12 feature S1/S2/S3.

Then run `seal_tracka_v12_auxiliary_analysis.py`.

These outputs are descriptive three-seed analyses only. No new hypothesis tests or multiple-comparison p-values are authorized.

## 11. PT-8 — evidence audit before selection

Before invoking the selector, verify every direct state has:

- one consistent selected-checkpoint SHA across replay/robustness/efficiency/XAI;
- one scientific source per state;
- one common closure-analysis source SHA;
- complete clean + 15-cell robustness evidence;
- exact XAI sample counts;
- inference-only markers;
- no protected-surface access.

The selector must reject a partial or mixed-lineage evidence index.

## 12. PT-9 — direct scientific-primary selection

Run the frozen selector exactly once:

```bash
python journal_extension/scripts/seal_tracka_v12_selection.py ...
```

Only R04/R06/R07/R13 participate. R05 and R12 remain auxiliary. There is no manual override.

This artifact alone must keep Track B and Track C closed.

## 13. PT-10 — comprehensive 21-state closure

Run:

```bash
python journal_extension/scripts/seal_tracka_v12_comprehensive_closure.py ...
```

Inputs are:

- direct-selection artifact;
- auxiliary-analysis artifact;
- immutable Wave-1 closure;
- immutable Wave-2 closure.

A PASS closure must resolve exactly 21 unique states.

If the selection status is `SELECTED`, the comprehensive closure may authorize Track-B/Track-C handoff of only the sealed scientific-primary family.

If the selection status is `CO_PRIMARY_TIE`, Track A closes scientifically but both downstream handoffs remain closed until the separately frozen deployment-feasibility tie gate resolves the tie.

## 14. Evidence persistence

**Private evidence:** checkpoint bytes, row-level clean predictions, row-level corruption predictions, XAI row evidence/panels, restricted initialization artifacts.

**Public-safe evidence:** recovery certificates, state evidence gates, robustness/efficiency/XAI summaries, auxiliary analysis, direct selection, comprehensive closure, and hashes of private evidence.

Never publish model/checkpoint/private dataset bytes to Git.

## 15. Final exit criteria

Track A is finished only when all are true:

1. 21/21 selected checkpoint lineages verified;
2. 11/11 continuation metadata recoveries certified;
3. readiness gate PASS;
4. 12/12 direct evidence bundles complete;
5. 12/12 XAI evidence bundles complete;
6. 9/9 auxiliary replay bundles complete;
7. auxiliary three-seed analysis PASS;
8. direct selector sealed;
9. comprehensive 21-state closure PASS;
10. public/private evidence hashes archived;
11. no V1-test, Track-B prediction, or Track-C candidate result influenced selection;
12. manuscript claim boundary remains candidate-system-under-frozen-protocol rather than architecture-only causality.

Only this state is `TRACK_A_CLOSED`.

## 16. Consumed V1 test remains a later gate

Do not open `DS-V1-TEST-CONSUMED` during Track-A closure.

Before any later one-shot use, create an explicit authority reconciliation between the original ten-state continuity rule and the strengthened selected-primary-only rule. This is intentionally outside the Track-A closure campaign.
