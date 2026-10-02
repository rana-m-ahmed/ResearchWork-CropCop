# Stage 01A-MGPU — Prompt 2 Architecture Decision Record

- **Artifact:** `01A_MGPU_02_ARCHITECTURE_DECISION_RECORD.md`
- **Decision time:** 2026-09-05T20:34:00+05:00
- **Input sentinel:** `01A_MGPU_01_SCIENCE_DIFF_SENTINEL.json`
- **Status:** **DECISION COMPLETE**
- **Scientific re-lock required:** **NO**

## 1. Decision question

How should one Kaggle T4×2 Saved Version use both GPUs while preserving the exact Stage-03R R04/R05 contract?

The governing constraint is not “use both GPUs somehow.” It is:

> each R04/R05 scientific state must remain behaviorally the existing single-GPU run with micro-batch 16, accumulation 4, effective batch 64, its own RNG/checkpoint lineage, and the frozen validation-only selection rule.

## 2. Evaluation dimensions

Each candidate was challenged against:

- Stage-03R compliance;
- local BatchNorm/forward-batch semantics;
- optimizer-step semantics;
- RNG and deterministic augmentation;
- resume identity;
- direct↔teacher pair fairness;
- checkpoint identity;
- validation behavior;
- failure isolation;
- Kaggle CPU/RAM/I/O contention;
- Git publication safety;
- durable persistence;
- common 12-hour wall-clock budget;
- implementation churn;
- expected wall-clock benefit.

## 3. Candidate decisions

### Candidate 1 — Current single-GPU, one child per Saved Version

**Decision: ACCEPT AS THE SCIENTIFIC CHILD SEMANTICS; REJECT AS THE FINAL RESOURCE-UTILIZATION ENVELOPE.**

Strengths:

- exact Stage-03R behavior;
- local forward batch remains 16;
- accumulation remains 4;
- effective batch remains 64;
- no cross-GPU gradients or normalization state;
- current RNG, checkpoint, resume and validation code stays authoritative;
- simplest failure semantics.

Weakness:

- on a T4×2 host, one GPU remains idle during an individual Saved Version;
- the project has independent scientific/calibration states available, so this wastes legitimate parallel capacity.

This candidate is therefore the **child primitive** that the accepted envelope must reuse.

### Candidate 2 — Historical-style `nn.DataParallel`

**Decision: REJECT.**

Why:

- one scientific state would be spread across both devices;
- the per-device local forward batch would differ from the current single-GPU batch semantics;
- BatchNorm-bearing MobileNetV4 replicas would observe different local statistics;
- the optimizer/gradient path becomes a replicated multi-device execution path;
- RNG/augmentation and checkpoint portability require additional equivalence work;
- it increases scientific implementation churn for no need, because independent states can occupy GPU 2;
- PyTorch's direction favors DDP over DataParallel for multi-GPU training, so reviving DataParallel would add both scientific and engineering debt.

Expected wall-clock benefit for one state is not sufficient justification for reopening the frozen execution semantics.

### Candidate 3 — DDP without SyncBatchNorm

**Decision: REJECT.**

A superficially attractive mapping such as local batch 8 × 2 GPUs × accumulation 4 = effective 64 preserves only the arithmetic effective-batch number. It **does not preserve the existing local forward batch of 16**.

Consequences:

- BatchNorm sees local batch 8 instead of 16;
- per-rank RNG streams and distributed sampling enter the run;
- checkpoint/resume gains world-size/rank semantics;
- distributed validation and aggregation become new scientific execution machinery;
- exact equivalence would need qualification or re-lock.

Using local batch 16 with two ranks either changes accumulation or produces effective batch 128. Neither is authorized.

### Candidate 4 — DDP + SyncBatchNorm

**Decision: REJECT.**

SyncBatchNorm repairs one DDP mismatch by synchronizing statistics, but it creates a **different** computation from the current single-GPU local-batch-16 forward.

It also introduces:

- all-reduce synchronization;
- distributed RNG/sampler state;
- rank-aware checkpoint and resume;
- distributed validation;
- additional failure coupling;
- world-size dependence.

It could be a defensible future design only under a formal scientific re-lock and equivalence plan. That cost has poor risk/reward because independent states already exist.

### Candidate 5 — Teacher on one GPU, student on the other

**Decision: REJECT.**

This is not equivalent to “one existing single-GPU R05 run”:

- tensors must cross devices every batch;
- teacher/student execution timing and memory behavior change;
- device-transfer synchronization becomes part of the intervention path;
- checkpoint/recovery and failure handling become coupled across two physical devices;
- only teacher states benefit; direct R04 states do not;
- it complicates direct↔teacher fairness because only one condition gains a cross-device architecture.

There is no scientific requirement to split the teacher/student graph, and the second GPU can instead run the paired independent state.

### Candidate 6 — Two isolated existing single-GPU child runs

**Decision: ACCEPT — TARGET ARCHITECTURE.**

Topology:

```text
Kaggle Saved Version / execution envelope
├── child A: CUDA_VISIBLE_DEVICES=0 → existing run_lane.py → existing single-GPU trainer
└── child B: CUDA_VISIBLE_DEVICES=1 → existing run_lane.py → existing single-GPU trainer
```

Why it wins:

- each scientific/calibration child still sees exactly one CUDA device;
- local forward batch remains 16;
- accumulation remains 4;
- effective batch remains 64;
- no gradients/parameters/BatchNorm state cross GPUs;
- each child owns its own Python process, CUDA context, optimizer, scaler, dataloader, RNG and checkpoint lineage;
- direct/teacher same-seed pair fairness is preserved;
- current checkpoint scientific identity need not include physical slot;
- continuation may move a child to another equivalent T4 without making a new experiment;
- one child failure need not destroy a valid sibling;
- parent can strip Git credentials from children and serialize publication;
- mutable roots can be made disjoint while immutable inputs remain shared;
- G2's three fixed calibration states naturally form a 2+1 technical queue;
- P1/P2/P3 naturally place the direct and teacher member of one seed pair side-by-side;
- the scientific trainer remains unchanged.

## 4. Comparative matrix

| Candidate | Stage-03R | local batch/BN | optimizer semantics | RNG/resume | pair fairness | failure isolation | implementation churn | expected wall-clock |
|---|---|---|---|---|---|---|---|---|
| current single-GPU | **PASS** | exact | exact | exact | exact | strong | none | baseline, 1 GPU idle |
| DataParallel | **FAIL without re-lock/equivalence** | changed replica-local behavior | changed multi-device path | more complex | weaker | coupled | high | uncertain benefit |
| DDP | **FAIL without re-lock** | local batch changes | world-size path | rank-aware | weaker | coupled | high | potentially faster one state |
| DDP + SyncBN | **FAIL without re-lock** | synchronized, not current local semantics | world-size path | rank-aware | weaker | coupled | very high | potentially faster one state |
| teacher/student split | **FAIL equivalence burden** | condition-specific topology | cross-device teacher path | coupled | asymmetric | coupled | high | R05-only benefit |
| two independent single-GPU children | **PASS** | exact per child | exact per child | isolated/existing | exact | strong | moderate orchestration-only | up to ~2 concurrent states |

The final row's benefit is deliberately stated as concurrency, not a promised 2× speedup. CPU, RAM, dataloader, storage and checkpoint I/O contention can reduce realized throughput. G2 must measure the qualified profile before principal launch.

## 5. Kaggle contention decision

The accepted design initially uses:

- `CROPCOP_NUM_WORKERS_PER_CHILD=2`;
- `OMP_NUM_THREADS=1`;
- `MKL_NUM_THREADS=1`;
- `OPENBLAS_NUM_THREADS=1`;
- `NUMEXPR_NUM_THREADS=1`;
- `TOKENIZERS_PARALLELISM=false`.

This is execution-only resource control. It does not alter sample order, augmentation semantics, losses, batch size, accumulation or checkpoint selection.

If G2 later proves that a different loader-worker count is technically better, the affected G2 profile must be rerun **before** principal launch. Model-quality outcomes may not drive the change.

## 6. Publication and durability decision

Children must not possess:

- `CROPCOP_GITHUB_TOKEN`
- `GITHUB_TOKEN`

The parent retains credentials and serializes public-safe evidence publication after child outputs are stable.

Scientific/calibration durability remains the current production backend. The MGPU refactor may validate it and enforce unique per-run locators, but it may not silently replace production recovery with ephemeral notebook storage.

## 7. Wall-clock and continuation decision

All children inherit the **same** `CROPCOP_NOTEBOOK_STARTED_MONOTONIC`.

The parent may not start a new child when the common safe deadline cannot accommodate useful work plus checkpoint/durable finalization.

If one principal child is terminal PASS and its sibling requires continuation:

- PASS child is not relaunched;
- only the nonterminal child resumes;
- same run ID/checkpoint lineage is retained;
- physical slot may change;
- unused GPU remains idle unless a future explicit envelope authorizes other work.

## 8. Formal verdict

| Candidate | Verdict |
|---|---|
| current single-GPU execution | **ACCEPT as child semantic primitive / REJECT as final T4×2 envelope** |
| DataParallel | **REJECT** |
| DDP | **REJECT** |
| DDP + SyncBatchNorm | **REJECT** |
| teacher/student GPU split | **REJECT** |
| two independent single-GPU children | **ACCEPT** |

`RE-LOCK` is not required because the accepted architecture can be implemented entirely above the frozen scientific computation.

## 9. Prompt-2 gate

# **ACCEPT — DUAL-INDEPENDENT-SINGLE-GPU EXECUTION ENVELOPE**

Implementation is now authorized, subject to the science-diff sentinel and the additive Stage-04 execution amendment.
