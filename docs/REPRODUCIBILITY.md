# Reproducibility guide

## Public verification

```bash
python scripts/validate_repository.py --strict
```

This verifies the evidence contract without requiring the private image corpus or model binaries.

## Paper build

```bash
make paper
```

## What can be reproduced publicly

- paper compilation;
- dataset/class/model identity checks;
- headline and per-class table consistency;
- paired state-transition summaries;
- PTQ selection table;
- claims and blocked-claim ledger;
- repository and release hashes.

## What requires restricted artifacts

Full reruns require the frozen image corpus, upstream model access, the exact reference/mobile checkpoints, raw predictions/logits, and the selected PTE. Their hashes are published so authorized evaluators can verify identity.

## Historical mobile configuration limitation

The selected compact state and component loss histories survive, but the exact historical optimizer, learning-rate schedule, distillation temperature, and objective coefficients are not fully preserved in human-readable form. The repository reports this gap rather than reconstructing values from curves.
