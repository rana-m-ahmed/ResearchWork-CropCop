# Track-A Strengthening Amendment v1 — Lock Integrity Hotfix

**Date:** 2026-09-12  
**Status:** `PASS — PRE-EXECUTION METADATA CORRECTION`  
**Affected artifact:** `AMENDMENT_LOCK_MANIFEST.json`

## Defect

The first pre-execution lock commit recorded a malformed/truncated SHA-256 string for `analysis_protocol.json`. The content file itself was unchanged and valid; only the manifest entry was wrong.

## Correction

The correct SHA-256 for `analysis_protocol.json` is:

`f2d7605c98861fc61d8dfe8678b46feb7ee17082d50caba65e480ec2442b6200`

All other amendment content hashes remain unchanged.

## Scientific-safety statement

This correction occurred before any amendment scientific run, amendment checkpoint, robustness output, classwise result, efficiency result, or external prediction was produced or inspected. It therefore corrects lock metadata only and does not alter a scientific outcome, hypothesis, seed, model family, objective, metric, corruption severity, claim boundary, or stopping rule.

The original Track-A Stage-02/03A evidence remains unchanged. `DS-V1-TEST-CONSUMED` remains unopened for this amendment.

## Gate

A1-G0 may be evaluated only from the corrected manifest and this hotfix record. The superseded malformed manifest state must not be cited as the amendment lock.
