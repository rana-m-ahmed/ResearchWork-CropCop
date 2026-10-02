# Track-C final hostile audit

Audit basis: remote branch `research/track-c-r07-lean-v1`, pre-audit evidence head `414d80f8a16f665e205028852c75cd3fd15ef12f`, descending from authority base `d8d627620acfa203a8bd9ba6bf87fb85395faa25` with no behind commits at the audit boundary.

## Verdict

`PASS_DEVICE_FP32_INT8` for the physical-device experiment, with bounded engineering claims and explicit evidence-grade limitations. No raw measurement file was modified by this audit.

## Authority and lineage

- Track-A-selected family remains R07 / ConvNeXt-Tiny Context.
- Deployment representative remains R07-CNXTT-CONTEXT-S1.
- Checkpoint SHA-256 remains `dc7fea2e8db91bf1fc023cb5e792b23b67659edec22e10a7c1d46b4010db3974`.
- V1 test was not opened for Track C.
- Track-B protected outcomes were not used for Track-C selection.
- INT8 and FP32 runtime artifacts remain hash-bound to the Prompt-1 foundation.

## Independently recomputed primary results

| Endpoint | INT8 | FP32 | Interpretation |
| --- | ---: | ---: | --- |
| Artifact bytes | 28,555,872 | 111,741,536 | INT8 74.44% smaller (3.91x size ratio) |
| Canonical host-device fidelity | 256/256 | 256/256 | Primary runtime-fidelity PASS |
| Model-only mean | 158.24 ms | 357.72 ms | Observed INT8 mean 55.76% lower |
| Model-only median | 175.73 ms | 366.26 ms | Observed INT8 median 52.02% lower |
| Model-only p95 | 191.91 ms | 377.75 ms | Observed INT8 p95 49.20% lower |
| Model-only throughput | 6.320/s | 2.795/s | Sequential reciprocal-of-mean summary |

INT8 RAW-256 E2E: n=768, mean 366.89 ms, median 234.02 ms, p95 598.41 ms, p99 2393.07 ms, max 4257.32 ms. The distribution is strongly heavy-tailed; 15/768 samples exceeded 1 s and 8 exceeded 3 s. Three pass medians were 228.30, 234.34 and 234.47 ms.

RAW-vs-canonical top-1 agreement is 254/256 (99.21875%); mismatch indices are 11 and 255. Canonical runtime fidelity remains 256/256 for both artifacts.

## Important interpretation findings

1. The earlier report wording `steady-state` was too strong. Both model-only campaigns show substantial within-run drift after the predeclared 50 warm-ups. All 1,000 samples remain valid and are retained; no post-hoc deletion is authorized.
2. The INT8 first-100 mean was 102.94 ms and final-100 mean was 185.41 ms (+80.11%). FP32 moved from 284.36 to 361.83 ms (+27.24%). This is evidence of non-stationary device behavior, not a reason to rerun or trim results.
3. E2E latency is not well represented by the mean alone. The paper should lead with median and p95 and disclose the long tail.
4. The committed `load_trials_int8.csv` contains `am_total_ms`; therefore the endpoint is fresh-process/activity-start plus loading, not isolated model-load time.
5. The public CPU/memory transcript supports one live PSS/RSS/thermal snapshot. It does not support a robust average CPU-utilization claim.
6. The higher reported E2E peak PSS/RSS, start/end thermal status, and app-private storage footprint are not backed by separate raw public transcripts. They may remain contextual observations but should not be primary machine-recomputed table entries unless the original capture is later published.
7. FP32 and INT8 were measured on the same physical device/runtime/thread policy but in separate campaign revisions/APKs. Their latency difference is a matched observational comparison, not a universal causal quantization speedup claim.

## Reproducibility repair made by this audit

The independent analyzer previously referenced staging-era filenames (`model_latency.csv`, `end_to_end.csv`, etc.) rather than the committed public filenames (`model_latency_int8.csv`, `end_to_end_int8.csv`, etc.). The analyzer is repaired to consume the public package directly. The validator now independently recomputes `analysis_summary.json` from raw CSVs and compares every numerical field before passing.

## Publication-safe conclusion

Track C successfully demonstrates exact-artifact execution of both the frozen R07-S1 FP32 and prospectively locked INT8 representations on one identified Pixel 7 configuration, with perfect canonical-tensor host-device top-1 agreement for both artifacts and substantial observed reductions in artifact size and model-only latency for INT8. The raw-image pipeline remains a separate 254/256 preprocessing-fidelity result. Claims must remain device-specific and must not imply Android-wide, field-generalization, production-readiness, energy, NPU/GPU, or robust CPU-utilization conclusions.
