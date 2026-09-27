# Track-C manuscript handoff

Use this file as the paper-facing bridge from the closed Track-C evidence. It intentionally omits implementation bureaucracy.

## Methods facts

- Final scientific family: R07 / TorchVision ConvNeXt-Tiny Context.
- Deployment representative: R07-CNXTT-CONTEXT-S1.
- Device: Google Pixel 7 (`panther`), ARM64, Android 17 / SDK 37.
- Runtime: ExecuTorch Android 1.3.1, XNNPACK CPU, four requested threads.
- Canonical device-fidelity surface: 256 frozen validation tensors.
- Raw-pipeline surface: the paired RAW-256 validation manifest.
- Model-only protocol: 50 untimed warm-ups + 1,000 timed forwards per artifact.
- E2E protocol: three passes over RAW-256 = 768 operations.
- Energy: not measured; no energy claim.

## Results table values

| Metric | INT8 R07-S1 | FP32 R07-S1 |
| --- | ---: | ---: |
| Artifact size | 28,555,872 B (27.23 MiB) | 111,741,536 B (106.56 MiB) |
| Canonical host-device top-1 agreement | 256/256 (100%) | 256/256 (100%) |
| Model-only mean latency | 158.24 ms | 357.72 ms |
| Model-only median latency | 175.73 ms | 366.26 ms |
| Model-only p95 latency | 191.91 ms | 377.75 ms |
| Sequential throughput | 6.320/s | 2.795/s |

INT8 artifact reduction: 74.44%. Observed INT8 latency reduction versus FP32: 55.76% by mean, 52.02% by median, 49.20% by p95.

INT8 raw E2E: median 234.02 ms, p95 598.41 ms, mean 366.89 ms, p99 2393.07 ms, max 4257.32 ms, n=768. Because the distribution is heavy-tailed, use median + p95 as the main paper summaries.

RAW-vs-canonical preprocessing fidelity: 254/256 (99.21875%). Keep this separate from canonical runtime fidelity.

## Wording to avoid

- `steady-state latency` for the full 1,000-forward campaigns.
- `four XNNPACK threads were verified`; only four threads were requested.
- robust CPU-utilization claims from the single process snapshot.
- product package-size claims from the app-private research directory.
- Android-wide or universal real-time claims.
- causal attribution of all latency difference to quantization alone.

## Suggested bounded result sentence

On a Google Pixel 7 using ExecuTorch 1.3.1/XNNPACK CPU, both the frozen FP32 and prospectively locked INT8 R07-S1 artifacts achieved 256/256 canonical-tensor host-device top-1 agreement. The INT8 artifact was 74.44% smaller and showed lower observed model-only latency under the matched device protocol (median 175.73 ms versus 366.26 ms; p95 191.91 ms versus 377.75 ms).
