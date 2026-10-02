# Track-C R07 physical-device execution report

Terminal experimental outcome: `PASS_DEVICE_FP32_INT8` with bounded, device-specific claims.

The sealed R07-S1 XNNPACK INT8 artifact was evaluated on a physical Google Pixel 7 (`panther`, ARM64, Android 17 / SDK 37) using ExecuTorch 1.3.1 and four requested threads. The artifact was 28,555,872 bytes with SHA-256 `2e0c54a1b5bb7c0018d0159a642e49e4f4bd79fcc1f940e215a40734a2695bb8`.

The INT8 tensor/latency/E2E campaign used the 29,924,924-byte APK with SHA-256 `a2bdd74088ff893447c869470c9fa9eabd65d79791042b87e985e340a541c67d`. The later FP32-capable benchmark APK used for matched FP32 execution was 29,924,924 bytes with SHA-256 `b6f0c4533aa1bc8b13a467d7d7ac83cd978f3928222da22ca7df5d2d15954d26`; its material harness addition was explicit selection and verification of the separately locked FP32 artifact.

Canonical-tensor host-to-device fidelity was 256/256 for INT8 and 256/256 for the matched FP32 artifact (111,741,536 bytes; SHA-256 `61556330cd9fc4725b4ff4759aef3835696b81865b2cbe030f815a4c0e8d5aec`). This is the primary runtime-fidelity acceptance result.

The post-warm-up INT8 model-only campaign comprised 50 untimed warm-ups and 1,000 timed forwards: mean 158.24 ms, median 175.73 ms, p95 191.91 ms and 6.320 forwards/s. The FP32 1,000-forward campaign measured mean 357.72 ms, median 366.26 ms, p95 377.75 ms and 2.795 forwards/s. The INT8 artifact is 74.45% smaller; observed mean, median and p95 model-only latency were respectively 55.76%, 52.02% and 49.20% lower than FP32 on this device/protocol.

These 1,000-forward campaigns must not be described as thermally stationary. INT8 mean latency rose from 102.94 ms in the first 100 timed forwards to 185.41 ms in the final 100; FP32 rose from 284.36 ms to 361.83 ms. The locked protocol retains all 1,000 samples, so no post-hoc trimming is used.

INT8 raw end-to-end evaluation comprised three measured passes over RAW-256 (768 operations): mean 366.89 ms, median 234.02 ms, p95 598.41 ms and 2.726 images/s. The distribution is heavy-tailed (p99 2393.07 ms; maximum 4257.32 ms; 15/768 samples exceeded 1 s), so median and p95 are the preferred paper-facing summaries. Pass medians were 228.30, 234.34 and 234.47 ms.

The Android raw pipeline achieved 254/256 top-1 agreement with the canonical tensors. The two retained disagreements were at sample indices 11 and 255. They do not alter the canonical-tensor runtime acceptance result and must be reported separately as preprocessing-pipeline fidelity rather than runtime fidelity.

Fresh-process INT8 activity-start/load trials (the committed field is `am_total_ms`) measured mean 249.40 ms, median 242.00 ms and p95 284.60 ms. This value includes fresh-process/activity startup plus artifact-loading work and is not an isolated `Module.load` microbenchmark. Warm-up-transition observations (n=10) measured mean 99.39 ms.

The public package contains one live process observation of 109,982 KB PSS / 191,780 KB RSS with thermal status 1. A single process snapshot is not sufficient for a campaign-level CPU-utilization claim. Higher E2E peak-memory, final thermal-status, and app-private-directory footprint values were operator observations recorded in the execution narrative but do not have their own raw public transcript in the current evidence package; they are excluded from primary machine-recomputed headline results.

Energy was not measured because no defensible external or OS-level energy instrument was available. No energy claim is made.

Claims are limited to the exact artifact pair, APK hashes, requested-thread policy, ExecuTorch/XNNPACK runtime, and this one identified Pixel 7 configuration. Track C does not establish Android-wide performance, field generalization, production readiness, NPU/GPU behavior, or energy efficiency.
