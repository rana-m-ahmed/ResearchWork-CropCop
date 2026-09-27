# Track-C R07 physical-device execution report

The sealed R07-S1 XNNPACK INT8 artifact was evaluated on a physical Google Pixel 7 (`panther`, ARM64, Android 17 / SDK 37) using ExecuTorch 1.3.1 and four requested threads. The artifact was 28,555,872 bytes with SHA-256 `2e0c54a1b5bb7c0018d0159a642e49e4f4bd79fcc1f940e215a40734a2695bb8`.

INT8 host-to-device canonical-tensor fidelity was 256/256. A matched FP32 artifact (111,741,536 bytes; SHA-256 `61556330cd9fc4725b4ff4759aef3835696b81865b2cbe030f815a4c0e8d5aec`) also achieved 256/256 canonical-tensor fidelity on the same device.

The INT8 steady-state model-only run comprised 50 untimed warm-ups and 1,000 timed forwards: mean 158.24 ms, median 175.73 ms, p95 191.91 ms and 6.320 forwards/s. The matched FP32 1,000-forward run measured mean 357.72 ms, median 366.26 ms, p95 377.75 ms and 2.795 forwards/s. The INT8 artifact is 74.45% smaller and its measured mean model-only latency is 55.77% lower on this device configuration.

INT8 raw end-to-end evaluation comprised three measured passes over RAW-256 (768 operations): mean 366.89 ms, median 234.02 ms, p95 598.41 ms and 2.726 images/s. The Android raw pipeline achieved 254/256 agreement with canonical tensors. The two retained disagreements are bounded decoder/resampler numerical differences after correcting EXIF/RGB/mean-pad/antialiased bicubic/uint8-quantization semantics; they are not used as a substitute for the canonical-tensor runtime acceptance result.

Fresh-process INT8 load trials (n=10) measured mean 247.30 ms, median 245.50 ms and p95 262.70 ms. Warm-up-transition observations (n=10) measured mean 99.39 ms. A live model run observed 109,982 KB PSS / 191,780 KB RSS, with thermal status 1. During the final raw E2E campaign, an observed peak was 332,573 KB PSS / 415,068 KB RSS; the process later reduced to approximately 214 MB resident. The final E2E campaign began at thermal status 2 and ended at status 2; neither was severe or critical.

Claims are limited to this exact artifact pair, harness revisions, requested-thread policy, and one identified Pixel 7 configuration. No product, population-level, energy, camera-workflow, or broad generalization claim is made.
