# Track-C R07 two-device audit

Audit verdict: the frozen Track-C INT8 execution was attempted on both the
primary Google Pixel 7 and the secondary Xiaomi POCO M3. This is a valid
two-device compatibility audit, but it is **not** a completed head-to-head
performance comparison: the POCO cannot execute the first forward under the
frozen runtime/artifact pair.

## Same frozen inputs

Both devices used the sealed R07-S1 XNNPACK INT8 artifact
`2e0c54a1b5bb7c0018d0159a642e49e4f4bd79fcc1f940e215a40734a2695bb8`
(28,555,872 bytes), the same locked input archive, the TENSOR-256 manifest
`a2be311ed3b182c478a0482d78ee4ca05cc0575cdf16b16d9ee79355762c073a`, the
RAW-256 manifest
`73c2beed647fd874be9a4187a872d33455a4a5dacf1e2b1180582a74b8a0a1ac`, and
the input lock
`1380b926aaf7dd144bec50c7194dd6b3a31428ae1bb0da7bba121ce595f697c8`.

| Gate | Pixel 7 | POCO M3 |
| --- | --- | --- |
| ARM64/runtime preflight | Pass | Pass (`arm64-v8a`, ExecuTorch 1.3.1) |
| Frozen INT8 artifact identity | Pass | Pass |
| Canonical TENSOR-256 forward | 256/256 host-device top-1 agreement | Native `SIGILL` on first forward; 0 rows |
| INT8 model-only campaign | 50 untimed warm-ups + 1,000 forwards | Not runnable |
| INT8 RAW end-to-end campaign | 768 operations | Not runnable |
| Valid performance comparison | Yes, within Pixel 7's FP32/INT8 protocol | No cross-device result exists |

The primary campaign is independently validated by
`analysis/validate_trackc_evidence.py`. Its measured rows and committed
closure are limited to the identified Pixel 7 configuration. The secondary
device certificate is independently validated by
`analysis/validate_trackc_secondary_device.py`.

## Live secondary-device recheck

On 2026-09-27, the attached POCO M3 (`M2010J19CG`, Android 12) was rechecked
with the exact APK and staged inputs. Direct app-private hashing confirmed the
artifact, both manifests, and input lock above; 256 tensor and 256 RAW files
were present. `load` succeeds. Launching `tensor_fidelity` with the explicit
registered activity component reproduced Android exit reason `APP
CRASH(NATIVE)`, status 4, with `SIGILL`/`ILL_ILLOPC` in
`libexecutorch_jni.so`. Its output CSV remained zero bytes.

This confirms the earlier POCO certificate rather than converting it into a
performance result. No artifact, quantization, model, input, or runtime was
changed.

## Remaining boundary

There is no unresolved Pixel 7 evidence gate. A successful POCO head-to-head
benchmark requires a separately versioned, device-compatible runtime/build or
a different compatible secondary device. Either would be a new experiment and
must not overwrite or be represented as the frozen Track-C result.
