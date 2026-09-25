# Track-C R07 pre-device readiness

Status: `BLOCKED_TRACKC_V1_VAL_MATERIALIZATION`

This is a source-data materialization blocker, not a device blocker and not a
failure of the frozen R07-S1 state. No physical performance, memory, CPU, or
host-to-Android fidelity claim has been produced.

## Verified foundation

The exact R07-S1 checkpoint at
`C:\Users\ranam\Downloads\selected.g00000170.dc7fea2e8db91bf1.ckpt` was
observed at 335,202,763 bytes with SHA-256:

```text
dc7fea2e8db91bf1fc023cb5e792b23b67659edec22e10a7c1d46b4010db3974
```

The imported Prompt-1 handoff is `PASS` with F0–F7 complete. Its public-safe
evidence binds the following already-executed restricted work:

| Item | Identity |
| --- | --- |
| Source replay | 16,368 DS-V1-VAL rows; frozen metrics reproduced within `1e-6` |
| FP32 ExecuTorch/XNNPACK artifact | `61556330cd9fc4725b4ff4759aef3835696b81865b2cbe030f815a4c0e8d5aec`, 111,741,536 bytes |
| Selected hybrid PTQ/XNNPACK artifact | `2e0c54a1b5bb7c0018d0159a642e49e4f4bd79fcc1f940e215a40734a2695bb8`, 28,555,872 bytes |
| PTQ validation top-1 agreement | `0.9957844574780058` against FP32 source predictions |
| Runtime | ExecuTorch Android `1.3.1`, XNNPACK CPU, four requested threads |

The exact scientific bindings are retained in
`shared_foundation/TRACKBC_R07_AUTHORITY_BINDING_v1.json`,
`R07_S1_VALIDATION_REPLAY_REPORT_v1.json`, and
`PROMPT1_FOUNDATION_HANDOFF.json`.

## Harness readiness

The research-only Android harness was built locally without including any
restricted model or image bytes in Git:

```text
APK: app-debug.apk
bytes: 27883460
SHA-256: bc8b965034f09b9392668c9ab7919ae57aa497174848269a4c4749e1ed1b3c4e
host unit tests: PASS
```

It recognizes only the frozen modes `tensor_fidelity`, `raw_fidelity`,
`load`, `warmup_transition`, `model_latency`, `end_to_end`, and
`system_snapshot`. It verifies the selected PTQ artifact hash before load and
refuses any input-dependent mode while manifests are absent.

## Blocking evidence

The canonical image-backed Final-V1 authority is the private Kaggle dataset:

```text
ranamuhammadahmed6/cropcop-finalized-v8-11-2026-1
```

Its expected manifest SHA-256 is:

```text
bdb82211ccc2059153724eea178a1680893a6b38ecc243fae484baa91dbf68e2
```

No hash-valid local Final-V1 materialization is available. A read-only request
to `https://www.kaggle.com/api/v1/datasets/view/ranamuhammadahmed6/cropcop-finalized-v8-11-2026-1`
returned HTTP 403 (`datasets.get` denied) without Kaggle credentials. The
available local `final_manifest.csv` files are test fixtures only.

Therefore the following required pre-device artifacts cannot yet be generated:

- `DS-DEVICE-RAW-256-R07`
- `DS-DEVICE-TENSOR-256-R07`
- their sealed manifests and hashes
- raw-pipeline and same-artifact Android fidelity inputs
- the final analysis lock

## Resume procedure

Provide authenticated read access to the canonical Final-V1 Kaggle dataset, or
a local image-backed Final-V1 root whose `audit/final_manifest.csv` and
`audit/class_to_idx.json` match the locked SHA-256 identities. Then:

1. validate that source without opening `DS-V1-TEST-CONSUMED`;
2. deterministically generate RAW-256 and TENSOR-256 from DS-V1-VAL;
3. seal the device and analysis locks;
4. provision the already hash-bound runtime artifact into the harness;
5. only then assess physical-device availability and execute the frozen device protocol.
