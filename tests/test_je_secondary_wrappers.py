from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "8904b100d223e4319776199c87ab397db23600ce"
WRAPPERS = {
    "K1": ROOT / "journal_extension/kaggle/secondary_K1_mechanism.ipynb",
    "K2": ROOT / "journal_extension/kaggle/secondary_K2_context.ipynb",
    "K3": ROOT / "journal_extension/kaggle/secondary_K3_qualification.ipynb",
}


def code_text(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    if nb.get("nbformat") != 4 or int(nb.get("nbformat_minor", -1)) < 5:
        raise AssertionError(f"invalid notebook format: {path}")
    parts = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        text = src if isinstance(src, str) else "".join(src)
        compile(text, f"{path.name}:cell-{i}", "exec")
        parts.append(text)
    return "\n".join(parts)


class SecondaryWrapperTests(unittest.TestCase):
    def test_manifest_hashes_exact_wrapper_bytes(self):
        manifest = json.loads((ROOT / "journal_extension/kaggle/secondary_wrapper_manifest.json").read_text())
        self.assertTrue(manifest["wrapper_only"])
        self.assertFalse(manifest["scientific_source_changed"])
        self.assertEqual(manifest["execution_source_sha"], SOURCE)
        self.assertEqual(set(manifest["notebooks"]), {str(p.relative_to(ROOT)) for p in WRAPPERS.values()})
        for path in WRAPPERS.values():
            row = manifest["notebooks"][str(path.relative_to(ROOT))]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"])
            self.assertEqual(path.stat().st_size, row["bytes"])

    def test_all_wrappers_compile_and_bind_only_qualified_source(self):
        for path in WRAPPERS.values():
            text = code_text(path)
            self.assertIn(f'AUTHORIZED_SOURCE_SHA = "{SOURCE}"', text)
            self.assertIn('SOURCE_BRANCH = "je-tracka-secondary-core-20260910"', text)
            self.assertNotIn('AUTHORIZED_SOURCE_SHA = "f171309', text)
            self.assertNotIn("DS-V1-TEST", text)
            self.assertNotIn("github_pat_", text)
            self.assertNotIn("ghp_", text)

    def test_batch_gate_and_global_clock_precede_credentials_and_clone(self):
        for path in WRAPPERS.values():
            text = code_text(path)
            self.assertLess(text.index("CROPCOP_NOTEBOOK_STARTED_MONOTONIC"), text.index('"git", "clone"'))
            self.assertLess(text.index("KAGGLE_KERNEL_RUN_TYPE"), text.index('get_secret("CROPCOP_GITHUB_TOKEN")'))

    def test_phase_specific_kaggle_api_credentials_are_least_privilege(self):
        text = code_text(WRAPPERS["K3"])
        self.assertIn('KAGGLE_API_PHASES = {', text)
        self.assertIn('"secondary-g1", "secondary-g2", "secondary-scientific"', text)
        self.assertIn('if PHASE in KAGGLE_API_PHASES:', text)
        self.assertLess(text.index('if PHASE in KAGGLE_API_PHASES:'), text.index('kaggle_username = get_secret("KAGGLE_USERNAME")'))

    def test_attached_evidence_is_identity_bound_not_filename_only(self):
        text = code_text(WRAPPERS["K3"])
        self.assertIn("def find_bound_json", text)
        self.assertIn('source_sha=AUTHORIZED_SOURCE_SHA, status="PASS"', text)
        self.assertIn("PRINCIPAL_G1_SEAL_SHA256", text)
        self.assertIn("g1_seal_sha256", text)

    def test_exact_account_and_envelope_mapping(self):
        k1 = code_text(WRAPPERS["K1"])
        k2 = code_text(WRAPPERS["K2"])
        k3 = code_text(WRAPPERS["K3"])
        self.assertIn('EXPECTED_KAGGLE_USERNAME = "ranaabdulrehmannn"', k1)
        self.assertIn('SCIENCE_ENVELOPE = "mechanism"', k1)
        self.assertIn('EXPECTED_KAGGLE_USERNAME = "sabahatabbas"', k2)
        self.assertIn('SCIENCE_ENVELOPE = "context"', k2)
        self.assertIn('EXPECTED_KAGGLE_USERNAME = "ranamuhammadahmed6"', k3)
        self.assertIn('LANE = "K3"', k3)

    def test_durable_templates_keep_literal_source_bound_run_id_placeholder(self):
        k1 = code_text(WRAPPERS["K1"])
        k2 = code_text(WRAPPERS["K2"])
        k3 = code_text(WRAPPERS["K3"])
        self.assertIn('f"{EXPECTED_KAGGLE_USERNAME}/{{run_id_lower}}"', k1)
        self.assertIn('f"{EXPECTED_KAGGLE_USERNAME}/{{run_id_lower}}"', k2)
        self.assertIn('f"{EXPECTED_KAGGLE_USERNAME}/sec-{{run_id_lower}}"', k3)

    def test_k3_gate_order_is_forward_only(self):
        k3 = code_text(WRAPPERS["K3"])
        for phase in ("smoke-write", "smoke-restore", "dual-gpu-smoke", "secondary-g1", "secondary-g2"):
            self.assertIn(phase, k3)
        self.assertIn("run_secondary_g1.py", k3)
        self.assertIn("run_secondary_g2_envelope.py", k3)
        self.assertNotIn("run_envelope.py", k3)


if __name__ == "__main__":
    unittest.main()
