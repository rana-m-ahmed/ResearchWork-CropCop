from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "journal_extension" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RUNTIME_STACK_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("numpy", "PIL", "torch")
)
RUNTIME_STACK_REASON = "exact Track-A runtime stack (numpy, Pillow, torch) is unavailable in this lightweight CPU-safe test environment"


@unittest.skipUnless(RUNTIME_STACK_AVAILABLE, RUNTIME_STACK_REASON)
class TrackAV12RuntimeAnalysisTests(unittest.TestCase):
    def test_locked_gaussian_noise_is_deterministic(self):
        import numpy as np
        from PIL import Image
        from cropcop_je.tracka_v12_posttraining import apply_locked_corruption

        array = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
        image = Image.fromarray(array, mode="RGB")
        first = np.asarray(apply_locked_corruption(image, stable_row_id="row-1", corruption="gaussian_noise_uint8", severity="2"))
        second = np.asarray(apply_locked_corruption(image, stable_row_id="row-1", corruption="gaussian_noise_uint8", severity="2"))
        third = np.asarray(apply_locked_corruption(image, stable_row_id="row-2", corruption="gaussian_noise_uint8", severity="2"))
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, third))

    def test_all_five_corruptions_preserve_rgb_image(self):
        import numpy as np
        from PIL import Image
        from cropcop_je.tracka_v12_analysis import ROBUSTNESS_CORRUPTIONS
        from cropcop_je.tracka_v12_posttraining import apply_locked_corruption

        image = Image.fromarray(np.full((32, 20, 3), 127, dtype=np.uint8), mode="RGB")
        for corruption in ROBUSTNESS_CORRUPTIONS:
            out = apply_locked_corruption(image, stable_row_id="row", corruption=corruption, severity="1")
            self.assertEqual(out.mode, "RGB")
            self.assertEqual(out.size, image.size)

    def test_cnn_target_resolution_and_gradcampp_shape(self):
        import torch
        from cropcop_je.tracka_v12_xai import gradcampp, module_by_path, resolve_cnn_target_layer

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 4, 3, padding=1)
                self.conv2 = torch.nn.Conv2d(4, 5, 3, padding=1)
                self.pool = torch.nn.AdaptiveAvgPool2d(1)
                self.classifier = torch.nn.Linear(5, 3)

            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                return self.classifier(self.pool(x).flatten(1))

        model = Tiny().eval()
        device = torch.device("cpu")
        path = resolve_cnn_target_layer(model, device)
        self.assertEqual(path, "conv2")
        result = gradcampp(model, torch.ones((1, 3, 256, 256)), module_by_path(model, path))
        self.assertEqual(tuple(result["heatmap"].shape), (256, 256))
        self.assertFalse(result["nonfinite"])

    def test_classifier_randomization_is_exactly_restorable(self):
        import torch
        from cropcop_je.tracka_v12_xai import deterministic_randomize_classifier, restore_classifier

        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 3))
        model.classifier = model[1]
        before = {key: value.detach().clone() for key, value in model.classifier.state_dict().items()}
        head, state = deterministic_randomize_classifier(model, seed=123)
        changed = any(not torch.equal(before[key], head.state_dict()[key]) for key in before)
        self.assertTrue(changed)
        restore_classifier(head, state)
        for key, value in before.items():
            self.assertTrue(torch.equal(value, head.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()
