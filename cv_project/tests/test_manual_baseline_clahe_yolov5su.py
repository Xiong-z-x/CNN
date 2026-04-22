from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "cv_project" / "scripts" / "05_manual_baseline_clahe_yolov5su.py"


def load_module(module_path: Path, module_name: str):
    if not module_path.exists():
        return None
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeTensorLike:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes:
    def __init__(self):
        self.xyxy = FakeTensorLike([[4.0, 5.0, 30.0, 28.0]])
        self.conf = FakeTensorLike([0.87])
        self.cls = FakeTensorLike([6.0])


class FakeResult:
    def __init__(self):
        self.boxes = FakeBoxes()
        self.names = {6: "car"}


class FakeModel:
    def __init__(self):
        self.calls = 0

    def predict(self, source, conf, iou, device, verbose):
        self.calls += 1
        if not isinstance(source, np.ndarray):
            raise TypeError("手写版基线这里应该直接把增强后的 ndarray 喂给模型。")
        return [FakeResult()]


class TestManualBaselineScript(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module(SCRIPT_PATH, "manual_baseline_clahe_yolov5su")

    def test_script_exists(self):
        self.assertTrue(SCRIPT_PATH.exists(), "05 手写版基线脚本还没创建。")

    def test_iter_source_images_supports_file_and_directory(self):
        self.assertIsNotNone(self.module, "05 手写版基线脚本还不能导入。")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_a = root / "a.jpg"
            image_b = root / "nested" / "b.png"
            image_b.parent.mkdir(parents=True, exist_ok=True)

            dummy = np.full((16, 16, 3), 127, dtype=np.uint8)
            Image.fromarray(dummy).save(image_a)
            Image.fromarray(dummy).save(image_b)

            directory_images = self.module.iter_source_images(root)
            file_images = self.module.iter_source_images(image_a)

            self.assertEqual([path.name for path in directory_images], ["a.jpg", "b.png"])
            self.assertEqual(file_images, [image_a])

    def test_apply_clahe_keeps_uint8_and_shape(self):
        self.assertIsNotNone(self.module, "05 手写版基线脚本还不能导入。")
        image = np.full((24, 24, 3), 80, dtype=np.uint8)
        enhanced = self.module.apply_clahe_bgr(
            image=image,
            clip_limit=2.0,
            tile_grid_size=(8, 8),
        )
        self.assertEqual(enhanced.shape, image.shape)
        self.assertEqual(enhanced.dtype, np.uint8)

    def test_run_manual_baseline_with_fake_model_writes_outputs(self):
        self.assertIsNotNone(self.module, "05 手写版基线脚本还不能导入。")
        fake_model = FakeModel()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "input_images"
            output_root = root / "manual_outputs"
            source_root.mkdir(parents=True, exist_ok=True)

            image_path = source_root / "demo.jpg"
            image = np.full((40, 48, 3), 90, dtype=np.uint8)
            Image.fromarray(image).save(image_path)

            summary = self.module.run_manual_baseline(
                source=source_root,
                output_root=output_root,
                weights="fake.pt",
                device="cpu",
                conf=0.25,
                iou=0.7,
                clip_limit=2.0,
                tile_grid_size=(8, 8),
                model=fake_model,
            )

            self.assertEqual(fake_model.calls, 1)
            self.assertEqual(summary["num_images"], 1)
            self.assertEqual(summary["total_detections"], 1)
            self.assertTrue((output_root / "enhanced" / "demo.jpg").exists())
            self.assertTrue((output_root / "predictions" / "demo.jpg").exists())
            self.assertTrue((output_root / "manual_baseline_summary.json").exists())

            manifest = json.loads((output_root / "manual_baseline_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["num_images"], 1)
            self.assertEqual(manifest["total_detections"], 1)
            self.assertEqual(manifest["images"][0]["detections"][0]["class_name"], "car")


if __name__ == "__main__":
    unittest.main()
