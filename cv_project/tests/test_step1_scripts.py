from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "cv_project" / "scripts"


def load_module(module_path: Path, module_name: str):
    if not module_path.exists():
        return None
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestVocPrepScript(unittest.TestCase):
    def setUp(self) -> None:
        self.script_path = SCRIPTS_DIR / "01_voc_download_and_yolo.py"
        self.module = load_module(self.script_path, "voc_download_and_yolo")

    def test_script_exists(self):
        self.assertTrue(self.script_path.exists(), "01 脚本还没落地到 scripts 目录。")

    def test_bbox_conversion_clips_values_into_unit_interval(self):
        self.assertIsNotNone(self.module, "01 脚本还不能导入。")
        self.assertTrue(
            hasattr(self.module, "voc_box_to_yolo"),
            "01 脚本里缺少 voc_box_to_yolo() 这个关键函数。",
        )
        result = self.module.voc_box_to_yolo(
            image_width=100,
            image_height=80,
            xmin=-5,
            ymin=-3,
            xmax=140,
            ymax=120,
        )
        self.assertEqual(len(result), 4)
        for value in result:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_class_mapping_is_fixed_and_rejects_unknown_name(self):
        self.assertIsNotNone(self.module, "01 脚本还不能导入。")
        self.assertTrue(
            hasattr(self.module, "class_name_to_id"),
            "01 脚本里缺少 class_name_to_id() 这个关键函数。",
        )
        self.assertEqual(self.module.class_name_to_id("person"), 14)
        with self.assertRaises(ValueError):
            self.module.class_name_to_id("not_a_voc_class")

    def test_find_local_archives_recognizes_both_required_voc2007_zips(self):
        import tempfile

        self.assertIsNotNone(self.module, "01 脚本还不能导入。")
        self.assertTrue(
            hasattr(self.module, "find_local_voc_archives"),
            "01 脚本里缺少 find_local_voc_archives()。",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "VOCtrainval_06-Nov-2007.zip").write_bytes(b"fake")
            (root / "VOCtest_06-Nov-2007.zip").write_bytes(b"fake")
            archives = self.module.find_local_voc_archives(root, "2007")
            self.assertEqual(
                sorted(path.name for path in archives),
                ["VOCtest_06-Nov-2007.zip", "VOCtrainval_06-Nov-2007.zip"],
            )


class TestDegradationScript(unittest.TestCase):
    def setUp(self) -> None:
        self.script_path = SCRIPTS_DIR / "02_degradation.py"
        self.module = load_module(self.script_path, "degradation")

    def test_script_exists(self):
        self.assertTrue(self.script_path.exists(), "02 脚本还没落地到 scripts 目录。")

    def test_degradation_keeps_uint8_shape_and_darkens_image(self):
        self.assertIsNotNone(self.module, "02 脚本还不能导入。")
        self.assertTrue(
            hasattr(self.module, "apply_low_light_degradation"),
            "02 脚本里缺少 apply_low_light_degradation() 这个关键函数。",
        )
        image = np.full((8, 8, 3), 180, dtype=np.uint8)
        degraded = self.module.apply_low_light_degradation(
            image=image,
            gamma=2.4,
            noise_std=0.0,
            rng=np.random.default_rng(0),
        )
        self.assertEqual(degraded.shape, image.shape)
        self.assertEqual(degraded.dtype, np.uint8)
        self.assertLess(float(degraded.mean()), float(image.mean()))

    def test_gamma_one_and_zero_noise_keeps_image_unchanged(self):
        self.assertIsNotNone(self.module, "02 脚本还不能导入。")
        image = np.array(
            [
                [[0, 16, 32], [64, 96, 128]],
                [[160, 192, 224], [255, 128, 32]],
            ],
            dtype=np.uint8,
        )
        degraded = self.module.apply_low_light_degradation(
            image=image,
            gamma=1.0,
            noise_std=0.0,
            rng=np.random.default_rng(123),
        )
        self.assertTrue(np.array_equal(degraded, image))


class TestRemainingPipelineFiles(unittest.TestCase):
    def test_clahe_script_exists_and_parses_grid(self):
        script_path = SCRIPTS_DIR / "03_clahe_process.py"
        module = load_module(script_path, "clahe_process")
        self.assertTrue(script_path.exists(), "03 脚本还没落地到 scripts 目录。")
        self.assertIsNotNone(module, "03 脚本还不能导入。")
        self.assertTrue(
            hasattr(module, "parse_tile_grid_size"),
            "03 脚本里缺少 parse_tile_grid_size()。",
        )
        self.assertEqual(module.parse_tile_grid_size("8x8"), (8, 8))
        with self.assertRaises(ValueError):
            module.parse_tile_grid_size("8")

    def test_zerodce_script_exists_and_can_convert_tensor_to_image(self):
        import torch

        script_path = SCRIPTS_DIR / "04_zerodce_process.py"
        module = load_module(script_path, "zerodce_process")
        self.assertTrue(script_path.exists(), "04 脚本还没落地到 scripts 目录。")
        self.assertIsNotNone(module, "04 脚本还不能导入。")
        self.assertTrue(
            hasattr(module, "tensor_to_bgr_image"),
            "04 脚本里缺少 tensor_to_bgr_image()。",
        )

        rgb_tensor = torch.tensor(
            [
                [[0.0, 1.0], [0.5, 0.25]],
                [[0.25, 0.5], [1.0, 0.0]],
                [[1.0, 0.0], [0.25, 0.5]],
            ],
            dtype=torch.float32,
        )
        bgr_image = module.tensor_to_bgr_image(rgb_tensor)
        self.assertEqual(bgr_image.shape, (2, 2, 3))
        self.assertEqual(bgr_image.dtype, np.uint8)
        self.assertEqual(int(bgr_image[0, 0, 0]), 255)
        self.assertEqual(int(bgr_image[0, 1, 2]), 255)

    def test_zerodce_forward_keeps_original_spatial_size_for_odd_shapes(self):
        import torch

        script_path = SCRIPTS_DIR / "04_zerodce_process.py"
        module = load_module(script_path, "zerodce_process_odd_shape")
        self.assertIsNotNone(module, "04 脚本还不能导入。")
        self.assertTrue(
            hasattr(module, "ZeroDCEPPInferenceNet"),
            "04 脚本里缺少 ZeroDCEPPInferenceNet 这个推理网络。",
        )

        model = module.ZeroDCEPPInferenceNet(scale_factor=12)
        image = torch.rand(1, 3, 375, 500, dtype=torch.float32)
        enhanced, curves = model(image)

        self.assertEqual(tuple(enhanced.shape), (1, 3, 375, 500))
        self.assertEqual(tuple(curves.shape), (1, 3, 375, 500))

    def test_eval_script_and_yaml_configs_exist(self):
        eval_path = PROJECT_ROOT / "cv_project" / "eval_pipeline.py"
        module = load_module(eval_path, "eval_pipeline")
        self.assertTrue(eval_path.exists(), "eval_pipeline.py 还没创建。")
        self.assertIsNotNone(module, "eval_pipeline.py 还不能导入。")
        self.assertTrue(
            hasattr(module, "to_yaml_path"),
            "eval_pipeline.py 里缺少 to_yaml_path()。",
        )
        self.assertEqual(module.to_yaml_path(Path(r"cv_project\datasets\VOC_Dark")), "cv_project/datasets/VOC_Dark")

        config_dir = PROJECT_ROOT / "cv_project" / "config"
        for yaml_name in ("dark.yaml", "clahe.yaml", "zdcepp.yaml"):
            yaml_path = config_dir / yaml_name
            self.assertTrue(yaml_path.exists(), f"{yaml_name} 还没创建。")
            content = yaml_path.read_text(encoding="utf-8")
            self.assertNotIn("\\", content, f"{yaml_name} 里不该出现反斜杠路径。")

    def test_autodl_launcher_exists_and_mentions_core_steps(self):
        launcher_path = PROJECT_ROOT / "cv_project" / "run_autodl.sh"
        self.assertTrue(launcher_path.exists(), "AutoDL 一键脚本还没创建。")
        content = launcher_path.read_text(encoding="utf-8")
        self.assertIn("01_voc_download_and_yolo.py", content)
        self.assertIn("02_degradation.py", content)
        self.assertIn("03_clahe_process.py", content)
        self.assertIn("04_zerodce_process.py", content)
        self.assertIn("eval_pipeline.py", content)


if __name__ == "__main__":
    unittest.main()
