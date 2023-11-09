import ast
import tempfile

import pkg_resources
import unittest

from super_gradients.convert_recipe_to_code import convert_recipe_to_code
from pathlib import Path


class TestConvertRecipeToCode(unittest.TestCase):
    def setUp(self) -> None:
        self.recipes_dir: Path = Path(pkg_resources.resource_filename("super_gradients.recipes", ""))
        self.recipes_that_should_work = [
            "cifar10_resnet.yaml",
            "cityscapes_al_ddrnet.yaml",
            "cityscapes_ddrnet.yaml",
            "cityscapes_pplite_seg50.yaml",
            "cityscapes_pplite_seg75.yaml",
            "cityscapes_regseg48.yaml",
            "cityscapes_segformer_b0.yaml",
            "cityscapes_segformer_b1.yaml",
            "cityscapes_segformer_b2.yaml",
            "cityscapes_segformer_b3.yaml",
            "cityscapes_segformer_b4.yaml",
            "cityscapes_segformer_b5.yaml",
            "cityscapes_stdc_base.yaml",
            "cityscapes_stdc_seg50.yaml",
            "cityscapes_stdc_seg75.yaml",
            "coco2017_pose_dekr_rescoring.yaml",
            "coco2017_pose_dekr_w32_no_dc.yaml",
            "coco2017_ppyoloe_l.yaml",
            "coco2017_ppyoloe_m.yaml",
            "coco2017_ppyoloe_s.yaml",
            "coco2017_ppyoloe_x.yaml",
            "coco2017_ssd_lite_mobilenet_v2.yaml",
            "coco2017_yolo_nas_s.yaml",
            "coco2017_yolox.yaml",
            "coco_segmentation_shelfnet_lw.yaml",
            "imagenet_efficientnet.yaml",
            "imagenet_mobilenetv2.yaml",
            "imagenet_mobilenetv3_large.yaml",
            "imagenet_mobilenetv3_small.yaml",
            "imagenet_regnetY.yaml",
            "imagenet_repvgg.yaml",
            "imagenet_resnet50.yaml",
            "imagenet_vit_base.yaml",
            "imagenet_vit_large.yaml",
            "supervisely_unet.yaml",
            "user_recipe_mnist_as_external_dataset_example.yaml",
            "user_recipe_mnist_example.yaml",
            "coco2017_yolo_nas_pose_m.yaml",
            "coco2017_yolo_nas_pose_l.yaml",
            "coco2017_yolo_nas_pose_n.yaml",
            "coco2017_yolo_nas_pose_s.yaml",
        ]

        self.recipes_that_does_not_work = [
            "cityscapes_kd_base.yaml",  # KD recipe not supported
            "imagenet_resnet50_kd.yaml",  # KD recipe not supported
            "imagenet_mobilenetv3_base.yaml",  # Base recipe (not complete) for other MobileNetV3 recipes
            "cityscapes_segformer.yaml",  # Base recipe (not complete) for other SegFormer recipes
            "roboflow_ppyoloe.yaml",  # Require explicit command line arguments
            "roboflow_yolo_nas_m.yaml",  # Require explicit command line arguments
            "roboflow_yolo_nas_s.yaml",  # Require explicit command line arguments
            "roboflow_yolo_nas_s_qat.yaml",  # Require explicit command line arguments
            "roboflow_yolox.yaml",  # Require explicit command line arguments
            "variable_setup.yaml",  # Not a recipe
            "script_generate_rescoring_data_dekr_coco2017.yaml",  # Not a recipe
        ]

    def test_all_recipes_are_tested(self):
        present_recipes = set(recipe.name for recipe in self.recipes_dir.glob("*.yaml"))
        known_recipes = set(self.recipes_that_should_work + self.recipes_that_does_not_work)
        new_recipes = present_recipes - known_recipes
        removed_recipes = known_recipes - present_recipes
        if len(new_recipes):
            self.fail(f"New recipes found: {new_recipes}. Please add them to the list of recipes to test.")
        if len(removed_recipes):
            self.fail(f"Removed recipes found: {removed_recipes}. Please remove them from the list of recipes to test.")

    def test_convert_recipes_that_should_work(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for recipe in self.recipes_that_should_work:
                with self.subTest(recipe=recipe):
                    output_script_path = Path(temp_dir) / Path(recipe).name
                    convert_recipe_to_code(recipe, self.recipes_dir, output_script_path)
                    src = output_script_path.read_text()
                    try:
                        ast.parse(src)
                    except SyntaxError as e:
                        self.fail(f"Recipe {recipe} failed to convert to python script: {e}")

    def test_convert_recipes_that_are_expected_to_fail(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for recipe in self.recipes_that_does_not_work:
                with self.subTest(recipe=recipe):
                    output_script_path = Path(temp_dir) / Path(recipe).name
                    with self.assertRaises(Exception):
                        convert_recipe_to_code(recipe, self.recipes_dir, output_script_path)


if __name__ == "__main__":
    unittest.main()
