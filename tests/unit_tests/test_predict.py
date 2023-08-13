import os
import unittest
import tempfile

from super_gradients.common.object_names import Models
from super_gradients.training import models


class TestModelPredict(unittest.TestCase):
    def setUp(self) -> None:
        rootdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.images = [
            os.path.join(rootdir, "documentation", "source", "images", "examples", "countryside.jpg"),
            os.path.join(rootdir, "documentation", "source", "images", "examples", "street_busy.jpg"),
            "https://deci-datasets-research.s3.amazonaws.com/image_samples/beatles-abbeyroad.jpg",
        ]

    def test_classification_models(self):
        with tempfile.TemporaryDirectory() as tmp_dirname:
            for model_name in {Models.RESNET18, Models.EFFICIENTNET_B0, Models.MOBILENET_V2, Models.REGNETY200}:
                model = models.get(model_name, pretrained_weights="imagenet")

                predictions = model.predict(self.images)
                predictions.show()
                predictions.save(output_folder=tmp_dirname)

    def test_pose_estimation_models(self):
        model = models.get(Models.DEKR_W32_NO_DC, pretrained_weights="coco_pose")

        with tempfile.TemporaryDirectory() as tmp_dirname:
            predictions = model.predict(self.images)
            predictions.show()
            predictions.save(output_folder=tmp_dirname)

    def test_detection_models(self):
        model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")

        with tempfile.TemporaryDirectory() as tmp_dirname:
            predictions = model.predict(self.images)
            predictions.show()
            predictions.save(output_folder=tmp_dirname)

    def test_segmentation_model(self):
        model = models.get(model_name=Models.PP_LITE_T_SEG75, arch_params={"use_aux_heads": False}, num_classes=19, pretrained_weights="cityscapes")

        with tempfile.TemporaryDirectory() as tmp_dirname:
            predictions = model.predict(self.images)
            predictions.show()
            predictions.save(output_folder=tmp_dirname)


if __name__ == "__main__":
    unittest.main()
