import os
import tempfile
import unittest

from torchvision.transforms import Compose, Normalize, Resize

from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training.transforms import Standardize


class TestModelsCoreMLExport(unittest.TestCase):
    def test_models_onnx_export_with_explicit_input_size(self):
        pretrained_model = models.get(Models.RESNET18, num_classes=1000, pretrained_weights="imagenet")
        preprocess = Compose([Resize(224), Standardize(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        with tempfile.TemporaryDirectory() as tmpdirname:
            out_path = os.path.join(tmpdirname, "resnet18.mlmodel")
            models.convert_to_coreml(model=pretrained_model, out_path=out_path, input_size=(3, 256, 256), pre_process=preprocess)
            self.assertTrue(os.path.isfile(out_path))

    def test_models_onnx_export_without_explicit_input_size_raises_error(self):
        pretrained_model = models.get(Models.RESNET18, num_classes=1000, pretrained_weights="imagenet")
        preprocess = Compose([Resize(224), Standardize(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        with self.assertRaises(KeyError):
            models.convert_to_coreml(model=pretrained_model, out_path="some-output-path.coreml", pre_process=preprocess)

    def test_models_coreml_export(self, **export_kwargs):
        pretrained_model = models.get(Models.YOLO_NAS_S, num_classes=1000, pretrained_weights="coco")

        # Just for the sake of testing, not really COCO preprocessing
        preprocess = Compose([Resize(224), Standardize(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        with tempfile.TemporaryDirectory() as tmpdirname:
            out_path = os.path.join(tmpdirname, "yolo_nas_s")
            model_path = models.convert_to_coreml(
                model=pretrained_model,
                out_path=out_path,
                pre_process=preprocess,
                prep_model_for_conversion_kwargs=dict(input_size=(1, 3, 640, 640)),
                **export_kwargs,
            )

            if export_kwargs.get("export_as_ml_program"):
                # Expecting a directory
                self.assertTrue(os.path.isdir(model_path))
                self.assertTrue(model_path.endswith(".mlpackage"))
            else:
                # Expecting a single file
                self.assertTrue(os.path.isfile(model_path))
                self.assertTrue(model_path.endswith(".mlmodel"))

    def test_models_coreml_export_as_mlprogram(self):
        self.test_models_coreml_export(export_as_ml_program=True)


if __name__ == "__main__":
    unittest.main()
