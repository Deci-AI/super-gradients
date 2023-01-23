import tempfile
import unittest

from super_gradients.common.object_names import Models
from super_gradients.training import models
from torchvision.transforms import Compose, Normalize, Resize
from super_gradients.training.transforms import Standardize
import os


class TestModelsONNXExport(unittest.TestCase):
    def test_models_onnx_export(self):
        pretrained_model = models.get(Models.RESNET18, num_classes=1000, pretrained_weights="imagenet")
        preprocess = Compose([Resize(224), Standardize(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        with tempfile.TemporaryDirectory() as tmpdirname:
            out_path = os.path.join(tmpdirname, "resnet18.onnx")
            models.convert_to_onnx(model=pretrained_model, out_path=out_path, input_shape=(3, 256, 256), pre_process=preprocess)


if __name__ == "__main__":
    unittest.main()
