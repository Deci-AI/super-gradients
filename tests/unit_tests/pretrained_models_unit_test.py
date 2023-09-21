import os
import shutil
import tempfile
import unittest

import numpy as np
import torch

import super_gradients
from super_gradients.common.object_names import Models
from super_gradients.training import Trainer
from super_gradients.training import models
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader
from super_gradients.training.metrics import Accuracy
from super_gradients.training.pretrained_models import MODEL_URLS, PRETRAINED_NUM_CLASSES
from super_gradients.training.processing.processing import default_yolo_nas_coco_processing_params


class PretrainedModelsUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        super_gradients.init_trainer()
        self.imagenet_pretrained_models = [Models.RESNET50, "repvgg_a0", "regnetY800"]

    def test_pretrained_resnet50_imagenet(self):
        trainer = Trainer("imagenet_pretrained_resnet50_unit_test")
        model = models.get(Models.RESNET50, pretrained_weights="imagenet")
        trainer.test(model=model, test_loader=classification_test_dataloader(), test_metrics_list=[Accuracy()], metrics_progress_verbose=True)

    def test_pretrained_regnetY800_imagenet(self):
        trainer = Trainer("imagenet_pretrained_regnetY800_unit_test")
        model = models.get(Models.REGNETY800, pretrained_weights="imagenet")
        trainer.test(model=model, test_loader=classification_test_dataloader(), test_metrics_list=[Accuracy()], metrics_progress_verbose=True)

    def test_pretrained_repvgg_a0_imagenet(self):
        trainer = Trainer("imagenet_pretrained_repvgg_a0_unit_test")
        model = models.get(Models.REPVGG_A0, pretrained_weights="imagenet", arch_params={"build_residual_branches": True})
        trainer.test(model=model, test_loader=classification_test_dataloader(), test_metrics_list=[Accuracy()], metrics_progress_verbose=True)

    def test_pretrained_models_load_preprocessing_params(self):
        """
        Test that checks whether preprocessing params from pretrained model load correctly.
        """
        state = {"net": models.get(Models.YOLO_NAS_S, num_classes=80).state_dict(), "processing_params": default_yolo_nas_coco_processing_params()}
        with tempfile.TemporaryDirectory() as td:
            checkpoint_path = os.path.join(td, "yolo_nas_s_coco.pth")
            torch.save(state, checkpoint_path)

            MODEL_URLS[Models.YOLO_NAS_S + "_test"] = checkpoint_path
            PRETRAINED_NUM_CLASSES["test"] = 80

            model = models.get(Models.YOLO_NAS_S, pretrained_weights="test")
            # .predict() would fail it model has no preprocessing params
            self.assertIsNotNone(model.predict(np.zeros(shape=(512, 512, 3), dtype=np.uint8)))

    def tearDown(self) -> None:
        if os.path.exists("~/.cache/torch/hub/"):
            shutil.rmtree("~/.cache/torch/hub/")


if __name__ == "__main__":
    unittest.main()
