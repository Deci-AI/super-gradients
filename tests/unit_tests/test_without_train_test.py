import shutil
import unittest
import os
from super_gradients import Trainer
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader, detection_test_dataloader, segmentation_test_dataloader
from super_gradients.training.metrics import Accuracy, Top5
from super_gradients.training import models
from super_gradients.training.metrics.detection_metrics import DetectionMetrics
from super_gradients.training.metrics.segmentation_metrics import PixelAccuracy, IoU
from super_gradients.training.models.detection_models.yolo_base import YoloPostPredictionCallback
from super_gradients.common.object_names import Models


class TestWithoutTrainTest(unittest.TestCase):
    @classmethod
    def setUp(cls):
        # NAMES FOR THE EXPERIMENTS TO LATER DELETE
        cls.folder_names = ["test_classification_model", "test_detection_model", "test_segmentation_model"]

    @classmethod
    def tearDownClass(cls) -> None:
        # ERASE ALL THE FOLDERS THAT WERE CREATED DURING THIS TEST
        for folder in cls.folder_names:
            if os.path.isdir(os.path.join("checkpoints", folder)):
                shutil.rmtree(os.path.join("checkpoints", folder))

    @staticmethod
    def get_classification_trainer(name=""):
        trainer = Trainer(name)
        model = models.get(Models.RESNET18, num_classes=5)
        return trainer, model

    @staticmethod
    def get_detection_trainer(name=""):
        trainer = Trainer(name)
        model = models.get(Models.YOLOX_S, num_classes=5)
        return trainer, model

    @staticmethod
    def get_segmentation_trainer(name=""):
        shelfnet_lw_arch_params = {"num_classes": 5}
        trainer = Trainer(name)
        model = models.get(Models.SHELFNET34_LW, arch_params=shelfnet_lw_arch_params)
        return trainer, model

    def test_test_without_train(self):
        trainer, model = self.get_classification_trainer(self.folder_names[0])
        assert isinstance(
            trainer.test(model=model, silent_mode=True, test_metrics_list=[Accuracy(), Top5()], test_loader=classification_test_dataloader()), dict
        )

        trainer, model = self.get_detection_trainer(self.folder_names[1])

        test_metrics = [DetectionMetrics(post_prediction_callback=YoloPostPredictionCallback(), num_cls=5)]

        assert isinstance(
            trainer.test(model=model, silent_mode=True, test_metrics_list=test_metrics, test_loader=detection_test_dataloader(image_size=320)), dict
        )

        trainer, model = self.get_segmentation_trainer(self.folder_names[2])
        assert isinstance(
            trainer.test(model=model, silent_mode=True, test_metrics_list=[IoU(21), PixelAccuracy()], test_loader=segmentation_test_dataloader()), dict
        )


if __name__ == "__main__":
    unittest.main()
