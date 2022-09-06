import shutil
import unittest
import os
from super_gradients import Trainer, \
    ClassificationTestDatasetInterface, \
    SegmentationTestDatasetInterface, DetectionTestDatasetInterface
from super_gradients.training.dataloaders.dataloaders import classification_test_dataloader, \
    detection_test_dataloader, segmentation_test_dataloader
from super_gradients.training.metrics import Accuracy, Top5
from super_gradients.training import MultiGPUMode, models
from super_gradients.training.models.detection_models.yolo_base import YoloPostPredictionCallback
from super_gradients.training.utils.detection_utils import DetectionCollateFN
from super_gradients.training.metrics.detection_metrics import DetectionMetrics
from super_gradients.training.metrics.segmentation_metrics import PixelAccuracy, IoU


class TestWithoutTrainTest(unittest.TestCase):
    @classmethod
    def setUp(cls):
        # NAMES FOR THE EXPERIMENTS TO LATER DELETE
        cls.folder_names = ['test_classification_model', 'test_detection_model', 'test_segmentation_model']

    @classmethod
    def tearDownClass(cls) -> None:
        # ERASE ALL THE FOLDERS THAT WERE CREATED DURING THIS TEST
        for folder in cls.folder_names:
            if os.path.isdir(os.path.join('checkpoints', folder)):
                shutil.rmtree(os.path.join('checkpoints', folder))

    @staticmethod
    def get_classification_trainer(name=''):
        trainer = Trainer(name, model_checkpoints_location='local')
        model = models.get("resnet18", num_classes=5)
        return trainer, model

    @staticmethod
    def get_detection_trainer(name=''):
        trainer = Trainer(name, model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF,
                          post_prediction_callback=YoloPostPredictionCallback())
        model = models.get("yolox_s", num_classes=5)
        return trainer, model

    @staticmethod
    def get_segmentation_trainer(name=''):
        shelfnet_lw_arch_params = {"num_classes": 5, "load_checkpoint": False}
        trainer = Trainer(name, model_checkpoints_location='local', multi_gpu=False)
        model = models.get('shelfnet34_lw', arch_params=shelfnet_lw_arch_params)
        return trainer, model

    def test_test_without_train(self):
        trainer, model = self.get_classification_trainer(self.folder_names[0])
        assert isinstance(trainer.test(model=model, silent_mode=True, test_metrics_list=[Accuracy(), Top5()], test_loader=classification_test_dataloader()), tuple)

        trainer, model = self.get_detection_trainer(self.folder_names[1])

        test_metrics = [DetectionMetrics(post_prediction_callback=trainer.post_prediction_callback, num_cls=5)]

        assert isinstance(trainer.test(model=model, silent_mode=True, test_metrics_list=test_metrics, test_loader=detection_test_dataloader(image_size=320)), tuple)

        trainer, model = self.get_segmentation_trainer(self.folder_names[2])
        assert isinstance(trainer.test(model=model, silent_mode=True, test_metrics_list=[IoU(21), PixelAccuracy()], test_loader=segmentation_test_dataloader()), tuple)


if __name__ == '__main__':
    unittest.main()
