import shutil
import unittest
import os
from super_gradients import Trainer, \
    ClassificationTestDatasetInterface, \
    SegmentationTestDatasetInterface, DetectionTestDatasetInterface
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
        dataset_params = {"batch_size": 4}
        dataset = ClassificationTestDatasetInterface(dataset_params=dataset_params)
        trainer.connect_dataset_interface(dataset)
        net = models.get("resnet18", arch_params={"num_classes": 5})
        return trainer, net

    @staticmethod
    def get_detection_trainer(name=''):
        dataset_params = {"batch_size": 4,
                          "test_batch_size": 4,
                          "dataset_dir": "/data/coco/",
                          "s3_link": None,
                          "image_size": 320,
                          "test_collate_fn": DetectionCollateFN(),
                          "train_collate_fn": DetectionCollateFN(),
                          }

        trainer = Trainer(name, model_checkpoints_location='local',
                        multi_gpu=MultiGPUMode.OFF,
                        post_prediction_callback=YoloPostPredictionCallback())
        dataset_interface = DetectionTestDatasetInterface(dataset_params=dataset_params)
        trainer.connect_dataset_interface(dataset_interface, data_loader_num_workers=4)
        net = models.get("yolox_s", arch_params={"num_classes": 5})
        return trainer, net

    @staticmethod
    def get_segmentation_trainer(name=''):
        shelfnet_lw_arch_params = {"num_classes": 5, "load_checkpoint": False}
        trainer = Trainer(name, model_checkpoints_location='local', multi_gpu=False)

        dataset_interface = SegmentationTestDatasetInterface()
        trainer.connect_dataset_interface(dataset_interface, data_loader_num_workers=8)
        net = models.get('shelfnet34_lw', arch_params=shelfnet_lw_arch_params)
        return trainer, net

    def test_test_without_train(self):
        trainer, net = self.get_classification_trainer(self.folder_names[0])
        assert isinstance(trainer.test(net=net, silent_mode=True, test_metrics_list=[Accuracy(), Top5()]), tuple)

        trainer, net = self.get_detection_trainer(self.folder_names[1])

        test_metrics = [DetectionMetrics(post_prediction_callback=trainer.post_prediction_callback, num_cls=5)]

        assert isinstance(trainer.test(net=net, silent_mode=True, test_metrics_list=test_metrics), tuple)

        trainer, net = self.get_segmentation_trainer(self.folder_names[2])
        assert isinstance(trainer.test(net=net, silent_mode=True, test_metrics_list=[IoU(21), PixelAccuracy()]), tuple)

    def test_test_on_valid_loader_without_train(self):
        trainer, net = self.get_classification_trainer(self.folder_names[0])
        assert isinstance(trainer.test(net=net, test_loader=trainer.valid_loader, silent_mode=True, test_metrics_list=[Accuracy(), Top5()]), tuple)

        trainer, net = self.get_detection_trainer(self.folder_names[1])

        test_metrics = [DetectionMetrics(post_prediction_callback=trainer.post_prediction_callback, num_cls=5)]

        assert isinstance(trainer.test(net=net, test_loader=trainer.valid_loader, silent_mode=True, test_metrics_list=test_metrics), tuple)

        trainer, net = self.get_segmentation_trainer(self.folder_names[2])
        assert isinstance(trainer.test(net=net, test_loader=trainer.valid_loader, silent_mode=True, test_metrics_list=[IoU(21), PixelAccuracy()]), tuple)


if __name__ == '__main__':
    unittest.main()
