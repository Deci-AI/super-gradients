import shutil
import unittest
import os
from super_gradients import SgModel, \
    ClassificationTestDatasetInterface, \
    SegmentationTestDatasetInterface, DetectionTestDatasetInterface
from super_gradients.training.metrics import Accuracy, Top5
from super_gradients.training import MultiGPUMode
from super_gradients.training.utils.detection_utils import base_detection_collate_fn
from super_gradients.training.datasets.datasets_utils import ComposedCollateFunction, MultiScaleCollateFunction
from super_gradients.training.utils.detection_utils import YoloV3NonMaxSuppression
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
        model = SgModel(name, model_checkpoints_location='local')
        dataset_params = {"batch_size": 4}
        dataset = ClassificationTestDatasetInterface(dataset_params=dataset_params)
        model.connect_dataset_interface(dataset)
        model.build_model("resnet18_cifar", load_checkpoint=False)
        return model

    @staticmethod
    def get_detection_trainer(name=''):
        collate_fn_holder = ComposedCollateFunction([base_detection_collate_fn, MultiScaleCollateFunction(320)])
        dataset_params = {"batch_size": 4,
                          "test_batch_size": 4,
                          "dataset_dir": "/data/coco/",
                          "s3_link": None,
                          "image_size": 320,
                          "test_collate_fn": base_detection_collate_fn,
                          "train_collate_fn": collate_fn_holder,
                          }

        yolo_v3_arch_params = {"image_size": 320, "iou_t": 0.225, "multi_gpu_mode": "distributed_data_parallel"}

        post_prediction_callback = YoloV3NonMaxSuppression()
        model = SgModel(name, model_checkpoints_location='local',
                        multi_gpu=MultiGPUMode.OFF,
                        post_prediction_callback=post_prediction_callback)
        dataset_interface = DetectionTestDatasetInterface(dataset_params=dataset_params)
        model.connect_dataset_interface(dataset_interface, data_loader_num_workers=4)
        model.build_model('yolo_v3', arch_params=yolo_v3_arch_params, load_checkpoint=False)
        return model

    @staticmethod
    def get_segmentation_trainer(name=''):
        shelfnet_lw_arch_params = {"num_classes": 5, "load_checkpoint": False}
        model = SgModel(name, model_checkpoints_location='local', multi_gpu=False)

        dataset_interface = SegmentationTestDatasetInterface()
        model.connect_dataset_interface(dataset_interface, data_loader_num_workers=8)
        model.build_model('shelfnet34', arch_params=shelfnet_lw_arch_params)
        return model

    def test_test_without_train(self):
        model = self.get_classification_trainer(self.folder_names[0])
        assert isinstance(model.test(silent_mode=True, test_metrics_list=[Accuracy(), Top5()]), tuple)

        model = self.get_detection_trainer(self.folder_names[1])

        test_metrics = [DetectionMetrics(post_prediction_callback=model.post_prediction_callback, num_cls=5)]

        assert isinstance(model.test(silent_mode=True, test_metrics_list=test_metrics), tuple)

        model = self.get_segmentation_trainer(self.folder_names[2])
        assert isinstance(model.test(silent_mode=True, test_metrics_list=[IoU(21), PixelAccuracy()]), tuple)

    def test_test_on_valid_loader_without_train(self):
        model = self.get_classification_trainer(self.folder_names[0])
        assert isinstance(model.test(test_loader=model.valid_loader, silent_mode=True, test_metrics_list=[Accuracy(), Top5()]), tuple)

        model = self.get_detection_trainer(self.folder_names[1])

        test_metrics = [DetectionMetrics(post_prediction_callback=model.post_prediction_callback, num_cls=5)]

        assert isinstance(model.test(test_loader=model.valid_loader, silent_mode=True, test_metrics_list=test_metrics), tuple)

        model = self.get_segmentation_trainer(self.folder_names[2])
        assert isinstance(model.test(test_loader=model.valid_loader, silent_mode=True, test_metrics_list=[IoU(21), PixelAccuracy()]), tuple)


if __name__ == '__main__':
    unittest.main()
