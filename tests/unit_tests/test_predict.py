import os
import unittest
import tempfile
from pathlib import Path

from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training.datasets import COCODetectionDataset

import cv2
import numpy as np


class TestModelPredict(unittest.TestCase):
    def setUp(self) -> None:
        rootdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.images = [
            os.path.join(rootdir, "documentation", "source", "images", "examples", "countryside.jpg"),
            os.path.join(rootdir, "documentation", "source", "images", "examples", "street_busy.jpg"),
            "https://deci-datasets-research.s3.amazonaws.com/image_samples/beatles-abbeyroad.jpg",
        ]
        self._set_images_with_targets()

    def _set_images_with_targets(self):
        mini_coco_data_dir = str(Path(__file__).parent.parent / "data" / "tinycoco")
        dataset = COCODetectionDataset(
            data_dir=mini_coco_data_dir, subdir="images/val2017", json_file="instances_val2017.json", input_dim=None, transforms=[], cache_annotations=False
        )
        # x's are np.ndarrays images of shape (H,W,3)
        # y's are np.ndarrays of shape (num_boxes,x1,y1,x2,y2,class_id)
        x1, y1, _ = dataset[0]
        x2, y2, _ = dataset[1]
        # images from COCODetectionDataset are RGB and images as np.ndarrays are expected to be BGR
        x2 = x2[:, :, ::-1]
        x1 = x1[:, :, ::-1]
        self.np_array_images = [x1, x2]
        self.np_array_target_bboxes = [y1[:, :4], y2[:, :4]]
        self.np_array_target_class_ids = [y1[:, 4], y2[:, 4]]

    def _prepare_video(self, path):
        video_width, video_height = 400, 400
        fps = 10
        num_frames = 20
        video_writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (video_width, video_height),
        )

        frames = np.zeros((num_frames, video_height, video_width, 3), dtype=np.uint8)
        for frame in frames:
            video_writer.write(frame)
        video_writer.release()

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
        for model_name in [Models.YOLO_NAS_S, Models.YOLOX_S, Models.PP_YOLOE_S]:
            model = models.get(model_name, pretrained_weights="coco")

            with tempfile.TemporaryDirectory() as tmp_dirname:
                predictions = model.predict(self.images)
                predictions.show()
                predictions.save(output_folder=tmp_dirname)
                for prediction in predictions._images_prediction_lst:
                    self.assertTrue(np.issubdtype(prediction.prediction.labels.dtype, np.integer))

    def test_detection_models_with_targets(self):
        for model_name in [Models.YOLO_NAS_S, Models.YOLOX_S, Models.PP_YOLOE_S]:
            model = models.get(model_name, pretrained_weights="coco")

            with tempfile.TemporaryDirectory() as tmp_dirname:
                predictions = model.predict(self.np_array_images)
                predictions.show(target_bboxes=self.np_array_target_bboxes, target_class_ids=self.np_array_target_class_ids, target_bboxes_format="xyxy")
                predictions.save(
                    output_folder=tmp_dirname,
                    target_bboxes=self.np_array_target_bboxes,
                    target_class_ids=self.np_array_target_class_ids,
                    target_bboxes_format="xyxy",
                )

    def test_segmentation_predict_pplite_t_seg75(self):
        model = models.get(model_name=Models.PP_LITE_T_SEG75, pretrained_weights="cityscapes")

        with tempfile.TemporaryDirectory() as tmp_dirname:
            predictions = model.predict(self.images)
            predictions.show()
            predictions.save(output_folder=tmp_dirname)

    def test_segmentation_predict_stdc1_seg50(self):
        model = models.get(model_name=Models.STDC1_SEG50, pretrained_weights="cityscapes")

        with tempfile.TemporaryDirectory() as tmp_dirname:
            predictions = model.predict(self.images)
            predictions.show()
            predictions.save(output_folder=tmp_dirname)

    def test_segmentation_predict_ddrnet23_slim(self):
        model = models.get(model_name=Models.DDRNET_23_SLIM, pretrained_weights="cityscapes")

        with tempfile.TemporaryDirectory() as tmp_dirname:
            predictions = model.predict(self.images)
            predictions.show()
            predictions.save(output_folder=tmp_dirname)

    def test_predict_class_names(self):
        for model_name in [Models.YOLO_NAS_S, Models.YOLOX_S, Models.PP_YOLOE_S]:
            model = models.get(model_name, pretrained_weights="coco")

            predictions = model.predict(self.np_array_images)
            _ = predictions.show(class_names=["person", "bicycle", "car", "motorcycle", "airplane", "bus"])

            with self.assertRaises(ValueError):
                _ = predictions.show(class_names=["human"])

    def test_predict_video(self):
        with tempfile.TemporaryDirectory() as tmp_dirname:
            video_path = os.path.join(tmp_dirname, "test.mp4")
            self._prepare_video(video_path)
            for model_name in [Models.YOLO_NAS_S, Models.YOLOX_S, Models.YOLO_NAS_POSE_S]:
                pretrained_weights = "coco"
                if model_name == Models.YOLO_NAS_POSE_S:
                    pretrained_weights += "_pose"
                model = models.get(model_name, pretrained_weights=pretrained_weights)

                predictions = model.predict(video_path)
                predictions.save(os.path.join(tmp_dirname, "test_predict_video_detection.mp4"))

                predictions = model.predict(video_path)
                predictions.save(os.path.join(tmp_dirname, "test_predict_video_detection.gif"))

    def test_predict_detection_skip_resize(self):
        for model_name in [Models.YOLO_NAS_S, Models.YOLOX_S, Models.PP_YOLOE_S]:
            model = models.get(model_name, pretrained_weights="coco")
            pipeline = model._get_pipeline(skip_image_resizing=True)

            dummy_images = [np.random.random((21, 21, 3)), np.random.random((21, 32, 3)), np.random.random((640, 640, 3))]
            expected_preprocessing_shape = [(3, 32, 32), (3, 32, 32), (3, 640, 640)]
            for image, expected_shape in zip(dummy_images, expected_preprocessing_shape):
                pred = model.predict(image, skip_image_resizing=True)
                self.assertEqual(image.shape, pred.draw().shape)

                preprocessed_shape = pipeline.image_processor.preprocess_image(image)[0].shape
                self.assertEqual(preprocessed_shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
