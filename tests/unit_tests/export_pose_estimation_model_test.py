import logging
import os
import tempfile
import unittest

import cv2
import numpy as np
import onnxruntime
import torch
from matplotlib import pyplot as plt
from torch import nn

from super_gradients.common.object_names import Models
from super_gradients.conversion.conversion_enums import ExportTargetBackend, DetectionOutputFormatMode
from super_gradients.conversion.gs_utils import import_onnx_graphsurgeon_or_fail_with_instructions
from super_gradients.module_interfaces import ExportablePoseEstimationModel, PoseEstimationModelExportResult
from super_gradients.training import models
from super_gradients.training.dataloaders import coco2017_val  # noqa
from super_gradients.training.models.pose_estimation_models.yolo_nas_pose.yolo_nas_pose_variants import YoloNASPoseDecodingModule
from super_gradients.training.processing.processing import (
    default_yolo_nas_pose_coco_processing_params,
    ComposeProcessing,
    ReverseImageChannels,
    KeypointsLongestMaxSizeRescale,
    KeypointsBottomRightPadding,
    StandardizeImage,
    ImagePermute,
)
from super_gradients.training.utils.media.image import load_image
from super_gradients.training.utils.visualization.pose_estimation import PoseVisualization

gs = import_onnx_graphsurgeon_or_fail_with_instructions()


class TestPoseEstimationModelExport(unittest.TestCase):
    def setUp(self) -> None:
        logging.getLogger().setLevel(logging.DEBUG)
        this_dir = os.path.dirname(__file__)
        self.test_image_path = os.path.join(this_dir, "../data/tinycoco/images/val2017/000000444010.jpg")

        self.default_params = default_yolo_nas_pose_coco_processing_params()
        self.default_model = Models.YOLO_NAS_POSE_S

        # Custom preprocessing params for 20 keypoints
        self.custom_params = dict(
            image_processor=ComposeProcessing(
                [
                    ReverseImageChannels(),
                    KeypointsLongestMaxSizeRescale(output_shape=(640, 640)),
                    KeypointsBottomRightPadding(output_shape=(640, 640), pad_value=127),
                    StandardizeImage(max_value=255.0),
                    ImagePermute(permutation=(2, 0, 1)),
                ]
            ),
            edge_links=[],  # No skeleton
            edge_colors=[],
            keypoint_colors=np.random.randint(0, 255, size=(20, 3)).tolist(),
        )

    def test_export_decoding_module_bs_3(self):
        num_pre_nms_predictions = 1000
        batch_size = 3
        module = YoloNASPoseDecodingModule(num_pre_nms_predictions)

        pred_bboxes_xyxy = torch.rand(batch_size, 8400, 4)
        pred_bboxes_conf = torch.rand(batch_size, 8400, 1).sigmoid()
        pred_pose_coords = torch.rand(batch_size, 8400, 20, 2)
        pred_pose_scores = torch.rand(batch_size, 8400, 20).sigmoid()

        inputs = (pred_bboxes_xyxy, pred_bboxes_conf, pred_pose_coords, pred_pose_scores)
        _ = module([inputs])  # Check that normal forward() works

        with tempfile.TemporaryDirectory() as tmpdirname:
            out_path = os.path.join(tmpdirname, "model.onnx")
            torch.onnx.export(module, (inputs,), out_path)

    def test_export_model_on_small_size(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            for model_type in [
                Models.YOLO_NAS_POSE_S,
            ]:
                out_path = os.path.join(tmpdirname, model_type + ".onnx")
                model: ExportablePoseEstimationModel = models.get(model_type, num_classes=17)
                model.set_dataset_processing_params(**default_yolo_nas_pose_coco_processing_params())
                export_result = model.export(
                    out_path,
                    input_image_shape=(64, 64),
                    num_pre_nms_predictions=2000,
                    max_predictions_per_image=1000,
                    output_predictions_format=DetectionOutputFormatMode.BATCH_FORMAT,
                )
                assert export_result.input_image_dtype == torch.uint8
                assert export_result.input_image_shape == (64, 64)
                print(export_result.usage_instructions)

    def test_export_model_with_batch_size_4(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            for model_type in [
                Models.YOLO_NAS_POSE_S,
            ]:
                out_path = os.path.join(tmpdirname, model_type + ".onnx")
                model: ExportablePoseEstimationModel = models.get(model_type, num_classes=17)
                model.set_dataset_processing_params(**default_yolo_nas_pose_coco_processing_params())
                export_result = model.export(
                    out_path,
                    batch_size=4,
                    input_image_shape=(640, 640),
                    num_pre_nms_predictions=2000,
                    max_predictions_per_image=1000,
                    output_predictions_format=DetectionOutputFormatMode.FLAT_FORMAT,
                )
                assert export_result.input_image_dtype == torch.uint8
                assert export_result.input_image_shape == (640, 640)
                print(export_result.usage_instructions)

    def test_the_most_common_export_use_case(self):
        """
        Test the most common export use case - export to ONNX with all default parameters
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            out_path = os.path.join(tmpdirname, "model.onnx")

            model: ExportablePoseEstimationModel = models.get(self.default_model, num_classes=17)
            model.set_dataset_processing_params(**self.default_params)

            export_result = model.export(out_path)
            assert export_result.input_image_dtype == torch.uint8
            assert export_result.input_image_shape == (640, 640)
            assert export_result.input_image_channels == 3

            print(export_result.usage_instructions)

    def test_models_produce_half(self):
        if not torch.cuda.is_available():
            self.skipTest("This test was skipped because target machine has not CUDA devices")

        input = torch.randn(1, 3, 640, 640).half().cuda()

        model = models.get(Models.YOLO_NAS_POSE_S, num_classes=17)
        model = nn.Sequential(model, model.get_decoding_module(100)).cuda().eval().half()
        output = model(input)
        assert output[0].dtype == torch.float16
        assert output[1].dtype == torch.float16

    def test_export_to_onnxruntime_flat(self):
        """
        Test export to ONNX with flat predictions
        """
        output_predictions_format = DetectionOutputFormatMode.FLAT_FORMAT
        confidence_threshold = 0.7
        nms_threshold = 0.6

        with tempfile.TemporaryDirectory() as tmpdirname:
            for model_type in [
                Models.YOLO_NAS_POSE_S,
            ]:
                model_name = str(model_type).lower().replace(".", "_")
                out_path = os.path.join(tmpdirname, f"{model_name}_onnxruntime_flat.onnx")

                # Intentionaly export with 20 keypoints to ensure NMS/postprocessing works correctly
                model_arch: ExportablePoseEstimationModel = models.get(model_name, num_classes=20)
                model_arch.set_dataset_processing_params(**self.custom_params)

                export_result = model_arch.export(
                    out_path,
                    engine=ExportTargetBackend.ONNXRUNTIME,
                    output_predictions_format=output_predictions_format,
                    confidence_threshold=confidence_threshold,
                    nms_threshold=nms_threshold,
                )
                print(export_result.usage_instructions)

                [flat_predictions] = self._run_inference_with_onnx(export_result, params=self.custom_params)

                # Check that all predictions have confidence >= confidence_threshold
                assert (flat_predictions[:, 5] >= confidence_threshold).all()

    def test_export_to_onnxruntime_batch_format(self):
        output_predictions_format = DetectionOutputFormatMode.BATCH_FORMAT
        confidence_threshold = 0.7
        nms_threshold = 0.6

        with tempfile.TemporaryDirectory() as tmpdirname:
            for model_type in [
                Models.YOLO_NAS_POSE_S,
            ]:
                model_name = str(model_type).lower().replace(".", "_")
                out_path = os.path.join(tmpdirname, f"{model_name}_onnxruntime_batch.onnx")

                # Intentionaly export with 20 keypoints to ensure NMS/postprocessing works correctly
                model_arch: ExportablePoseEstimationModel = models.get(model_name, num_classes=20)
                model_arch.set_dataset_processing_params(**self.custom_params)
                export_result = model_arch.export(
                    out_path,
                    engine=ExportTargetBackend.ONNXRUNTIME,
                    output_predictions_format=output_predictions_format,
                    nms_threshold=nms_threshold,
                    confidence_threshold=confidence_threshold,
                )
                print(export_result.usage_instructions)

                self._run_inference_with_onnx(export_result, params=self.custom_params)

    def _run_inference_with_onnx(self, export_result: PoseEstimationModelExportResult, params=None):
        if params is None:
            params = self.default_params
        # onnx_filename = out_path, input_shape = export_result.image_shape, output_predictions_format = output_predictions_format

        image = self._get_image_as_bchw(export_result.input_image_shape)
        image_8u = self._get_image(export_result.input_image_shape)

        session = onnxruntime.InferenceSession(export_result.output)
        inputs = [o.name for o in session.get_inputs()]
        outputs = [o.name for o in session.get_outputs()]
        result = session.run(outputs, {inputs[0]: image})

        num_keypoints = len(params["keypoint_colors"])
        if export_result.output_predictions_format == DetectionOutputFormatMode.FLAT_FORMAT:
            flat_predictions = result[0]  # [N, (batch_index, x1, y1, x2, y2, score, num_keypoints * 3)]
            print(flat_predictions.shape[1])
            print(1 + 4 + 1 + num_keypoints * 3)
            assert flat_predictions.shape[1] == 1 + 4 + 1 + num_keypoints * 3

            boxes = flat_predictions[:, 1:5]
            scores = flat_predictions[:, 5]
            poses = flat_predictions[:, 6:].reshape(-1, num_keypoints, 3)

            image_8u = PoseVisualization.draw_poses(
                image=image_8u,
                poses=poses,
                boxes=boxes,
                scores=scores,
                is_crowd=None,
                show_keypoint_confidence=True,
                edge_links=params["edge_links"],
                edge_colors=params["edge_colors"],
                keypoint_colors=params["keypoint_colors"],
            )

        else:
            # Hard-coded unpacking for batch size 1
            [num_predictions], [pred_boxes], [pred_scores], [pred_joints] = result

            image_8u = PoseVisualization.draw_poses(
                image=image_8u,
                poses=pred_joints[0 : num_predictions[0]],
                boxes=pred_boxes[0 : num_predictions[0]],
                scores=pred_scores[0 : num_predictions[0]],
                is_crowd=None,
                show_keypoint_confidence=True,
                edge_links=params["edge_links"],
                edge_colors=params["edge_colors"],
                keypoint_colors=params["keypoint_colors"],
            )

        plt.figure(figsize=(10, 10))
        plt.imshow(image_8u)
        plt.title(os.path.basename(export_result.output))
        plt.tight_layout()
        plt.show()

        return result

    def _get_image_as_bchw(self, image_shape=(640, 640)):
        """

        :param image_shape: Output image shape (rows, cols)
        :return: Image in NCHW format
        """

        image = load_image(self.test_image_path)
        image = cv2.resize(image, dsize=tuple(reversed(image_shape)), interpolation=cv2.INTER_LINEAR)
        image = np.transpose(np.expand_dims(image, 0), (0, 3, 1, 2))
        return image

    def _get_image(self, image_shape=(640, 640)):
        """

        :param image_shape: Output image shape (rows, cols)
        :return: Image in HWC format
        """

        image = load_image(self.test_image_path)
        image = cv2.resize(image, dsize=tuple(reversed(image_shape)), interpolation=cv2.INTER_LINEAR)
        return image


if __name__ == "__main__":
    unittest.main()
