import os.path
import tempfile
import unittest

import numpy as np
import onnx
import onnxruntime as ort
import torch.jit

from super_gradients.training.datasets.data_formats import (
    ConcatenatedTensorFormat,
    BoundingBoxesTensorSliceItem,
    TensorSliceItem,
    XYXYCoordinateFormat,
    NormalizedXYWHCoordinateFormat,
    CXCYWHCoordinateFormat,
    YXYXCoordinateFormat,
    NormalizedCXCYWHCoordinateFormat,
    DetectionOutputAdapter,
)

from super_gradients.training.datasets.data_formats.bbox_formats.normalized_cxcywh import xyxy_to_normalized_cxcywh

NORMALIZED_XYWH_SCORES_LABELS = ConcatenatedTensorFormat(
    layout=(
        BoundingBoxesTensorSliceItem(name="bboxes", format=NormalizedXYWHCoordinateFormat()),
        TensorSliceItem(length=1, name="scores"),
        TensorSliceItem(length=1, name="labels"),
    )
)

CXCYWH_SCORES_LABELS = ConcatenatedTensorFormat(
    layout=(
        BoundingBoxesTensorSliceItem(name="bboxes", format=CXCYWHCoordinateFormat()),
        TensorSliceItem(length=1, name="scores"),
        TensorSliceItem(length=1, name="labels"),
    )
)

CXCYWH_LABELS_SCORES_DISTANCE_ATTR = ConcatenatedTensorFormat(
    layout=(
        BoundingBoxesTensorSliceItem(name="bboxes", format=CXCYWHCoordinateFormat()),
        TensorSliceItem(length=1, name="labels"),
        TensorSliceItem(length=1, name="scores"),
        TensorSliceItem(length=1, name="distance"),
        TensorSliceItem(length=4, name="attributes"),
    )
)

ATTR_YXYX = ConcatenatedTensorFormat(
    layout=(
        TensorSliceItem(length=4, name="attributes"),
        BoundingBoxesTensorSliceItem(name="bboxes", format=YXYXCoordinateFormat()),
    )
)


class TestDetectionOutputAdapter(unittest.TestCase):
    @torch.no_grad()
    def test_select_only_some_outputs(self):
        adapter = DetectionOutputAdapter(CXCYWH_LABELS_SCORES_DISTANCE_ATTR, ATTR_YXYX, image_shape=(640, 640)).eval()

        example_inputs = (
            torch.randn((300, CXCYWH_LABELS_SCORES_DISTANCE_ATTR.num_channels)),
            torch.randn((4, 300, CXCYWH_LABELS_SCORES_DISTANCE_ATTR.num_channels)),
        )

        for expected_input in example_inputs:
            intermediate = adapter(expected_input)
            self.assertEqual(ATTR_YXYX.num_channels, intermediate.size(-1))

    @torch.no_grad()
    def test_output_adapter_convert_vice_versa(self):
        adapter = DetectionOutputAdapter(NORMALIZED_XYWH_SCORES_LABELS, CXCYWH_SCORES_LABELS, image_shape=(640, 640)).eval()
        adapter_back = DetectionOutputAdapter(CXCYWH_SCORES_LABELS, NORMALIZED_XYWH_SCORES_LABELS, image_shape=(640, 640)).eval()

        example_inputs = (
            torch.randn((300, NORMALIZED_XYWH_SCORES_LABELS.num_channels)),
            torch.randn((4, 300, NORMALIZED_XYWH_SCORES_LABELS.num_channels)),
        )

        for expected_input in example_inputs:
            intermediate = adapter(expected_input)
            output_actual = adapter_back(intermediate)

            self.assertTrue(torch.allclose(expected_input, output_actual, atol=1e-4))

    @torch.no_grad()
    def test_output_adapter_can_be_traced(self):
        adapter = DetectionOutputAdapter(NORMALIZED_XYWH_SCORES_LABELS, CXCYWH_SCORES_LABELS, image_shape=(640, 640)).eval()

        example_inputs = (
            torch.randn((300, NORMALIZED_XYWH_SCORES_LABELS.num_channels)),
            torch.randn((4, 300, NORMALIZED_XYWH_SCORES_LABELS.num_channels)),
        )

        for inp in example_inputs:
            traced_adapter = torch.jit.trace(adapter, example_inputs=inp, strict=True)

            output_expected = adapter(inp)
            output_actual = traced_adapter(inp)
            self.assertTrue(output_expected.eq(output_actual).all())

    @torch.no_grad()
    def test_output_adapter_can_be_scripted(self):
        adapter = DetectionOutputAdapter(NORMALIZED_XYWH_SCORES_LABELS, CXCYWH_SCORES_LABELS, image_shape=(640, 640)).eval()

        example_inputs = (
            torch.randn((300, NORMALIZED_XYWH_SCORES_LABELS.num_channels)),
            torch.randn((4, 300, NORMALIZED_XYWH_SCORES_LABELS.num_channels)),
        )

        for inp in example_inputs:
            scripted_adapter = torch.jit.script(adapter, example_inputs=[inp])

            output_expected = adapter(inp)
            output_actual = scripted_adapter(inp)
            self.assertTrue(output_expected.eq(output_actual).all())

    @torch.no_grad()
    def test_output_adapter_can_be_onnx_exported(self):
        adapter = DetectionOutputAdapter(NORMALIZED_XYWH_SCORES_LABELS, CXCYWH_SCORES_LABELS, image_shape=(640, 640)).eval()
        example_inputs = (
            torch.randn((300, NORMALIZED_XYWH_SCORES_LABELS.num_channels)),
            torch.randn((4, 300, NORMALIZED_XYWH_SCORES_LABELS.num_channels)),
        )

        for inp in example_inputs:
            expected_output = adapter(inp).numpy()

            with tempfile.TemporaryDirectory() as tmpdirname:
                adapter_fname = os.path.join(tmpdirname, "adapter.onnx")
                torch.onnx.export(adapter, inp, f=adapter_fname, input_names=["predictions"], output_names=["output_predictions"], opset_version=11)

                onnx_model = onnx.load(adapter_fname)
                onnx.checker.check_model(onnx_model)

                ort_sess = ort.InferenceSession(adapter_fname)

                actual_output = ort_sess.run(None, {"predictions": inp.numpy()})[0]

            np.testing.assert_allclose(actual_output, expected_output)

    def test_output_adapter_manual_case(self):

        image_shape = 640, 640

        expected_bboxes_xyxy = np.array(
            [
                [256, 320, 340, 400],
                [32, 64, 100, 150],
                [0, 0, 100, 100],
            ]
        )

        input_bboxes_cxcywh = xyxy_to_normalized_cxcywh(expected_bboxes_xyxy, image_shape)
        input_labels = np.arange(len(expected_bboxes_xyxy))
        input = torch.from_numpy(np.concatenate([input_bboxes_cxcywh, input_labels[:, None]], axis=-1))
        print(input.numpy())

        input_format = ConcatenatedTensorFormat(
            layout=(
                BoundingBoxesTensorSliceItem(name="bboxes", format=NormalizedCXCYWHCoordinateFormat()),
                TensorSliceItem(name="class", length=1),
            )
        )

        output_format = ConcatenatedTensorFormat(
            layout=(
                TensorSliceItem(name="class", length=1),
                BoundingBoxesTensorSliceItem(name="bboxes", format=XYXYCoordinateFormat()),
            )
        )

        output_adapter = DetectionOutputAdapter(input_format, output_format, image_shape)
        output = output_adapter(input)
        output_bboxes = output[:, 1:].numpy()
        print(output.numpy())
        np.testing.assert_allclose(output_bboxes, expected_bboxes_xyxy)


if __name__ == "__main__":
    unittest.main()
