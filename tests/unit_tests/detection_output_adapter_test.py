import os.path
import unittest

import torch.jit

from super_gradients.training.utils.bbox_formats import NormalizedXYWHCoordinateFormat, CXCYWHCoordinateFormat, YXYXCoordinateFormat
from super_gradients.training.utils.output_adapters import DetectionOutputAdapter, ConcatenatedTensorFormat, BoundingBoxesTensorSliceItem, TensorSliceItem

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
    def setUp(self) -> None:
        self.exported_onnx_file = os.path.join(os.path.dirname(__file__), "output_adapter.onnx")

    def doCleanups(self) -> None:
        if os.path.exists(self.exported_onnx_file):
            os.unlink(self.exported_onnx_file)

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
            torch.onnx.export(adapter, inp, self.exported_onnx_file)


if __name__ == "__main__":
    unittest.main()
