import unittest

import torch.jit

from super_gradients.training.utils.bbox_formats import NormalizedXYWHCoordinateFormat, CXCYWHCoordinateFormat, YXYXCoordinateFormat
from super_gradients.training.utils.output_adapters.detection_adapter import DetectionOutputAdapter
from super_gradients.training.utils.output_adapters.formats import ConcatenatedTensorFormat, BoundingBoxesTensorSliceItem, TensorSliceItem


class TestDetectionOutputAdapter(unittest.TestCase):
    NORMALIZED_XYWH_SCORES_LABELS = ConcatenatedTensorFormat(
        layout=(
            BoundingBoxesTensorSliceItem(location=slice(0, 4), name="bboxes", format=NormalizedXYWHCoordinateFormat()),
            TensorSliceItem(location=slice(4, 5), name="scores"),
            TensorSliceItem(location=slice(5, 6), name="labels"),
        )
    )

    CXCYWH_LABELS_SCORES = ConcatenatedTensorFormat(
        layout=(
            BoundingBoxesTensorSliceItem(location=slice(0, 4), name="bboxes", format=CXCYWHCoordinateFormat()),
            TensorSliceItem(location=slice(4, 5), name="labels"),
            TensorSliceItem(location=slice(6, 7), name="scores"),
        )
    )

    CXCYWH_LABELS_SCORES_DISTANCE = ConcatenatedTensorFormat(
        layout=(
            BoundingBoxesTensorSliceItem(location=slice(0, 4), name="bboxes", format=CXCYWHCoordinateFormat()),
            TensorSliceItem(location=slice(4, 5), name="labels"),
            TensorSliceItem(location=slice(5, 6), name="scores"),
            TensorSliceItem(location=slice(6, 7), name="distance"),
        )
    )

    LABELS_SCORES_DISTANCE_YXYX = ConcatenatedTensorFormat(
        layout=(
            TensorSliceItem(location=slice(0, 1), name="labels"),
            TensorSliceItem(location=slice(1, 2), name="scores"),
            TensorSliceItem(location=slice(2, 3), name="distance"),
            BoundingBoxesTensorSliceItem(location=slice(3, 7), name="bboxes", format=YXYXCoordinateFormat()),
        )
    )

    @torch.no_grad()
    def test_output_adapter_convert_vice_versa(self):
        adapter = DetectionOutputAdapter(self.CXCYWH_LABELS_SCORES_DISTANCE, self.LABELS_SCORES_DISTANCE_YXYX, image_shape=(640, 640)).eval()
        adapter_back = DetectionOutputAdapter(self.LABELS_SCORES_DISTANCE_YXYX, self.CXCYWH_LABELS_SCORES_DISTANCE, image_shape=(640, 640)).eval()

        example_inputs = (
            torch.randn((300, 7)),
            torch.randn((4, 300, 7)),
        )

        for expected_input in example_inputs:
            intermediate = adapter(expected_input)
            output_actual = adapter_back(intermediate)

            self.assertTrue(torch.allclose(expected_input, output_actual, atol=1e-4))

    @torch.no_grad()
    def test_output_adapter_can_be_traced(self):
        adapter = DetectionOutputAdapter(self.NORMALIZED_XYWH_SCORES_LABELS, self.CXCYWH_LABELS_SCORES, image_shape=(640, 640)).eval()

        example_inputs = (
            torch.randn((300, 6)),
            torch.randn((4, 300, 6)),
        )

        for inp in example_inputs:
            traced_adapter = torch.jit.trace(adapter, example_inputs=inp, strict=True)

            output_expected = adapter(inp)
            output_actual = traced_adapter(inp)
            self.assertTrue(output_expected.eq(output_actual).all())

    @torch.no_grad()
    def test_output_adapter_can_be_scripted(self):
        adapter = DetectionOutputAdapter(self.NORMALIZED_XYWH_SCORES_LABELS, self.CXCYWH_LABELS_SCORES, image_shape=(640, 640)).eval()

        example_inputs = (
            torch.randn((300, 6)),
            torch.randn((4, 300, 6)),
        )

        for inp in example_inputs:
            scripted_adapter = torch.jit.script(adapter, example_inputs=[inp])

            output_expected = adapter(inp)
            output_actual = scripted_adapter(inp)
            self.assertTrue(output_expected.eq(output_actual).all())

    @torch.no_grad()
    def test_output_adapter_can_be_onnx_exported(self):
        adapter = DetectionOutputAdapter(self.NORMALIZED_XYWH_SCORES_LABELS, self.CXCYWH_LABELS_SCORES, image_shape=(640, 640)).eval()
        example_inputs = (
            torch.randn((300, 6)),
            torch.randn((4, 300, 6)),
        )

        for inp in example_inputs:
            torch.onnx.export(adapter, inp, "adapter.onnx")


if __name__ == "__main__":
    unittest.main()
