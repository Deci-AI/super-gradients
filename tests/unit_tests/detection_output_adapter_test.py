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

    YXYX_LABELS_SCORES_DISTANCE = ConcatenatedTensorFormat(
        layout=(
            BoundingBoxesTensorSliceItem(location=slice(0, 4), name="bboxes", format=YXYXCoordinateFormat()),
            TensorSliceItem(location=slice(4, 5), name="labels"),
            TensorSliceItem(location=slice(5, 6), name="scores"),
            TensorSliceItem(location=slice(6, 7), name="distance"),
        )
    )

    @torch.no_grad()
    def test_output_adapter_can_be_traced(self):
        adapter = DetectionOutputAdapter(self.NORMALIZED_XYWH_SCORES_LABELS, self.CXCYWH_LABELS_SCORES, image_shape=(640, 640)).eval()

        example_inputs = (
            torch.randn((300, 6)),
            torch.randn((4, 300, 6)),
        )

        traced_adapter = torch.jit.trace(adapter, example_inputs=example_inputs, strict=True)
        for inp in example_inputs:
            output_expected = adapter(inp)
            output_actual = traced_adapter(inp)
            self.assertTrue(output_expected.eq(output_actual).all())

    @torch.no_grad()
    def test_output_adapter_can_be_scripted(self):
        adapter = DetectionOutputAdapter(self.NORMALIZED_XYWH_SCORES_LABELS, self.CXCYWH_LABELS_SCORES, image_shape=(640, 640)).eval()
        scripted_adapter = torch.jit.script(adapter)
        example_inputs = (
            torch.randn((300, 6)),
            torch.randn((4, 300, 6)),
        )

        for inp in example_inputs:
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
