import torch
import unittest
import torch.nn

from super_gradients.training.utils.segmentation_utils import forward_with_sliding_window_wrapper


class SlidingWindowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.num_classes = 1

    def _assert_tensors_equal(self, tensor1, tensor2):
        self.assertTrue(torch.allclose(tensor1, tensor2, atol=1e-6))

    def test_input_smaller_than_crop_size_and_crop_size_equal_stride_size(self):
        input_size = (512, 512)
        crop_size = (640, 640)
        stride_size = (640, 640)
        model = DummyModel()

        input_tensor = torch.randn((1, 1) + input_size)
        reconstructed_input = forward_with_sliding_window_wrapper(model.forward, input_tensor, stride_size, crop_size, self.num_classes)
        self._assert_tensors_equal(input_tensor, reconstructed_input)

    def test_input_smaller_than_crop_size_and_stride_size_larger_than_crop_size(self):
        input_size = (512, 512)
        crop_size = (640, 640)
        stride_size = (768, 768)
        model = DummyModel()

        input_tensor = torch.randn((1, 1) + input_size)
        with self.assertRaises(ValueError):
            forward_with_sliding_window_wrapper(model.forward, input_tensor, stride_size, crop_size, self.num_classes)

    def test_input_smaller_than_crop_size_and_stride_size_less_than_crop_size(self):
        input_size = (512, 512)
        crop_size = (640, 640)
        stride_size = (384, 384)
        model = DummyModel()

        input_tensor = torch.randn((1, 1) + input_size)
        reconstructed_input = forward_with_sliding_window_wrapper(model.forward, input_tensor, stride_size, crop_size, self.num_classes)

        self._assert_tensors_equal(input_tensor, reconstructed_input)

    def test_input_equal_to_crop_size_and_crop_size_equal_stride_size(self):
        input_size = (512, 512)
        crop_size = (512, 512)
        stride_size = (512, 512)
        model = DummyModel()

        input_tensor = torch.randn((1, 1) + input_size)
        reconstructed_input = forward_with_sliding_window_wrapper(model.forward, input_tensor, stride_size, crop_size, self.num_classes)
        self._assert_tensors_equal(input_tensor, reconstructed_input)

    def test_input_equal_to_crop_size_and_stride_size_larger_than_crop_size(self):
        input_size = (512, 512)
        crop_size = (512, 512)
        stride_size = (640, 640)
        model = DummyModel()

        input_tensor = torch.randn((1, 1) + input_size)
        with self.assertRaises(ValueError):
            forward_with_sliding_window_wrapper(model.forward, input_tensor, stride_size, crop_size, self.num_classes)

    def test_input_equal_to_crop_size_and_stride_size_less_than_crop_size(self):
        input_size = (512, 512)
        crop_size = (512, 512)
        stride_size = (384, 384)
        model = DummyModel()

        input_tensor = torch.randn((1, 1) + input_size)
        reconstructed_input = forward_with_sliding_window_wrapper(model.forward, input_tensor, stride_size, crop_size, self.num_classes)

        self._assert_tensors_equal(input_tensor, reconstructed_input)

    def test_input_larger_than_crop_size_and_crop_size_equal_stride_size(self):
        input_size = (513, 513)
        crop_size = (512, 512)
        stride_size = (512, 512)
        model = DummyModel()

        input_tensor = torch.randn((1, 1) + input_size)
        reconstructed_input = forward_with_sliding_window_wrapper(model.forward, input_tensor, stride_size, crop_size, self.num_classes)
        self._assert_tensors_equal(input_tensor, reconstructed_input)

    def test_input_larger_than_crop_size_and_stride_size_larger_than_crop_size(self):
        input_size = (513, 513)
        crop_size = (512, 512)
        stride_size = (640, 640)
        model = DummyModel()

        input_tensor = torch.randn((1, 1) + input_size)
        with self.assertRaises(ValueError):
            forward_with_sliding_window_wrapper(model.forward, input_tensor, stride_size, crop_size, self.num_classes)

    def test_input_larger_than_crop_size_and_stride_size_less_than_crop_size(self):
        input_size = (513, 513)
        crop_size = (512, 512)
        stride_size = (384, 384)
        model = DummyModel()

        input_tensor = torch.randn((1, 1) + input_size)
        reconstructed_input = forward_with_sliding_window_wrapper(model.forward, input_tensor, stride_size, crop_size, self.num_classes)

        self._assert_tensors_equal(input_tensor, reconstructed_input)

    def test_odd_sized_input(self):
        input_size = (13, 13)
        crop_size = (3, 3)
        stride_size = (2, 2)
        model = DummyModel()

        input_tensor = torch.randn((1, 1) + input_size)
        reconstructed_input = forward_with_sliding_window_wrapper(model.forward, input_tensor, stride_size, crop_size, self.num_classes)

        self._assert_tensors_equal(input_tensor, reconstructed_input)


class DummyModel(torch.nn.Module):
    def forward(self, x):
        return x


if __name__ == "__main__":
    unittest.main()
