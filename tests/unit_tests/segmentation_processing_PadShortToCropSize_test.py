import unittest
import numpy as np

from super_gradients.training.processing import SegmentationPadShortToCropSize
from super_gradients.training.utils.predict.predictions import SegmentationPrediction


class SegmentationPadShortToCropSizeTest(unittest.TestCase):
    def test_pad_normal_input(self):
        crop_size = (512, 512)
        fill_image = 0
        pad_transform = SegmentationPadShortToCropSize(crop_size, fill_image)

        # Test for images with different dimensions
        input_image_1 = np.zeros((640, 640))
        output_image_1, metadata_1 = pad_transform.preprocess_image(input_image_1)
        preprocess_output_size_1 = max(crop_size[0], input_image_1.shape[0]), max(crop_size[1], input_image_1.shape[1])
        self.assertEqual(output_image_1.shape, preprocess_output_size_1)

        input_image_2 = np.ones((800, 400))
        output_image_2, metadata_2 = pad_transform.preprocess_image(input_image_2)
        preprocess_output_size_2 = max(crop_size[0], input_image_2.shape[0]), max(crop_size[1], input_image_2.shape[1])
        self.assertEqual(output_image_2.shape, preprocess_output_size_2)

        # Test for crop_size smaller than the input image
        input_image_3 = np.zeros((320, 320))
        output_image_3, metadata_3 = pad_transform.preprocess_image(input_image_3)
        self.assertEqual(output_image_3.shape, crop_size)

    def test_pad_1x1_image(self):
        crop_size = (512, 512)
        fill_image = 0
        pad_transform = SegmentationPadShortToCropSize(crop_size, fill_image)

        input_image = np.ones((1, 1))
        output_image, metadata = pad_transform.preprocess_image(input_image)
        self.assertEqual(output_image.shape, crop_size)

        # test postprocessing
        prediction_obj = SegmentationPrediction(output_image, output_image.shape, output_image.shape)
        output_prediction = pad_transform.postprocess_predictions(prediction_obj, metadata)

        # Check if the output segmentation map has the correct dimensions after removing padding
        self.assertEqual(output_prediction.segmentation_map.shape, input_image.shape)
        self.assertEqual(output_prediction.segmentation_map.all(), True)

    def test_pad_edge_cases(self):
        crop_size = (512, 512)
        fill_image = 0
        pad_transform = SegmentationPadShortToCropSize(crop_size, fill_image)

        # Test for crop_size equal to the input image size
        input_image_1 = np.zeros((512, 512))
        output_image_1, metadata_1 = pad_transform.preprocess_image(input_image_1)
        self.assertEqual(output_image_1.shape, crop_size)

        # Test for crop_size smaller than the input image
        input_image_2 = np.zeros((400, 400))
        output_image_2, metadata_2 = pad_transform.preprocess_image(input_image_2)
        self.assertEqual(output_image_2.shape, crop_size)

    def test_postprocess_predictions(self):
        crop_size = (512, 512)
        fill_image = 0
        pad_transform = SegmentationPadShortToCropSize(crop_size, fill_image)

        # Create a segmentation prediction object with a known segmentation map shape
        input_image_shape = (400, 400)
        input_image = np.ones(input_image_shape)
        segmentation_map, metadata = pad_transform.preprocess_image(input_image)
        prediction_obj = SegmentationPrediction(segmentation_map, crop_size, crop_size)

        # Apply the postprocess_predictions method with known padding_coordinates
        output_prediction = pad_transform.postprocess_predictions(prediction_obj, metadata)

        # Check if the output segmentation map has the correct dimensions after removing padding
        self.assertEqual(output_prediction.segmentation_map.shape, input_image_shape)
        self.assertEqual(output_prediction.segmentation_map.all(), True)


if __name__ == "__main__":
    unittest.main()
