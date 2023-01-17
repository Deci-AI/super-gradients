import unittest

import torch
from torchvision.transforms import Compose, ToTensor
from super_gradients.training.transforms.transforms import SegRescale, SegRandomRescale, SegCropImageAndMask, SegPadShortToCropSize
from PIL import Image
from super_gradients.training.datasets.segmentation_datasets.segmentation_dataset import SegmentationDataSet


class SegmentationTransformsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.default_image_value = 0
        self.default_mask_value = 0

    def create_sample(self, size):
        sample = {
            "image": Image.new(mode="RGB", size=size, color=self.default_image_value),
            "mask": Image.new(mode="L", size=size, color=self.default_mask_value),
        }
        return sample

    def test_rescale_with_scale_factor(self):
        # test raise exception for negative and zero scale factor
        kwargs = {"scale_factor": -2}
        self.failUnlessRaises(ValueError, SegRescale, **kwargs)
        kwargs = {"scale_factor": 0}
        self.failUnlessRaises(ValueError, SegRescale, **kwargs)

        # test scale down
        sample = self.create_sample((1024, 512))
        rescale_scale05 = SegRescale(scale_factor=0.5)
        out = rescale_scale05(sample)
        self.assertEqual((512, 256), out["image"].size)

        # test scale up
        sample = self.create_sample((1024, 512))
        rescale_scale2 = SegRescale(scale_factor=2.0)
        out = rescale_scale2(sample)
        self.assertEqual((2048, 1024), out["image"].size)

        # test scale_factor is stronger than other params
        sample = self.create_sample((1024, 512))
        rescale_scale05 = SegRescale(scale_factor=0.5, short_size=300, long_size=600)
        out = rescale_scale05(sample)
        self.assertEqual((512, 256), out["image"].size)

    def test_rescale_with_short_size(self):
        # test raise exception for negative and zero short_size
        kwargs = {"short_size": 0}
        self.failUnlessRaises(ValueError, SegRescale, **kwargs)
        kwargs = {"short_size": -200}
        self.failUnlessRaises(ValueError, SegRescale, **kwargs)

        # test scale by short size
        sample = self.create_sample((1024, 512))
        rescale_short256 = SegRescale(short_size=256)
        out = rescale_short256(sample)
        self.assertEqual((512, 256), out["image"].size)

        # test short_size is stronger than long_size
        sample = self.create_sample((1024, 512))
        rescale_scale05 = SegRescale(short_size=301, long_size=301)
        out = rescale_scale05(sample)
        self.assertEqual((602, 301), out["image"].size)

    def test_rescale_with_long_size(self):
        # test raise exception for negative and zero short_size
        kwargs = {"long_size": 0}
        self.failUnlessRaises(ValueError, SegRescale, **kwargs)
        kwargs = {"long_size": -200}
        self.failUnlessRaises(ValueError, SegRescale, **kwargs)

        # test scale by long size
        sample = self.create_sample((1024, 512))
        rescale_long256 = SegRescale(long_size=256)
        out = rescale_long256(sample)
        self.assertEqual((256, 128), out["image"].size)

    def test_random_rescale(self):
        # test passing scales argument
        random_rescale = SegRandomRescale(scales=0.1)
        self.assertEqual((0.1, 1), random_rescale.scales)

        random_rescale = SegRandomRescale(scales=1.2)
        self.assertEqual((1, 1.2), random_rescale.scales)

        random_rescale = SegRandomRescale(scales=(0.5, 1.2))
        self.assertEqual((0.5, 1.2), random_rescale.scales)

        kwargs = {"scales": -0.5}
        self.failUnlessRaises(ValueError, SegRandomRescale, **kwargs)

        # test random rescale
        size = [1024, 512]
        scales = [0.8, 1.2]
        sample = self.create_sample(size)
        random_rescale = SegRandomRescale(scales=(0.8, 1.2))
        min_size = [scales[0] * s for s in size]
        max_size = [scales[1] * s for s in size]

        out = random_rescale(sample)
        for i in range(len(min_size)):
            self.assertGreaterEqual(out["image"].size[i], min_size[i])
            self.assertLessEqual(out["image"].size[i], max_size[i])

    def test_padding(self):
        # test arguments are valid
        pad = SegPadShortToCropSize(crop_size=200)
        self.assertEqual((200, 200), pad.crop_size)

        kwargs = {"crop_size": (0, 200)}
        self.failUnlessRaises(ValueError, SegPadShortToCropSize, **kwargs)

        kwargs = {"crop_size": 200, "fill_image": 256}
        self.failUnlessRaises(ValueError, SegPadShortToCropSize, **kwargs)

        kwargs = {"crop_size": 200, "fill_mask": 256}
        self.failUnlessRaises(ValueError, SegPadShortToCropSize, **kwargs)

        in_size = (512, 256)

        out_size = (512, 512)
        sample = self.create_sample(in_size)
        padding = SegPadShortToCropSize(crop_size=out_size)
        out = padding(sample)
        self.assertEqual(out_size, out["image"].size)

        # pad to odd size
        out_size = (512, 501)
        sample = self.create_sample(in_size)
        padding = SegPadShortToCropSize(crop_size=out_size)
        out = padding(sample)
        self.assertEqual(out_size, out["image"].size)

    def test_padding_fill_values(self):
        image_to_tensor = ToTensor()
        # test fill mask
        in_size = (256, 128)
        out_size = (256, 256)
        # padding fill values
        fill_mask_value = 32
        fill_image_value = 127

        sample = self.create_sample(in_size)
        padding = SegPadShortToCropSize(crop_size=out_size, fill_mask=fill_mask_value, fill_image=fill_image_value)
        out = padding(sample)

        out_mask = SegmentationDataSet.target_transform(out["mask"])
        # same as SegmentationDataset transform just without normalization to easily keep track of values.
        out_image = image_to_tensor(out["image"])

        # test transformed mask values
        original_values = out_mask[128 // 2 : -128 // 2].unique().tolist()
        pad_values = torch.cat([out_mask[: 128 // 2], out_mask[-128 // 2 :]], dim=0).unique().tolist()

        self.assertEqual(len(original_values), 1)
        self.assertEqual(original_values[0], self.default_mask_value)

        self.assertEqual(len(pad_values), 1)
        self.assertEqual(pad_values[0], fill_mask_value)

        # test transformed image values
        original_values = out_image[:, 128 // 2 : -128 // 2].unique().tolist()
        pad_values = torch.cat([out_image[:, : 128 // 2], out_image[:, -128 // 2 :]], dim=1).unique().tolist()

        self.assertEqual(len(original_values), 1)
        self.assertEqual(original_values[0], self.default_image_value)

        self.assertEqual(len(pad_values), 1)
        self.assertAlmostEqual(pad_values[0], fill_image_value / 255, delta=1e-5)

    def test_crop(self):
        # test arguments are valid
        pad = SegCropImageAndMask(crop_size=200, mode="center")
        self.assertEqual((200, 200), pad.crop_size)

        kwargs = {"crop_size": (0, 200), "mode": "random"}
        self.failUnlessRaises(ValueError, SegCropImageAndMask, **kwargs)
        # test unsupported mode
        kwargs = {"crop_size": (200, 200), "mode": "deci"}
        self.failUnlessRaises(ValueError, SegCropImageAndMask, **kwargs)

        in_size = (1024, 512)
        out_size = (128, 256)

        crop_center = SegCropImageAndMask(crop_size=out_size, mode="center")
        crop_random = SegCropImageAndMask(crop_size=out_size, mode="random")

        sample = self.create_sample(in_size)
        out_center = crop_center(sample)

        sample = self.create_sample(in_size)
        out_random = crop_random(sample)

        self.assertEqual(out_size, out_center["image"].size)
        self.assertEqual(out_size, out_random["image"].size)

    def test_rescale_padding(self):
        in_size = (1024, 512)
        out_size = (512, 512)
        sample = self.create_sample(in_size)

        transform = Compose([SegRescale(long_size=out_size[0]), SegPadShortToCropSize(crop_size=out_size)])  # rescale to (512, 256)  # pad to (512, 512)
        out = transform(sample)
        self.assertEqual(out_size, out["image"].size)

    def test_random_rescale_padding_random_crop(self):
        img_size = (1024, 512)
        crop_size = (256, 128)
        sample = self.create_sample(img_size)

        transform = Compose(
            [SegRandomRescale(scales=(0.1, 2.0)), SegPadShortToCropSize(crop_size=crop_size), SegCropImageAndMask(crop_size=crop_size, mode="random")]
        )

        out = transform(sample)
        self.assertEqual(crop_size, out["image"].size)


if __name__ == "__main__":
    unittest.main()
