import unittest

import super_gradients
import torch
import os


class DeciDataSetIntegrationTest(unittest.TestCase):

    def setUp(self) -> None:
        super_gradients.init_trainer()
        self.class_to_test = COCODetectionDataSet

    @classmethod
    def setUpClass(cls) -> None:
        cls.coco_dataset_params = {"batch_size": 1,
                                   "test_batch_size": 1,
                                   "dataset_dir": "/data/coco/",
                                   "s3_link": None,
                                   "image_size": 416,
                                   "degrees": 1.98,  # image rotation (+/- deg)
                                   "translate": 0.05,  # image translation (+/- fraction)
                                   "scale": 0.05,  # image scale (+/- gain)
                                   "shear": 0.641,
                                   "hsv_h": 0.0138,  # image HSV-Hue augmentation (fraction)
                                   "hsv_s": 0.678,  # image HSV-Saturation augmentation (fraction)
                                   "hsv_v": 0.36,  # image HSV-Value augmentation (fraction)
                                   }

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test_coco_dataset_subclass_mosaic_loading_labels_cached(self, class_inclusion_list=['chair', 'dining table']):
        """

        Plots a single image with single bbox of an object from the sub class list, when in mosaic mode.
        @param class_inclusion_list: list(str) list of sub class names (from coco classes).
        @return:
        """
        test_batch_size = 64

        # TZAG COCO DATASET LOCATION
        coco_dataset = COCODetectionDataSet('/data/coco/', 'val2017.txt', batch_size=test_batch_size, img_size=640,
                                            dataset_hyper_params=self.coco_dataset_params,
                                            augment=True,
                                            cache_labels=True,
                                            cache_images=False,
                                            sample_loading_method='mosaic',
                                            class_inclusion_list=class_inclusion_list)

        self.assertTrue(len(coco_dataset) > 0)

        # LOAD DATA USING A DATA LOADER
        nw = min([os.cpu_count(), test_batch_size if test_batch_size > 1 else 0, 4])  # number of workers
        dataloader = torch.utils.data.DataLoader(coco_dataset,
                                                 batch_size=test_batch_size,
                                                 num_workers=nw,
                                                 shuffle=True,
                                                 # Shuffle=True unless rectangular training is used
                                                 pin_memory=True,
                                                 collate_fn=base_detection_collate_fn)

        plot_coco_datasaet_images_with_detections(dataloader, num_images_to_plot=1)

    def test_coco_dataset_subclass_integration_rectangular_loading_labels_cached(self, class_inclusion_list=['chair', 'dining table']):
        """

        Plots a single image with single bbox of an object from the sub class list, when in mosaic mode.
        @param class_inclusion_list: list(str) list of sub class names (from coco classes).
        @return:
        """
        test_batch_size = 64

        # TZAG COCO DATASET LOCATION
        coco_dataset = COCODetectionDataSet(
            '/data/coco/', 'val2017.txt', img_size=640, batch_size=test_batch_size,
            dataset_hyper_params=self.coco_dataset_params,
            cache_labels=True,
            cache_images=False,
            augment=False,
            sample_loading_method='rectangular',
            class_inclusion_list=class_inclusion_list
        )

        # LOAD DATA USING A DATA LOADER
        nw = min([os.cpu_count(), test_batch_size if test_batch_size > 1 else 0, 4])  # number of workers
        dataloader = torch.utils.data.DataLoader(coco_dataset,
                                                 batch_size=test_batch_size,
                                                 num_workers=nw,
                                                 pin_memory=True,
                                                 collate_fn=base_detection_collate_fn
                                                 )

        plot_coco_datasaet_images_with_detections(dataloader, num_images_to_plot=1)


if __name__ == '__main__':
    unittest.main()
