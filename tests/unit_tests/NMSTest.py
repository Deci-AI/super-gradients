
import unittest
import urllib

import torch
from PIL import Image
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import CoCoDetectionDatasetInterface
import torchvision.transforms as transforms

from src.super_gradients.training.utils.detection_utils import non_max_suppression, crowd_detection_collate_fn
from super_gradients import SgModel
from super_gradients.training import MultiGPUMode


class NMSTest(unittest.TestCase):
    def setUp(self):
        urllib.request.urlretrieve('https://ultralytics.com/images/zidane.jpg', "zidane.jpg")
        self.example_image = Image.open('zidane.jpg')

    def test_something(self):
        trainer = SgModel('coco_pretrained_yolov5s', model_checkpoints_location='local',
                          multi_gpu=MultiGPUMode.OFF)

        trainer.build_model("yolo_v5s", arch_params={'num_classes': 80},
                            checkpoint_params={"pretrained_weights": "coco"})

        transform = transforms.Compose([
            transforms.PILToTensor(),
        ])

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.example_image = self.example_image.resize((384, 576))
        example_image: torch.Tensor = transform(self.example_image)
        example_image = example_image.type(torch.FloatTensor)
        example_image = torch.stack([example_image] * 8, dim=0)
        example_image = example_image.to(device)
        trainer.net.eval()
        res = trainer.net(example_image)
        nms_res = non_max_suppression(res[0], conf_thres=0.7, iou_thres=0.8)



if __name__ == '__main__':
    unittest.main()
