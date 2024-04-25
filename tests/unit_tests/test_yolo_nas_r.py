import unittest

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from super_gradients.training.losses.yolo_nas_r_loss import cxcywhr_iou
from super_gradients.training.models.detection_models.yolo_nas_r.yolo_nas_r_post_prediction_callback import rboxes_nms, optimized_rboxes_nms
from super_gradients.training.utils.visualization.obb import OBBVisualization


class TestYoloNasR(unittest.TestCase):
    def test_cxcywhr_iou_convergence(self):
        x = torch.tensor([[9, 11, 10, 10, 0]]).float()
        x = torch.nn.Parameter(x)

        y = torch.tensor([[100, 128, 156, 64, 1]])
        optimizer = torch.optim.Adam([x], lr=0.1)

        for _ in range(20):
            for _ in range(50):
                optimizer.zero_grad()
                iou_loss = 1 - cxcywhr_iou(x, y, CIoU=True)
                l1_loss = torch.nn.functional.l1_loss(x[..., 0:2], y[..., 0:2])
                loss = l1_loss + iou_loss
                loss.backward()
                optimizer.step()

            image = np.zeros((256, 256, 3), dtype=np.uint8)
            image = OBBVisualization.draw_obb(
                image,
                np.concatenate([x.detach().cpu().numpy(), y.detach().cpu().numpy()], axis=0),
                None,
                np.array([0, 1]),
                ["Pred", "True"],
                [(0, 255, 0), (255, 0, 0)],
            )
            plt.figure()
            plt.imshow(image)
            plt.title(f"IOU: {cxcywhr_iou(x, y).item()} LOSS: {loss.item():.2f}")
            plt.tight_layout()
            plt.show()

    def test_cxcywhr_iou(self):
        boxes1 = torch.rand([2, 5])
        boxes2 = torch.rand([3, 5])
        iou = cxcywhr_iou(boxes1, boxes2)
        iou2 = cxcywhr_iou(boxes1, boxes1)
        print(iou)
        print(iou2)

    def test_rboxes_nms(self):
        boxes = torch.rand([2, 5])
        boxes[:, 2:] = torch.abs(boxes[:, 2:])
        scores = torch.rand([2])
        keep = rboxes_nms(boxes, scores, 0.5)
        print(keep)

    def test_optimized_rboxes_nms(self):
        boxes = torch.tensor(
            [
                [1, 1, 2, 2, 0],
                [10, 10, 10, 10, 1],
                [1, 1, 2, 2, 0],
            ]
        )

        keep1 = rboxes_nms(boxes, torch.tensor([0.8, 0.9, 0.3]), 0.5)
        keep2 = optimized_rboxes_nms(boxes, torch.tensor([0.8, 0.9, 0.3]), 0.5)
        print(keep1)
        print(keep2)

    def test_profile_nms(self):
        boxes = torch.randn([1024, 5])
        s = cv2.getTickCount()
        keep1 = rboxes_nms(boxes, torch.rand([1024]), 0.5)
        f = cv2.getTickCount()
        print((f - s) / cv2.getTickFrequency())

        boxes = torch.randn([1024, 5])
        s = cv2.getTickCount()
        keep2 = optimized_rboxes_nms(boxes, torch.rand([1024]), 0.5)
        f = cv2.getTickCount()
        print((f - s) / cv2.getTickFrequency())

        self.assertTrue(torch.all(keep1 == keep2))

        boxes = torch.randn([1024, 5]).cuda()
        s = cv2.getTickCount()
        keep1 = rboxes_nms(boxes, torch.rand([1024]).cuda(), 0.5)
        f = cv2.getTickCount()
        print((f - s) / cv2.getTickFrequency())

        boxes = torch.randn([1024, 5]).cuda()
        s = cv2.getTickCount()
        keep2 = optimized_rboxes_nms(boxes, torch.rand([1024]).cuda(), 0.5)
        f = cv2.getTickCount()
        print((f - s) / cv2.getTickFrequency())

        self.assertTrue(torch.all(keep1 == keep2))


if __name__ == "__main__":
    unittest.main()
