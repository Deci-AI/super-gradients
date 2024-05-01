import torch
import torch.nn as nn
import torchvision
from typing import List

from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.processing.processing import default_yolo_nas_coco_processing_params


class SlidingWindowInferenceWrapper(nn.Module):
    def __init__(self, model, tile_size, tile_step, post_prediction_callback, nms_threshold=0.65, max_predictions_per_image=300, nms_top_k=1000):
        super().__init__()
        self.model = model
        self.tile_size = tile_size
        self.tile_step = tile_step
        self.post_prediction_callback = post_prediction_callback
        self.nms_threshold = nms_threshold
        self.max_predictions_per_image = max_predictions_per_image
        self.nms_top_k = nms_top_k

    def _filter_max_predictions(self, res: List) -> List:
        res[:] = [im[: self.max_predictions_per_image] if (im is not None and im.shape[0] > self.max_predictions_per_image) else im for im in res]

        return res

    def forward_sliding_window(self, images):
        batch_size, _, _, _ = images.shape
        all_detections = [[] for _ in range(batch_size)]  # Create a list for each image in the batch

        # Generate and process each tile
        for img_idx in range(batch_size):
            single_image = images[img_idx : img_idx + 1]  # Extract each image
            tiles = self._generate_tiles(single_image, self.tile_size, self.tile_step)
            for tile, (start_x, start_y) in tiles:
                tile_detections = self.model(tile)
                # Apply local NMS using post_prediction_callback
                tile_detections = self.post_prediction_callback(tile_detections)
                # Adjust detections to global image coordinates
                for img_i_tile_detections in tile_detections:
                    if len(img_i_tile_detections) > 0:
                        img_i_tile_detections[:, :4] += torch.tensor([start_x, start_y, start_x, start_y], device=tile.device)
                        all_detections[img_idx].append(img_i_tile_detections)

        # Concatenate and apply global NMS for each image's detections
        final_detections = []
        for detections in all_detections:
            if detections:
                detections = torch.cat(detections, dim=0)
                # Apply global NMS
                pred_bboxes = detections[:, :4]
                pred_cls_conf = detections[:, 4]
                pred_cls_label = detections[:, 5]

                if pred_cls_conf.size(0) > self.nms_top_k:
                    topk_candidates = torch.topk(pred_cls_conf, k=self.nms_top_k, largest=True)
                    pred_cls_conf = pred_cls_conf[topk_candidates.indices]
                    pred_cls_label = pred_cls_label[topk_candidates.indices]
                    pred_bboxes = pred_bboxes[topk_candidates.indices, :]

                idx_to_keep = torchvision.ops.boxes.batched_nms(boxes=pred_bboxes, scores=pred_cls_conf, idxs=pred_cls_label, iou_threshold=self.nms_threshold)

                final_detections.append(detections[idx_to_keep])
            else:
                final_detections.append(torch.empty(0, 6).to(images.device))  # Empty tensor for images with no detections

        if self.max_predictions_per_image is not None:
            final_detections = self._filter_max_predictions(final_detections)
        return final_detections

    def _generate_tiles(self, image, tile_size, tile_step):
        _, _, h, w = image.shape
        tiles = []
        for y in range(0, h - tile_size + 1, tile_step):
            for x in range(0, w - tile_size + 1, tile_step):
                tile = image[:, :, y : y + tile_size, x : x + tile_size]
                tiles.append((tile, (x, y)))
        return tiles


if __name__ == "__main__":
    from super_gradients.training.models import get
    from super_gradients.common.object_names import Models
    import os
    from super_gradients.training.dataloaders import coco2017_val_yolo_nas
    from super_gradients.training.utils.detection_utils import DetectionVisualization
    import cv2

    data_dir = os.environ.get("SUPER_GRADIENTS_COCO_DATASET_DIR", "/data/coco")

    dl = coco2017_val_yolo_nas(dataset_params=dict(data_dir=data_dir), dataloader_params=dict(batch_size=4))
    x, y, _ = next(iter(dl))
    # x_repeat = torch.zeros((4,3,1280,1280))
    # x_repeat[:, :, 0:640, 0:640] = x
    # x_repeat[:, :, 640:1280, 0:640] = x
    # x_repeat[:, :, 640: 1280, 640:1280] = x
    # x_repeat[:, :, 0: 640, 640:1280] = x\
    input_dim = [1280, 1280]
    img = cv2.imread("/home/shay.aharon/cars-for-sale-parking-sale-4f07c1178051f8b82c8bbc640fb3c27d.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # r = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
    # desired_size = (int(img.shape[1] * r), int(img.shape[0] * r))
    # img = cv2.resize(src=img, dsize=desired_size, interpolation=cv2.INTER_LINEAR).astype(np.uint8)

    pp = default_yolo_nas_coco_processing_params()
    processor = pp["image_processor"]

    # Switch from HWC to CHW
    # img_chw = np.transpose(img_rgb, (2, 0, 1))

    # Convert to tensor
    # img_tensor = torch.from_numpy(img_chw).float() / 255.

    # Unsqueeze to add the batch dimension
    img_tensor, _ = processor.preprocess_image(img)
    img_tensor = torch.from_numpy(img_tensor)
    img_tensor = img_tensor.unsqueeze(0)
    model = get(Models.YOLO_NAS_S, pretrained_weights="coco")
    ppcb = PPYoloEPostPredictionCallback(score_threshold=0.25, nms_top_k=1000, max_predictions=300, nms_threshold=0.7)
    sm = SlidingWindowInferenceWrapper(model, 640, 64, post_prediction_callback=ppcb)
    out_sliding_window = sm(img_tensor)
    DetectionVisualization.visualize_batch(
        image_tensor=img_tensor,
        pred_boxes=out_sliding_window,
        target_boxes=y,
        batch_name="640_tile_64_step_on_cars_large_1280_bgr_no_cv2resize_fix",
        class_names=COCO_DETECTION_CLASSES_LIST,
        checkpoint_dir="/home/shay.aharon/sw_outputs",
    )

    # Example of how to set up and use the SlidingWindowInferenceWrapper:
    # from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
    #
    # nms_callback = PPYoloEPostPredictionCallback(score_threshold=0.3, nms_threshold=0.5, nms_top_k=200, max_predictions=100)
    # sw_inference = SlidingWindowInferenceWrapper(ppyoloe_model, tile_size=512, tile_step=256,
    #                                              post_prediction_callback=nms_callback)
    #
    # # Forward an image through the sliding window inference wrapper
    # image = torch.rand(1, 3, 1024, 1024)  # Example image tensor
    # result = sw_inference(image)
