from super_gradients.training.models.detection_models.pp_yolo_e.post_prediction_callback import PPYoloEPostPredictionCallback
from super_gradients.training import models
from super_gradients.training import dataloaders
from super_gradients.training.metrics.detection_metrics import DetectionMetrics
from tqdm import tqdm

model = models.get("ppyoloe_s", pretrained_weights="coco")
model.eval()

post_prediction_callback = PPYoloEPostPredictionCallback(score_threshold=0.25, nms_threshold=0.7, max_predictions=30, nms_top_k=300)


dataset = dataloaders.coco2017_val(dataset_params={"max_num_samples": 1280})
metric = DetectionMetrics(post_prediction_callback=post_prediction_callback, num_cls=80)


for images, labels, _ in tqdm(dataset):
    preds = model(images)
    metric.update(preds=preds, target=labels, device="cpu", inputs=images)

print(metric.compute())
