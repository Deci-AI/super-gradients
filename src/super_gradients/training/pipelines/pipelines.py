from abc import ABC, abstractmethod

import torch

from super_gradients.training.models.detection_models.yolo_base import SgDetectionModule
from super_gradients.training.pipelines.image_processors import ImageProcessor, RescalePadDetection
from super_gradients.training.pipelines.predictions import Prediction


class Pipeline(ABC):
    def __init__(self, model, image_processor: ImageProcessor, post_prediction_processor: callable = None):
        self.model = model
        self.image_processor = image_processor
        self.post_prediction_processor = post_prediction_processor

    @abstractmethod
    def __call__(self, image) -> Prediction:
        pass

    def _predict(self, image):
        from super_gradients.training.utils.load_image import load_image

        image = load_image(image)

        model_input = self.image_processor.preprocess_image(image)

        model_input = torch.Tensor(model_input).unsqueeze(0)  # .to(self.model.device)
        model_outputs = self.model(model_input)

        # TODO: Find a way to make sure every post_prediction_processor returns xyxy format for bboxes
        if self.post_prediction_processor:
            model_outputs = self.post_prediction_processor(model_outputs)

        model_outputs = self.image_processor.postprocess_preds(model_outputs)  # TODO: This should be skiped for classification

        return image, model_outputs

    #
    # - DetectionNormalize:
    #     mean: [ 123.675, 116.28, 103.53 ]
    #     std: [ 58.395,  57.12,  57.375 ]


class DetectionPipeline(Pipeline):
    def __init__(self, model: SgDetectionModule, iou=0.65, conf=0.01):

        super().__init__(
            model=model,
            image_processor=RescalePadDetection(),
            post_prediction_processor=model.get_post_prediction_callback(iou=iou, conf=conf),
        )

    def __call__(self, image) -> Prediction:
        image, model_outputs = self._predict(image)
        single_output = model_outputs[0]
        return Prediction(_image=image, _boxes=single_output[:4], _classes=single_output[4], _scores=single_output[5])
