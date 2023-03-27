from abc import ABC, abstractmethod
from typing import Dict, List

import torch

from super_gradients.training.models.detection_models.yolo_base import SgDetectionModule
from super_gradients.training.transforms.reversable_image_processors import (
    ReversibleDetectionProcessor,
    ReversibleDetectionPadToSize,
    ReversibleDetectionPaddedRescale,
    ReversibleDetectionNormalize,
    ReversibleDetectionImagePermute,
)
from super_gradients.training.pipelines.predictions import Prediction
from super_gradients.training.models import YoloBase, PPYoloE


class Pipeline(ABC):
    @abstractmethod
    def __call__(self, image) -> Prediction:
        pass


class DetectionPipeline(Pipeline):
    def __init__(self, model: SgDetectionModule, image_processors: List[ReversibleDetectionProcessor], post_prediction_processor: callable = None):
        self.model = model
        self.image_processors = image_processors
        self.post_prediction_processor = post_prediction_processor
        super().__init__()

    def __call__(self, image) -> Prediction:
        from super_gradients.training.utils.load_image import load_image

        original_image = load_image(image)
        np_image = original_image.copy()

        for image_processor in self.image_processors:
            image_processor.calibrate(np_image)
            np_image = image_processor.apply_to_image(np_image)

        model_input = torch.Tensor(np_image).unsqueeze(0)  # .to(self.model.device)
        model_outputs = self.model(model_input)

        if self.post_prediction_processor:
            model_outputs = self.post_prediction_processor(model_outputs, device=model_input.device)
            model_outputs = model_outputs or torch.zeros((0, 5), dtype=torch.float32)

        np_output = model_outputs[0].detach().cpu().numpy()
        for image_processor in self.image_processors[::-1]:
            np_output = image_processor.apply_reverse_to_targets(np_output)

        return Prediction(_image=original_image, _boxes=np_output[:4], _classes=np_output[4], _scores=np_output[5])

    @classmethod
    def from_pretrained(cls, model: SgDetectionModule, iou: float = 0.65, conf: float = 0.01):
        """Instantiates a DetectionPipeline using a pretrained model. This is only supported for models pretrained by SuperGradients."""

        image_processors = None
        for model_class, _image_processors in MODELS_PROCESSORS.items():
            if isinstance(model, model_class):
                image_processors = _image_processors
        if image_processors is None:
            raise ValueError(f"Model {cls} is not supported by this pipeline.")

        post_prediction_processor = model.get_post_prediction_callback(iou=iou, conf=conf)
        return cls(model=model, image_processors=image_processors, post_prediction_processor=post_prediction_processor)


# TODO: Find a way to map this with checkpoints...
# Map models classes to image processors required to run the model
MODELS_PROCESSORS: Dict[type, List[ReversibleDetectionProcessor]] = {
    YoloBase: [
        ReversibleDetectionPaddedRescale(target_size=(640, 640), swap=(2, 0, 1)),
    ],
    PPYoloE: [
        ReversibleDetectionPadToSize(output_size=(640, 640), pad_value=0),
        ReversibleDetectionNormalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
        ReversibleDetectionImagePermute(permutation=(2, 0, 1)),
    ],
}
