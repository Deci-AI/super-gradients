from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any

import numpy as np
import torch

from super_gradients.training.models.sg_module import SgModule
from super_gradients.training.utils.load_image import load_image
from super_gradients.training.models.predictions import Prediction, ClassificationPrediction, SegmentationPrediction, DetectionPrediction
from super_gradients.training.transforms.processing import Processing


class Pipeline(ABC):
    def __init__(self, model: SgModule, image_processor: Optional[Processing] = None):
        super().__init__()
        self.model = model
        self.image_processor = image_processor or get_model_image_processor(model)

    @abstractmethod
    def __call__(self, image: torch.Tensor) -> Prediction:
        """Apply the pipeline and return a prediction object of the relevant Task."""
        pass

    def _run(self, image) -> Tuple[np.ndarray, Any]:
        """Run the pipeline and return (image, predictions)"""
        original_image = load_image(image)

        np_image, processing_metadata = self.image_processor.preprocess_image(image=original_image.copy())

        model_input = torch.Tensor(np_image).unsqueeze(0)
        raw_output = self.model(model_input)

        model_outputs = self.decode_model_raw_prediction(raw_output)

        np_output = model_outputs[0].detach().cpu().numpy()

        np_output = self.image_processor.postprocess_predictions(predictions=np_output, metadata=processing_metadata)

        return original_image, np_output

    @abstractmethod
    def decode_model_raw_prediction(self, raw_predictions: torch.Tensor) -> torch.Tensor:
        """Decode the raw predictions from the model into a normal format."""
        pass


class ClassificationPipeline(Pipeline):
    def __init__(self, model: SgModule, image_processor: Optional[Processing] = None):
        super().__init__(model=model, image_processor=image_processor)

    def __call__(self, image: torch.Tensor) -> ClassificationPrediction:
        image, predictions = self._run(image)
        # TODO: Find a way to handle different datasets...
        return ClassificationPrediction(image=image, _class=predictions, class_names=[])

    def decode_model_raw_prediction(self, raw_predictions: torch.Tensor) -> torch.Tensor:
        return raw_predictions


class SegmentationPipeline(Pipeline):
    def __init__(self, model: SgModule, image_processor: Optional[Processing] = None):
        super().__init__(model=model, image_processor=image_processor)

    def __call__(self, image: torch.Tensor) -> SegmentationPrediction:
        image, predictions = self._run(image)
        # TODO: Find a way to handle different datasets...
        return SegmentationPrediction(image=image, _mask=predictions, class_names=[])

    def decode_model_raw_prediction(self, raw_predictions: torch.Tensor) -> torch.Tensor:
        return raw_predictions.argmax(dim=1).astype(np.uint8)


class DetectionPipeline(Pipeline):
    def __init__(
        self,
        model: SgModule,
        class_names: List[str],
        post_prediction_callback,
        image_processor: Optional[Processing] = None,
    ):
        super().__init__(model=model, image_processor=image_processor)
        self.class_names = class_names  # COCO_DETECTION_CLASSES_LIST
        self.post_prediction_callback = post_prediction_callback

    def __call__(self, image: torch.Tensor) -> DetectionPrediction:
        image, predictions = self._run(image)
        return DetectionPrediction(
            image=image,
            _boxes=predictions[:4],
            _classes=predictions[4],
            _scores=predictions[5],
            class_names=self.class_names,
        )

    def decode_model_raw_prediction(self, raw_predictions) -> torch.Tensor:
        """Decode the raw predictions from the model into a normal format."""
        decoded_predictions = self.post_prediction_callback(raw_predictions, device="cpu")  # TODO: add device
        if decoded_predictions == [None]:  # TODO: Support batch
            return torch.zeros((0, 5), dtype=torch.float32)
        return decoded_predictions


def get_model_image_processor(model: SgModule) -> Processing:
    if hasattr(model, "image_processor"):
        return model.image_processor
    raise ValueError(f"Model {model.__call__} is not supported by this pipeline.")


# MODELS_PROCESSORS: Dict[type, Processing] = {
#     YoloBase: DetectionPaddedRescale(output_size=(640, 640), swap=(2, 0, 1)),
#     PPYoloE: ComposeProcessing(
#         [
#             DetectionPadToSize(output_size=(640, 640), pad_value=0),
#             NormalizeImage(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
#             ImagePermute(permutation=(2, 0, 1)),
#         ]
#     ),
#     DDRNetCustom: ComposeProcessing(
#         [
#             SegmentationRescale(output_shape=(480, 320)),
#             NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ImagePermute(permutation=(2, 0, 1)),
#         ]
#     ),
# }
