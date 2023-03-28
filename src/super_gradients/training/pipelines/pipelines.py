from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any, Union
from contextlib import contextmanager

import numpy as np
import torch

from super_gradients.training.utils.load_image import load_images, ImageType
from super_gradients.training.utils.detection_utils import DetectionPostPredictionCallback
from super_gradients.training.models.sg_module import SgModule
from super_gradients.training.models.predictions import Predictions, DetectionPredictions
from super_gradients.training.transforms.processing import Processing, ComposeProcessing


@contextmanager
def eval_mode(model: torch.nn.Module) -> None:
    """Set a model in evaluation mode and deactivate gradient computation, undo at the end.

    :param model: The model to set in evaluation mode.
    """
    _starting_mode = model.training
    model.eval()
    with torch.no_grad():
        yield
    model.train(mode=_starting_mode)


class Pipeline(ABC):
    def __init__(self, model: SgModule, image_processor: Union[Processing, List[Processing]], device: Optional[str] = "cpu"):
        super().__init__()
        self.model = model
        self.device = device

        if isinstance(image_processor, list):
            image_processor = ComposeProcessing(image_processor)
        self.image_processor = image_processor

    @abstractmethod
    def __call__(self, image: Union[ImageType, List[ImageType]]) -> Predictions:
        """Apply the pipeline and return a prediction object of the relevant Task."""
        pass

    def _run(self, images: Union[ImageType, List[ImageType]]) -> Tuple[List[np.ndarray], List[Any]]:
        """Run the pipeline and return (image, predictions). The pipeline is made of 4 steps:
        1. Load images - Loading the images into a list of numpy arrays.
        2. Preprocess - Encode the image in the shape/format expected by the model
        3. Predict - Run the model on the preprocessed image
        4. Postprocess - Decode the output of the model so that the predictions are in the shape/format of original image.
        """

        images = load_images(images)

        # Preprocess
        preprocessed_images, processing_metadatas = [], []
        for image in images:
            preprocessed_image, processing_metadata = self.image_processor.preprocess_image(image=image.copy())
            preprocessed_images.append(preprocessed_image)
            processing_metadatas.append(processing_metadata)

        # Predict
        with eval_mode(self.model):
            torch_inputs = torch.Tensor(np.array(preprocessed_images))
            raw_model_output = self.model(torch_inputs)
            torch_predictions = self.decode_model_raw_prediction(raw_model_output)

        # Postprocess
        predictions = []
        for torch_prediction, processing_metadata in zip(torch_predictions, processing_metadatas):
            prediction = torch_prediction.detach().cpu().numpy()
            prediction = self.image_processor.postprocess_predictions(predictions=prediction, metadata=processing_metadata)
            predictions.append(prediction)

        return images, predictions

    @abstractmethod
    def decode_model_raw_prediction(self, raw_model_output: Any) -> torch.Tensor:
        """Decode the raw predictions from the model into a normal format."""
        pass


class DetectionPipeline(Pipeline):
    def __init__(
        self,
        model: SgModule,
        class_names: List[str],
        post_prediction_callback: DetectionPostPredictionCallback,
        device: Optional[str] = "cpu",
        image_processor: Optional[Processing] = None,
    ):
        super().__init__(model=model, device=device, image_processor=image_processor)
        self.post_prediction_callback = post_prediction_callback
        self.class_names = class_names

    def __call__(self, images: torch.Tensor) -> DetectionPredictions:
        images, predictions = self._run(images=images)
        return DetectionPredictions(images=images, predictions=predictions, class_names=self.class_names)

    def decode_model_raw_prediction(self, raw_predictions) -> List[torch.Tensor]:
        """Decode the raw predictions from the model into a normal format."""
        decoded_predictions = self.post_prediction_callback(raw_predictions, device="cpu")  # TODO: add device
        decoded_predictions = [
            decoded_prediction if decoded_prediction is not None else torch.zeros((0, 6), dtype=torch.float32) for decoded_prediction in decoded_predictions
        ]
        return decoded_predictions


# TODO: Add PPYOLE

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
