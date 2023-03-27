from abc import ABC, abstractmethod

from super_gradients.training.transforms.transforms import rescale_and_pad_to_size


class ImageProcessor(ABC):
    @abstractmethod
    def preprocess_image(self, image):
        pass

    @abstractmethod
    def postprocess_preds(self, raw_predictions):
        pass


class DetectionImageProcessor(ImageProcessor):
    @abstractmethod
    def preprocess_image(self, image):
        pass

    @abstractmethod
    def postprocess_preds(self, raw_predictions):
        pass


class RescalePadDetection(DetectionImageProcessor):
    def __init__(self, target_size=(640, 640), swap=(2, 0, 1)):
        # Input params
        self.target_size = target_size
        self.swap = swap

        # State
        self.r = None

    def preprocess_image(self, image):
        if self.r is not None:
            raise RuntimeError("ImageProcessor.preprocess can only be used once. Please create a new ImageProcessor instance.")

        image, r = rescale_and_pad_to_size(image, input_size=self.target_size, swap=self.swap)
        self.r = r
        return image

    def postprocess_pred(self, pred, bbox_format="xyxy"):
        # TODO: Think if we need to hande cases where bbox_format is not xyxy after nms.
        pred = pred.detach().cpu().numpy()
        pred[:, :4] = pred[:, :4] / self.r  # TODO: check if this is correct
        return pred

    def postprocess_preds(self, preds):
        if preds == [None]:
            return []
        return [self.postprocess_pred(pred) for pred in preds]
