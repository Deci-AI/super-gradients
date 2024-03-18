from super_gradients.training.utils.detection_utils import DetectionPostPredictionCallback


class IdentityPostPredictionCallback(DetectionPostPredictionCallback):
    def forward(self, p, device=None):
        return [p]
