from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training.models import DEKRPoseEstimationModel

dekr: DEKRPoseEstimationModel = models.get(Models.DEKR_W32_NO_DC, pretrained_weights="coco_pose")
dekr.predict("team.png").show()
