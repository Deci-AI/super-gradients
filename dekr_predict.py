from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training.models import DEKRPoseEstimationModel

dekr: DEKRPoseEstimationModel = models.get(Models.DEKR_W32_NO_DC, pretrained_weights="coco_pose")
result = dekr.predict("team.png", conf=0.2)
result.show(keypoint_radius=8, show_confidence=False, joint_thickness=4)
result.save(".")
