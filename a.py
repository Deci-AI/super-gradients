import os
from super_gradients.common.object_names import Models
from super_gradients.training import models

os.environ["UPLOAD_LOGS"] = "FALSE"


if __name__ == "__main__":

    model = models.get(Models.YOLO_NAS_POSE_S, num_classes=17)
    model.export("2023_09_23_coco_yolo_nas_pose_s_1x640x640.onnx", input_image_shape=(640, 640), preprocessing=False, postprocessing=False, onnx_simplify=True)

    model = models.get(Models.YOLO_NAS_POSE_M, num_classes=17)
    model.export("2023_09_23_coco_yolo_nas_pose_m_1x640x640.onnx", input_image_shape=(640, 640), preprocessing=False, postprocessing=False, onnx_simplify=True)

    model = models.get(Models.YOLO_NAS_POSE_L, num_classes=17)
    model.export("2023_09_23_coco_yolo_nas_pose_l_1x640x640.onnx", input_image_shape=(640, 640), preprocessing=False, postprocessing=False, onnx_simplify=True)
