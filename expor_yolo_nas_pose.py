from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training.models import convert_to_onnx

convert_to_onnx(
    models.get(Models.YOLO_NAS_POSE_S, num_classes=18),
    out_path="serket_yolo_nas_pose_s_1152x640.onnx",
    prep_model_for_conversion_kwargs=dict(input_size=(1, 3, 640, 1152)),
)
convert_to_onnx(
    models.get(Models.YOLO_NAS_POSE_M, num_classes=18),
    out_path="serket_yolo_nas_pose_m_1152x640.onnx",
    prep_model_for_conversion_kwargs=dict(input_size=(1, 3, 640, 1152)),
)
convert_to_onnx(
    models.get(Models.YOLO_NAS_POSE_L, num_classes=18),
    out_path="serket_yolo_nas_pose_l_1152x640.onnx",
    prep_model_for_conversion_kwargs=dict(input_size=(1, 3, 640, 1152)),
)
