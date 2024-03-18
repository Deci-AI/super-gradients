from dataclasses import dataclass


@dataclass
class UnstructuredODModelConfig:
    model_name: str
    num_classes: int
    checkpoint_repo_id: str
    checkpoint_filename: str


UnstructuredYoloXMAR24_1_1_Config = UnstructuredODModelConfig(
    model_name="yolox_mar24_1_1", num_classes=19, checkpoint_repo_id="unstructuredio/yolox_l_unstructured_mar24_1", checkpoint_filename="checkpoint_best.pth"
)
UnstructuredYoloXMAR24_2_1_Config = UnstructuredODModelConfig(
    model_name="yolox_mar24_2_1", num_classes=19, checkpoint_repo_id="unstructuredio/yolox_l_unstructured_mar24_2", checkpoint_filename="checkpoint_best.pth"
)


MODEL_CONFIGS = {"yolox_mar24_1_1": UnstructuredYoloXMAR24_1_1_Config, "yolox_mar24_2_1": UnstructuredYoloXMAR24_2_1_Config}
