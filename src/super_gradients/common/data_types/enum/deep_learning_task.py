from enum import Enum


class DeepLearningTask(str, Enum):
    CLASSIFICATION = "classification"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    OBJECT_DETECTION = "object_detection"
    DEPTH_ESTIMATION = "depth_estimation"
    POSE_ESTIMATION = "pose_estimation"
    NLP = "nlp"
    OTHER = "other"
