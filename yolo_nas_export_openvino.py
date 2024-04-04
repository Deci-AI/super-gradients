import logging
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
from matplotlib import pyplot as plt
from super_gradients.common.object_names import Models
from super_gradients.conversion import ExportTargetBackend, ExportQuantizationMode
from super_gradients.inference import iterate_over_detection_predictions_in_batched_format
from super_gradients.module_interfaces import ExportableObjectDetectionModel
from super_gradients.training import models
from super_gradients.training.dataloaders import coco2017_val_yolo_nas
from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST
from super_gradients.training.utils.detection_utils import DetectionVisualization
from super_gradients.training.utils.media.image import load_image


# benchmark_app -m .\yolo_nas_openvino_int8.xml -api sync -d CPU -t 90
# benchmark_app -m .\yolo_nas_openvino_fp16.xml -api sync -d CPU -t 90
# benchmark_app -m .\yolo_nas_openvino_fp32.xml -api sync -d CPU -t 90


def main():
    common_export_params = dict(
        engine=ExportTargetBackend.OPENVINO,
        confidence_threshold=0.5,
        preprocessing=False,
        postprocessing=True,
    )
    logging.basicConfig(level=logging.DEBUG)
    m: ExportableObjectDetectionModel = models.get(Models.YOLO_NAS_S, pretrained_weights="coco").eval()

    Path("yolo_nas_openvino_fp32.xml").unlink(missing_ok=True)
    Path("yolo_nas_openvino_fp32.bin").unlink(missing_ok=True)
    m.export(output="yolo_nas_openvino_fp32.xml", **common_export_params)
    print("Exported yolo_nas_openvino_fp32")

    Path("yolo_nas_openvino_fp16.xml").unlink(missing_ok=True)
    Path("yolo_nas_openvino_fp16.bin").unlink(missing_ok=True)
    m.export(
        output="yolo_nas_openvino_fp16.xml",
        **common_export_params,
        quantization_mode=ExportQuantizationMode.FP16,
    )
    print("Exported yolo_nas_openvino_fp16")

    Path("yolo_nas_openvino_int8.xml").unlink(missing_ok=True)
    Path("yolo_nas_openvino_int8.bin").unlink(missing_ok=True)
    m.export(
        output="yolo_nas_openvino_int8.xml",
        **common_export_params,
        quantization_mode=ExportQuantizationMode.INT8,
        calibration_loader=coco2017_val_yolo_nas(dataset_params=dict(data_dir="G:/coco2017"), dataloader_params=dict(num_workers=0)),
        calibration_batches=500,
        quantization_skip_layers=[
            ".+reg_convs.+",
            ".+cls_convs.+",
            ".+cls_pred.+",
            ".+reg_pred.+",
        ],
    )
    print("Exported yolo_nas_openvino_int8")

    # device = "AUTO"
    # ov_config = {}
    # core = ov.Core()
    #
    # ov_model = core.read_model("yolo_nas_openvino_int8.xml")
    # compiled_ov_model = core.compile_model(ov_model, device, ov_config)
    #
    # image = load_image("https://deci-pretrained-models.s3.amazonaws.com/sample_images/beatles-abbeyroad.jpg")
    # image = cv2.resize(image, (640, 640))
    # model_input = np.moveaxis(image, -1, 0)[np.newaxis, ...] / 255.0
    #
    # request = compiled_ov_model.create_infer_request()
    # input_layer = compiled_ov_model.input(0)
    # request.infer(inputs={input_layer.any_name: model_input})
    # num_predictions = request.get_output_tensor(0).data
    # detected_bboxes = request.get_output_tensor(1).data
    # detected_scores = request.get_output_tensor(2).data
    # detected_labels = request.get_output_tensor(3).data
    # print(num_predictions)
    # predictions = (num_predictions, detected_bboxes, detected_scores, detected_labels)
    #
    # # Do something with predictions for image with index image_index
    # image = image.copy()
    # class_names = COCO_DETECTION_CLASSES_LIST
    # # color_mapping = DetectionVisualization._generate_color_mapping(len(class_names))
    #
    # image_index, pred_boxes, pred_scores, pred_classes = next(iter(iterate_over_detection_predictions_in_batched_format(predictions)))
    #
    # predicted_boxes = np.concatenate([pred_boxes, pred_scores[:, np.newaxis], pred_classes[:, np.newaxis]], axis=1)
    #
    # image = DetectionVisualization.visualize_image(image_np=np.array(image), class_names=class_names, pred_boxes=predicted_boxes)
    #
    # plt.figure(figsize=(8, 8))
    # plt.imshow(image)
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
