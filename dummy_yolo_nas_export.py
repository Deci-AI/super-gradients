import logging

import numpy as np

# import onnx
# from super_gradients.training.dataloaders import coco2017_val
from super_gradients.training.models import CustomizableDetector
from super_gradients.training.utils.export_utils import infer_image_shape_from_model

import onnxruntime


def main():
    logging.getLogger().setLevel(logging.DEBUG)
    import super_gradients

    yolo_nas: CustomizableDetector = super_gradients.training.models.get("yolo_nas_s", pretrained_weights="coco").cuda()
    print(infer_image_shape_from_model(yolo_nas))

    yolo_nas.export(
        "yolo_nas_s.onnx",
        image_shape=(640, 640),
        preprocessing=False,
        postprocessing=True,
        quantize=False,
    )
    # print(onnx.load("yolo_nas_s.onnx"))

    session = onnxruntime.InferenceSession("yolo_nas_s.onnx")
    inputs = [o.name for o in session.get_inputs()]
    outputs = [o.name for o in session.get_outputs()]
    result = session.run(outputs, {inputs[0]: np.random.rand(1, 3, 640, 640).astype(np.float32)})
    print(result, result[0].shape)

    #
    # yolo_nas.export(
    #     "yolo_nas_s_quantized.onnx",
    #     image_shape=(640,640),
    #     preprocessing=False,
    #     postprocessing=True,
    #     quantize=True,
    # )
    # session = onnxruntime.InferenceSession("yolo_nas_s_quantized.onnx")
    # outputs = [o.name for o in session.get_outputs()]
    # result = session.run(outputs, np.random.rand(1,3,640,640).astype(np.float32))
    # print(result)
    #
    # calibration_loader = coco2017_val(dataset_params=dict(data_dir="e:/coco2017"), dataloader_params=dict(num_workers=0))
    # yolo_nas.export(
    #     "yolo_nas_s_quantized.onnx",
    #     image_shape=(640,640),
    #     preprocessing=False,
    #     postprocessing=True,
    #     quantize=True,
    #     calibration_loader=calibration_loader
    # )
    #
    # session = onnxruntime.InferenceSession("yolo_nas_s_quantized.onnx")
    # outputs = [o.name for o in session.get_outputs()]
    # result = session.run(outputs, np.random.rand(1,3,640,640).astype(np.float32))
    # print(result)


if __name__ == "__main__":
    main()
