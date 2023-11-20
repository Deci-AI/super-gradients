import numpy as np
from torch import nn

from super_gradients.conversion.conversion_utils import torch_dtype_to_numpy_dtype
from super_gradients.conversion.conversion_enums import ExportTargetBackend, ExportQuantizationMode, DetectionOutputFormatMode


def build_preprocessing_hint_text(preprocessing_module: nn.Module) -> str:
    module_repr = repr(preprocessing_module)
    return f"""
Exported model already contains preprocessing (normalization) step, so you don't need to do it manually.
Preprocessing steps to be applied to input image are:
{module_repr}
"""


def build_postprocessing_hint_text(num_pre_nms_predictions, max_predictions_per_image, nms_threshold, confidence_threshold, output_predictions_format) -> str:
    return f"""
Exported model contains postprocessing (NMS) step with the following parameters:
    num_pre_nms_predictions={num_pre_nms_predictions}
    max_predictions_per_image={max_predictions_per_image}
    nms_threshold={nms_threshold}
    confidence_threshold={confidence_threshold}
    output_predictions_format={output_predictions_format}

"""


def build_usage_instructions_for_pose_estimation(
    *,
    output,
    batch_size,
    input_image_channels,
    input_image_shape,
    input_image_dtype,
    preprocessing,
    preprocessing_module,
    postprocessing,
    postprocessing_module,
    num_pre_nms_predictions,
    max_predictions_per_image,
    confidence_threshold,
    nms_threshold,
    output_predictions_format,
    engine: ExportTargetBackend,
    quantization_mode: ExportQuantizationMode,
) -> str:
    # Add usage instructions
    usage_instructions = f"""
Model exported successfully to {output}
Model expects input image of shape [{batch_size}, {input_image_channels}, {input_image_shape[0]}, {input_image_shape[1]}]
Input image dtype is {input_image_dtype}"""

    if preprocessing:
        preprocessing_hint_text = build_preprocessing_hint_text(preprocessing_module)
        usage_instructions += f"\n{preprocessing_hint_text}"

    if postprocessing:
        postprocessing_hint_text = build_postprocessing_hint_text(
            num_pre_nms_predictions, max_predictions_per_image, nms_threshold, confidence_threshold, output_predictions_format
        )
        usage_instructions += f"\n{postprocessing_hint_text}"

    if engine in (ExportTargetBackend.ONNXRUNTIME, ExportTargetBackend.TENSORRT):
        dtype_name = np.dtype(torch_dtype_to_numpy_dtype(input_image_dtype)).name

        usage_instructions += f"""
Exported model is in ONNX format and can be used with ONNXRuntime
To run inference with ONNXRuntime, please use the following code snippet:

    import onnxruntime
    import numpy as np
    session = onnxruntime.InferenceSession("{output}", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    inputs = [o.name for o in session.get_inputs()]
    outputs = [o.name for o in session.get_outputs()]

    example_input_image = np.zeros(({batch_size}, {input_image_channels}, {input_image_shape[0]}, {input_image_shape[1]})).astype(np.{dtype_name})
    predictions = session.run(outputs, {{inputs[0]: example_input_image}})

Exported model can also be used with TensorRT
To run inference with TensorRT, please see TensorRT deployment documentation
You can benchmark the model using the following code snippet:

    trtexec --onnx={output} {'--int8' if quantization_mode == ExportQuantizationMode.INT8 else '--fp16'} --avgRuns=100 --duration=15

"""

    if postprocessing is True:
        if output_predictions_format == DetectionOutputFormatMode.FLAT_FORMAT:
            usage_instructions += f"""
Exported model has predictions in {output_predictions_format} format:

# flat_predictions is a 2D array of [N,K] shape
# Each row represents (image_index, x_min, y_min, x_max, y_max, confidence, joints...)
# Please note all values are floats, so you have to convert them to integers if needed

[flat_predictions] = predictions"""

            if batch_size == 1:
                usage_instructions += """
pred_bboxes = flat_predictions[:, 1:5]
pred_scores = flat_predictions[:, 5]
pred_joints = flat_predictions[:, 6:].reshape((len(pred_bboxes), -1, 3))
for i in range(len(pred_bboxes)):
    confidence = pred_scores[i]
    x_min, y_min, x_max, y_max = pred_bboxes[i]
    print(f"Detected pose with confidence={{confidence}}, x_min={{x_min}}, y_min={{y_min}}, x_max={{x_max}}, y_max={{y_max}}")
    for joint_index, (x, y, confidence) in enumerate(pred_joints[i]):")
        print(f"Joint {{joint_index}} has coordinates x={{x}}, y={{y}}, confidence={{confidence}}")

"""

            else:
                usage_instructions += f"""
for current_sample in range({batch_size}):
    predictions_for_current_sample = predictions[predictions[0] == current_sample]
    print("Predictions for sample " + str(current_sample))
    pred_bboxes = predictions_for_current_sample[:, 1:5]
    pred_scores = predictions_for_current_sample[:, 5]
    pred_joints = predictions_for_current_sample[:, 6:].reshape((len(pred_bboxes), -1, 3))
    for i in range(len(pred_bboxes)):
        confidence = pred_scores[i]
        x_min, y_min, x_max, y_max = pred_bboxes[i]
        print(f"Detected pose with confidence={{confidence}}, x_min={{x_min}}, y_min={{y_min}}, x_max={{x_max}}, y_max={{y_max}}")
        for joint_index, (x, y, confidence) in enumerate(pred_joints[i]):
            print(f"Joint {{joint_index}} has coordinates x={{x}}, y={{y}}, confidence={{confidence}}")

"""

        elif output_predictions_format == DetectionOutputFormatMode.BATCH_FORMAT:
            # fmt: off
            usage_instructions += f"""Exported model has predictions in {output_predictions_format} format:

    num_detections, pred_boxes, pred_scores, pred_joints = predictions
    for image_index in range(num_detections.shape[0]):
        for i in range(num_detections[image_index,0]):
            confidence = pred_scores[image_index, i]
            x_min, y_min, x_max, y_max = pred_boxes[image_index, i]
            pred_joints = pred_joints[image_index, i]
            print(f"Detected pose with confidence={{confidence}}, x_min={{x_min}}, y_min={{y_min}}, x_max={{x_max}}, y_max={{y_max}}")
            for joint_index, (x, y, confidence) in enumerate(pred_joints[i]):
                print(f"Joint {{joint_index}} has coordinates x={{x}}, y={{y}}, confidence={{confidence}}")

"""
    elif postprocessing is False:
        usage_instructions += """Model exported with postprocessing=False
No decoding or NMS is added to the model, so you will have to decode predictions manually.
Please refer to the documentation for the model you exported"""
    elif isinstance(postprocessing_module, nn.Module):
        usage_instructions += f""""Exported model contains a custom postprocessing step.
We are unable to provide usage instructions to user-provided postprocessing module
But here is the human-friendly representation of the postprocessing module:
        {repr(postprocessing_module)}"""

    return usage_instructions
