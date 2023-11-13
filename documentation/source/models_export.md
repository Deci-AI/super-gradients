# This tutorial shows how to export SG models to ONNX format for deployment to ONNX-compatible runtimes and accelerators.


From this tutorial you will learn:

* How to export Object Detection model to ONNX and it with ONNXRuntime / TensorRT
* How to enable FP16 / INT8 quantization and export a model with calibration
* How to customize NMS parameters and number of detections per image
* How to choose whether to use TensorRT or ONNXRuntime as a backend

## New Export API

A new export API is introduced in SG 3.2.0. It is aimed to simplify the export process and allow end-to-end export of SG models to ONNX format with a single line of code.

### Currently supported models

- YoloNAS
- PPYoloE

### Supported features

- Exporting a model to OnnxRuntime and TensorRT
- Exporting a model with preprocessing (e.g. normalizing/standardizing image according to normalization parameters during training)
- Exporting a model with postprocessing (e.g. predictions decoding and NMS) - you obtain the ready-to-consume bounding box outputs
- FP16 / INT8 quantization support with calibration
- Pre- and post-processing steps can be customized by the user if needed
- Customising input image shape and batch size
- Customising NMS parameters and number of detections per image
- Customising output format (flat or batched)


```python
!pip install -qq super_gradients==3.4.0
```

### Minimalistic export example

Let start with the most simple example of exporting a model to ONNX format.
We will use YoloNAS-S model in this example. All models that suports new export API now expose a `export()` method that can be used to export a model. There is one mandatory argument that should be passed to the `export()` method - the path to the output file. Currently, only `.onnx` format is supported, but we may add support for CoreML and other formats in the future.


```python
from super_gradients.common.object_names import Models
from super_gradients.training import models

model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")

export_result = model.export("yolo_nas_s.onnx")
```

A lot of work just happened under the hood:

* A model was exported to ONNX format using default batch size of 1 and input image shape that was used during training
* A preprocessing and postprocessing steps were attached to ONNX graph
* For pre-processing step, the normalization parameters were extracted from the model itself (to be consistent with the image normalization and channel order used during training)
* For post-processing step, the NMS parameters were also extracted from the model and NMS module was attached to the graph
* ONNX graph was checked and simplified to improve compatibility with ONNX runtimes.

A returned value of `export()` method is an instance of `ModelExportResult` class. 
First of all it serves the purpose of storing all the information about the exported model in a single place. 
It also provides a convenient way to get an example of running the model and getting the output:


```python
export_result
```




    Model exported successfully to yolo_nas_s.onnx
    Model expects input image of shape [1, 3, 640, 640]
    Input image dtype is torch.uint8
    Exported model already contains preprocessing (normalization) step, so you don't need to do it manually.
    Preprocessing steps to be applied to input image are:
    Sequential(
      (0): CastTensorTo(dtype=torch.float32)
      (1): ApplyMeanStd(mean=[0.], scale=[255.])
    )
    
    Exported model contains postprocessing (NMS) step with the following parameters:
        num_pre_nms_predictions=1000
        max_predictions_per_image=1000
        nms_threshold=0.7
        confidence_threshold=0.25
        output_predictions_format=batch
    
    Exported model is in ONNX format and can be used with ONNXRuntime
    To run inference with ONNXRuntime, please use the following code snippet:
    
        import onnxruntime
        import numpy as np
        session = onnxruntime.InferenceSession("yolo_nas_s.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        inputs = [o.name for o in session.get_inputs()]
        outputs = [o.name for o in session.get_outputs()]
        example_input_image = np.zeros((1, 3, 640, 640)).astype(np.uint8)
        predictions = session.run(outputs, {inputs[0]: example_input_image})
    
    Exported model has predictions in batch format:
    
        num_detections, pred_boxes, pred_scores, pred_classes = predictions
        for image_index in range(num_detections.shape[0]):
          for i in range(num_detections[image_index,0]):
            class_id = pred_classes[image_index, i]
            confidence = pred_scores[image_index, i]
            x_min, y_min, x_max, y_max = pred_boxes[image_index, i]
            print(f"Detected object with class_id={class_id}, confidence={confidence}, x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")



That's it. You can now use the exported model with any ONNX-compatible runtime or accelerator.



```python
import cv2
import numpy as np
from super_gradients.training.utils.media.image import load_image
import onnxruntime

image = load_image("https://deci-pretrained-models.s3.amazonaws.com/sample_images/beatles-abbeyroad.jpg")
image = cv2.resize(image, (export_result.input_image_shape[1], export_result.input_image_shape[0]))
image_bchw = np.transpose(np.expand_dims(image, 0), (0, 3, 1, 2))

session = onnxruntime.InferenceSession(export_result.output, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
inputs = [o.name for o in session.get_inputs()]
outputs = [o.name for o in session.get_outputs()]
result = session.run(outputs, {inputs[0]: image_bchw})

result[0].shape, result[1].shape, result[2].shape, result[3].shape
```




    ((1, 1), (1, 1000, 4), (1, 1000), (1, 1000))



In the next section we unpack the result of prediction and show how to use it.

## Output format for detection models

If `preprocessing=True` (default value) then all models will be exported with NMS. If `preprocessing=False` models will be exported without NMS and raw model outputs will be returned. In this case, you will need to apply NMS yourself. This is useful if you want to use a custom NMS implementation that is not ONNX-compatible. In most cases you will want to use default `preprocessing=True`. It is also possible to pass a custom `nn.Module` as a `postprocessing` argument to the `export()` method. This module will be attached to the exported ONNX graph instead of the default NMS module. We encourage users to read the documentation of the `export()` method to learn more about the advanced options.

When exporting an object detection model with postprocessing enabled, the prediction format can be one of two:

* A "flat" format - `DetectionOutputFormatMode.FLAT_FORMAT`
* A "batched" format - `DetectionOutputFormatMode.BATCH_FORMAT`

You can select the desired output format by setting `export(..., output_predictions_format=DetectionOutputFormatMode.BATCH_FORMAT)`.

### Flat format

A detection results returned as a single tensor of shape `[N, 7]`, where `N` is the number of detected objects in the entire batch. Each row in the tensor represents a single detection result and has the following format:

`[batch_index, x1, y1, x2, y2, class score, class index]`

When exporting a model with batch size of 1 (default mode) you can ignore the first column as all boxes will belong to the single sample. In case you export model with batch size > 1 you have to iterate over this array like so:

```python
for sample_index in export_result.batch_size:
    detections_for_sample_i = flat_predictions[flat_predictions[:, 0] == sample_index]
    for (x1, y1, x2, y2, class_score, class_index) in detections_for_sample_i:
        class_index = int(class_index) # convert from float to int
        # do something with the detection predictions
```

### Batch format

A second supported format is so-called "batch". It matches with output format of TensorRT's NMS implementation. The return value in this case is tuple of 4 tensors:

* `num_predictions` - [B, 1] - A number of predictions per sample
* `pred_boxes` - [B, N, 4] - A coordinates of the predicted boxes in X1, Y1, X2, Y2 format
* `pred_scores` - [B, N] - A scores of the predicted boxes
* `pred_classes` - [B, N] - A class indices of the predicted boxes

Here `B` corresponds to batch size and `N` is the maximum number of detected objects per image.
In order to get the actual number of detections per image you need to iterate over `num_predictions` tensor and get the first element of each row.

Now when you're familiar with the output formats, let's see how to use them.
To start, it's useful to take a look at the values of the predictions with a naked eye:



```python
num_predictions, pred_boxes, pred_scores, pred_classes = result
num_predictions
```




    array([[25]], dtype=int64)




```python
np.set_printoptions(threshold=50, edgeitems=3)
pred_boxes, pred_boxes.shape
```




    (array([[[439.55383, 253.22733, 577.5956 , 548.11975],
             [ 35.71795, 249.40926, 176.62216, 544.69794],
             [182.39618, 249.49301, 301.44122, 529.3324 ],
             ...,
             [ -1.     ,  -1.     ,  -1.     ,  -1.     ],
             [ -1.     ,  -1.     ,  -1.     ,  -1.     ],
             [ -1.     ,  -1.     ,  -1.     ,  -1.     ]]], dtype=float32),
     (1, 1000, 4))




```python
np.set_printoptions(threshold=50, edgeitems=5)
pred_scores, pred_scores.shape
```




    (array([[ 0.9694027,  0.9693378,  0.9665707,  0.9619047,  0.7538769, ...,
             -1.       , -1.       , -1.       , -1.       , -1.       ]],
           dtype=float32),
     (1, 1000))




```python
np.set_printoptions(threshold=50, edgeitems=10)
pred_classes, pred_classes.shape
```




    (array([[ 0,  0,  0,  0,  0,  0,  0,  0,  2,  2, ..., -1, -1, -1, -1, -1,
             -1, -1, -1, -1, -1]], dtype=int64),
     (1, 1000))



### Visualizing predictions

For sake of this tutorial we will use a simple visualization function that is tailored for batch_size=1 only.
You can use it as a starting point for your own visualization code.


```python
from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST
from super_gradients.training.utils.detection_utils import DetectionVisualization
import matplotlib.pyplot as plt


def show_predictions_from_batch_format(image, predictions):
    num_predictions, pred_boxes, pred_scores, pred_classes = predictions

    assert num_predictions.shape[0] == 1, "Only batch size of 1 is supported by this function"

    num_predictions = int(num_predictions.item())
    pred_boxes = pred_boxes[0, :num_predictions]
    pred_scores = pred_scores[0, :num_predictions]
    pred_classes = pred_classes[0, :num_predictions]

    image = image.copy()
    class_names = COCO_DETECTION_CLASSES_LIST
    color_mapping = DetectionVisualization._generate_color_mapping(len(class_names))

    for (x1, y1, x2, y2, class_score, class_index) in zip(pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3], pred_scores, pred_classes):
        image = DetectionVisualization.draw_box_title(
            image_np=image,
            x1=int(x1),
            y1=int(y1),
            x2=int(x2),
            y2=int(y2),
            class_id=class_index,
            class_names=class_names,
            color_mapping=color_mapping,
            box_thickness=2,
            pred_conf=class_score,
        )

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.tight_layout()
    plt.show()

```


```python
show_predictions_from_batch_format(image, result)
```


    
![png](models_export_files/models_export_19_0.png)
    


### Changing the output format

You can explicitly specify output format of the predictions by setting the `output_predictions_format` argument of `export()` method. Let's see how it works:



```python
from super_gradients.conversion import DetectionOutputFormatMode

export_result = model.export("yolo_nas_s.onnx", output_predictions_format=DetectionOutputFormatMode.FLAT_FORMAT)
export_result
```




    Model exported successfully to yolo_nas_s.onnx
    Model expects input image of shape [1, 3, 640, 640]
    Input image dtype is torch.uint8
    Exported model already contains preprocessing (normalization) step, so you don't need to do it manually.
    Preprocessing steps to be applied to input image are:
    Sequential(
      (0): CastTensorTo(dtype=torch.float32)
      (1): ApplyMeanStd(mean=[0.], scale=[255.])
    )
    
    Exported model contains postprocessing (NMS) step with the following parameters:
        num_pre_nms_predictions=1000
        max_predictions_per_image=1000
        nms_threshold=0.7
        confidence_threshold=0.25
        output_predictions_format=flat
    
    Exported model is in ONNX format and can be used with ONNXRuntime
    To run inference with ONNXRuntime, please use the following code snippet:
    
        import onnxruntime
        import numpy as np
        session = onnxruntime.InferenceSession("yolo_nas_s.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        inputs = [o.name for o in session.get_inputs()]
        outputs = [o.name for o in session.get_outputs()]
        example_input_image = np.zeros((1, 3, 640, 640)).astype(np.uint8)
        predictions = session.run(outputs, {inputs[0]: example_input_image})
    
    Exported model has predictions in flat format:
    
        # flat_predictions is a 2D array of [N,7] shape
        # Each row represents (image_index, x_min, y_min, x_max, y_max, confidence, class_id)
        # Please note all values are floats, so you have to convert them to integers if needed
        [flat_predictions] = predictions
        for (_, x_min, y_min, x_max, y_max, confidence, class_id) in flat_predictions[0]:
            class_id = int(class_id)
            print(f"Detected object with class_id={class_id}, confidence={confidence}, x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")



Now we exported a model that produces predictions in `flat` format. Let's run the model like before and see the result:


```python
session = onnxruntime.InferenceSession(export_result.output, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
inputs = [o.name for o in session.get_inputs()]
outputs = [o.name for o in session.get_outputs()]
result = session.run(outputs, {inputs[0]: image_bchw})
result[0].shape
```




    (25, 7)




```python
def show_predictions_from_flat_format(image, predictions):
    [flat_predictions] = predictions

    image = image.copy()
    class_names = COCO_DETECTION_CLASSES_LIST
    color_mapping = DetectionVisualization._generate_color_mapping(len(class_names))

    for (sample_index, x1, y1, x2, y2, class_score, class_index) in flat_predictions[flat_predictions[:, 0] == 0]:
        class_index = int(class_index)
        image = DetectionVisualization.draw_box_title(
                    image_np=image,
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    class_id=class_index,
                    class_names=class_names,
                    color_mapping=color_mapping,
                    box_thickness=2,
                    pred_conf=class_score,
                )

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.tight_layout()
    plt.show()

```


```python
show_predictions_from_flat_format(image, result)
```


    
![png](models_export_files/models_export_25_0.png)
    


### Changing postprocessing settings

You can control a number of parameters in the NMS settings as well as maximum number of detections per image before and after NMS step:

* IOU threshold for NMS - `nms_iou_threshold`
* Score threshold for NMS - `nms_score_threshold`
* Maximum number of detections per image before NMS - `max_detections_before_nms`
* Maximum number of detections per image after NMS - `max_detections_after_nms`

For sake of demonstration, let's export a model that would produce at most one detection per image with confidence threshold above 0.8 and NMS IOU threshold of 0.5. Let's use at most 100 predictions per image before NMS step:


```python
export_result = model.export(
    "yolo_nas_s_top_1.onnx",
    confidence_threshold = 0.8,
    nms_threshold = 0.5,
    num_pre_nms_predictions = 100,
    max_predictions_per_image = 1,
    output_predictions_format = DetectionOutputFormatMode.FLAT_FORMAT
)

session = onnxruntime.InferenceSession(export_result.output, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
inputs = [o.name for o in session.get_inputs()]
outputs = [o.name for o in session.get_outputs()]
result = session.run(outputs, {inputs[0]: image_bchw})

show_predictions_from_flat_format(image, result)
```


    
![png](models_export_files/models_export_27_0.png)
    


### Export of quantized model

You can export a model with quantization to FP16 or INT8. To do so, you need to specify the `quantization_mode` argument of `export()` method.

Important notes:
* Quantization to FP16 requires CUDA / MPS device available and would not work on CPU-only machines.

Let's see how it works:


```python

from super_gradients.conversion.conversion_enums import ExportQuantizationMode

export_result = model.export(
    "yolo_nas_s_int8.onnx",
    output_predictions_format=DetectionOutputFormatMode.FLAT_FORMAT,
    quantization_mode=ExportQuantizationMode.INT8 # or ExportQuantizationMode.FP16
)

session = onnxruntime.InferenceSession(export_result.output, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
inputs = [o.name for o in session.get_inputs()]
outputs = [o.name for o in session.get_outputs()]
result = session.run(outputs, {inputs[0]: image_bchw})

show_predictions_from_flat_format(image, result)
```


    
![png](models_export_files/models_export_29_0.png)
    


### Advanced INT-8 quantization options

When quantizing a model using `quantization_mode==ExportQuantizationMode.INT8` you can pass a DataLoader to export() function to collect correct statistics of activations to prodice a more accurate quantized model.
We expect the DataLoader to return either a tuple of tensors or a single tensor. In case a tuple of tensors is returned by data-loader the first element will be used as input image.
You can use existing data-loaders from SG here as is.

**Important notes**
* A `calibration_loader` should use same image normalization parameters that were used during training.

In the example below we use a dummy data-loader for sake of showing how to use this feature. You should use your own data-loader here.


```python
import torch
from torch.utils.data import DataLoader
from super_gradients.conversion import ExportQuantizationMode

# THIS IS ONLY AN EXAMPLE. YOU SHOULD USE YOUR OWN DATA-LOADER HERE
dummy_calibration_dataset = [torch.randn((3, 640, 640), dtype=torch.float32) for _ in range(32)]
dummy_calibration_loader = DataLoader(dummy_calibration_dataset, batch_size=8, num_workers=0)
# THIS IS ONLY AN EXAMPLE. YOU SHOULD USE YOUR OWN DATA-LOADER HERE

export_result = model.export(
    "yolo_nas_s_int8_with_calibration.onnx",
    output_predictions_format=DetectionOutputFormatMode.FLAT_FORMAT,
    quantization_mode=ExportQuantizationMode.INT8,
    calibration_loader=dummy_calibration_loader
)

session = onnxruntime.InferenceSession(export_result.output, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
inputs = [o.name for o in session.get_inputs()]
outputs = [o.name for o in session.get_outputs()]
result = session.run(outputs, {inputs[0]: image_bchw})

show_predictions_from_flat_format(image, result)
```

     25%|█████████████████████████████████                                                                                                   | 4/16 [00:11<00:34,  2.91s/it]
    


    
![png](models_export_files/models_export_31_1.png)
    


### Limitations

* Dynamic batch size / input image shape is not supported yet. You can only export a model with a fixed batch size and input image shape.
* TensorRT of version 8.4.1 or higher is required.
* Quantization to FP16 requires CUDA / MPS device available.

### Supported backends

Currently, we support two backends for exporting models:

* ONNX Runtime
* TensorRT

The only difference between these two backends is what NMS implementation will be used.
ONNX Runtime uses NMS implementation from ONNX opset, while TensorRT uses its own NMS implementation which is expected to be faster.

A disadvantage of TensorRT backend is that you cannot run model exported for TensorRT backend by ONNX Runtime.
You can, however, run models exported for ONNX Runtime backend inside TensorRT.

Therefore, ONNX Runtime backend is recommended for most use-cases and is used by default.

You can specify the desired execution backend by setting the `execution_backend` argument of `export()` method:

```python
from super_gradients.conversion import ExportTargetBackend

model.export(..., engine=ExportTargetBackend.ONNXRUNTIME)
```

```python
from super_gradients.conversion import ExportTargetBackend

model.export(..., engine=ExportTargetBackend.TENSORRT)
```

## Legacy low-level export API

The .export() API is a new high-level API that is recommended for most use-cases.
However old low-level API is still available for advanced users:

* https://docs.deci.ai/super-gradients/docstring/training/models.html#training.models.conversion.convert_to_onnx
* https://docs.deci.ai/super-gradients/docstring/training/models.html#training.models.conversion.convert_to_coreml
