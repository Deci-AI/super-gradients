# Pose Estimation Models Export

This tutorial shows how to export YoloNAS-Pose model to ONNX format for deployment to ONNX-compatible runtimes and accelerators.

From this tutorial you will learn:

* How to export YoloNAS-Pose model to ONNX and run it with ONNXRuntime / TensorRT
* How to enable FP16 / INT8 quantization and export a model with calibration
* How to customize NMS parameters and number of detections per image
* How to choose whether to use TensorRT or ONNXRuntime as a backend

### Supported pose estimation models

- YoloNAS-Pose N,S,M,L
 
### Supported features

- Exporting a model to OnnxRuntime and TensorRT
- Exporting a model with preprocessing (e.g. normalizing/standardizing image according to normalization parameters during training)
- Exporting a model with postprocessing (e.g. predictions decoding and NMS) - you obtain the ready-to-consume bounding box outputs
- FP16 / INT8 quantization support with calibration
- Pre- and post-processing steps can be customized by the user if needed
- Customising input image shape and batch size
- Customising NMS parameters and number of detections per image
- Customising output format (flat or batched)

### Support matrix

It is important to note that different versions of TensorRT has varying support of ONNX opsets. 
The support matrix below shows the compatibility of different versions of TensorRT runtime in regard to batch size and output format.
We recommend to use the latest version of TensorRT available.

| Batch Size | Format | OnnxRuntime 1.13.1 | TensorRT 8.4.2 | TensorRT 8.5.3 | TensorRT 8.6.1 |
|------------|--------|--------------------|----------------|----------------|----------------|
| 1          | Flat   | Yes                | Yes            | Yes            | Yes            |
| >1         | Flat   | Yes                | Yes            | Yes            | Yes            |
| 1          | Batch  | Yes                | No             | No             | Yes            |
| >1         | Batch  | Yes                | No             | No             | Yes            |



```python
!pip install -qq super-gradients==3.3.1
```

### Minimalistic export example

Let start with the most simple example of exporting a model to ONNX format.
We will use YoloNAS-S model in this example. All models that suports new export API now expose a `export()` method that can be used to export a model. There is one mandatory argument that should be passed to the `export()` method - the path to the output file. Currently, only `.onnx` format is supported, but we may add support for CoreML and other formats in the future.


```python
from super_gradients.common.object_names import Models
from super_gradients.training import models

model = models.get(Models.YOLO_NAS_POSE_S, pretrained_weights="coco_pose")

export_result = model.export("yolo_nas_pose_s.onnx")
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




    
    Model exported successfully to yolo_nas_pose_s.onnx
    Model expects input image of shape [1, 3, 640, 640]
    Input image dtype is torch.uint8
    
    Exported model already contains preprocessing (normalization) step, so you don't need to do it manually.
    Preprocessing steps to be applied to input image are:
    Sequential(
      (0): CastTensorTo(dtype=torch.float32)
      (1): ChannelSelect(channels_indexes=tensor([2, 1, 0]))
      (2): ApplyMeanStd(mean=[0.], scale=[255.])
    )
    
    
    Exported model contains postprocessing (NMS) step with the following parameters:
        num_pre_nms_predictions=1000
        max_predictions_per_image=1000
        nms_threshold=0.7
        confidence_threshold=0.05
        output_predictions_format=batch
    
    
    Exported model is in ONNX format and can be used with ONNXRuntime
    To run inference with ONNXRuntime, please use the following code snippet:
    
        import onnxruntime
        import numpy as np
        session = onnxruntime.InferenceSession("yolo_nas_pose_s.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        inputs = [o.name for o in session.get_inputs()]
        outputs = [o.name for o in session.get_outputs()]
    
        example_input_image = np.zeros((1, 3, 640, 640)).astype(np.uint8)
        predictions = session.run(outputs, {inputs[0]: example_input_image})
    
    Exported model can also be used with TensorRT
    To run inference with TensorRT, please see TensorRT deployment documentation
    You can benchmark the model using the following code snippet:
    
        trtexec --onnx=yolo_nas_pose_s.onnx --fp16 --avgRuns=100 --duration=15
    
    Exported model has predictions in batch format:
    
        num_detections, pred_boxes, pred_scores, pred_joints = predictions
        for image_index in range(num_detections.shape[0]):
            for i in range(num_detections[image_index,0]):
                confidence = pred_scores[image_index, i]
                x_min, y_min, x_max, y_max = pred_boxes[image_index, i]
                pred_joints = pred_joints[image_index, i]
                print(f"Detected pose with confidence={confidence}, x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
                for joint_index, (x, y, confidence) in enumerate(pred_joints[i]):
                    print(f"Joint {joint_index} has coordinates x={x}, y={y}, confidence={confidence}")
    



That's it. You can now use the exported model with any ONNX-compatible runtime or accelerator.



```python
import cv2
import numpy as np
from super_gradients.training.utils.media.image import load_image
import onnxruntime

image = load_image("https://deci-pretrained-models.s3.amazonaws.com/sample_images/beatles-abbeyroad.jpg")
image = cv2.resize(image, (export_result.input_image_shape[1], export_result.input_image_shape[0]))
image_bchw = np.transpose(np.expand_dims(image, 0), (0, 3, 1, 2))

session = onnxruntime.InferenceSession(export_result.output,
                                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
inputs = [o.name for o in session.get_inputs()]
outputs = [o.name for o in session.get_outputs()]
result = session.run(outputs, {inputs[0]: image_bchw})

result[0].shape, result[1].shape, result[2].shape, result[3].shape
```




    ((1, 1), (1, 1000, 4), (1, 1000), (1, 1000, 17, 3))



In the next section we unpack the result of prediction and show how to use it.

## Output format for detection models

If `preprocessing=True` (default value) then all models will be exported with NMS. If `preprocessing=False` models will be exported without NMS and raw model outputs will be returned. In this case, you will need to apply NMS yourself. This is useful if you want to use a custom NMS implementation that is not ONNX-compatible. In most cases you will want to use default `preprocessing=True`. It is also possible to pass a custom `nn.Module` as a `postprocessing` argument to the `export()` method. This module will be attached to the exported ONNX graph instead of the default NMS module. We encourage users to read the documentation of the `export()` method to learn more about the advanced options.

When exporting an object detection model with postprocessing enabled, the prediction format can be one of two:

* A "flat" format - `DetectionOutputFormatMode.FLAT_FORMAT`
* A "batched" format - `DetectionOutputFormatMode.BATCH_FORMAT`

You can select the desired output format by setting `export(..., output_predictions_format=DetectionOutputFormatMode.BATCH_FORMAT)`.

### Flat format

A detection results returned as a single tensor of shape `[N, 6 + 3 * NumKeypoints]`, where `N` is the number of detected objects in the entire batch. Each row in the tensor represents a single detection result and has the following format:

`[batch_index, x1, y1, x2, y2, pose confidence, (x,y,score) * num_keypoints]`

When exporting a model with batch size of 1 (default mode) you can ignore the first column as all boxes will belong to the single sample. In case you export model with batch size > 1 you have to iterate over this array like so:


```python
def iterate_over_flat_predictions(predictions, batch_size):
    [flat_predictions] = predictions

    for image_index in range(batch_size):
        mask = flat_predictions[:, 0] == image_index
        pred_bboxes = flat_predictions[mask, 1:5]
        pred_scores = flat_predictions[mask, 5]
        pred_joints = flat_predictions[mask, 6:].reshape((len(pred_bboxes), -1, 3))
        yield image_index, pred_bboxes, pred_scores, pred_joints
```

Iteration over the predictions would be as follows:

```python
for image_index, pred_bboxes, pred_scores, pred_joints in iterate_over_flat_predictions(predictions, batch_size):
   ... # Do something useful with the predictions
```

### Batch format

A second supported format is so-called "batch". It matches with output format of TensorRT's NMS implementation. The return value in this case is tuple of 4 tensors:

* `num_predictions` - [B, 1] - A number of predictions per sample
* `pred_boxes` - [B, N, 4] - A coordinates of the predicted boxes in X1, Y1, X2, Y2 format
* `pred_scores` - [B, N] - A scores of the predicted boxes
* `pred_classes` - [B, N] - A class indices of the predicted boxes

Here `B` corresponds to batch size and `N` is the maximum number of detected objects per image.
In order to get the actual number of detections per image you need to iterate over `num_predictions` tensor and get the first element of each row.

A corresponding code snippet for iterating over the batch predictions would look like this:


```python
def iterate_over_batch_predictions(predictions, batch_size):
    num_detections, batch_boxes, batch_scores, batch_joints = predictions
    for image_index in range(batch_size):
        num_detection_in_image = num_detections[image_index, 0]

        pred_scores = batch_scores[image_index, :num_detection_in_image]
        pred_boxes = batch_boxes[image_index, :num_detection_in_image]
        pred_joints = batch_joints[image_index, :num_detection_in_image].reshape((len(pred_scores), -1, 3))

        yield image_index, pred_boxes, pred_scores, pred_joints
```

And similary to flat format, iteration over the predictions would be as follows:

```python
for image_index, pred_bboxes, pred_scores, pred_joints in iterate_over_batch_predictions(predictions, batch_size):
   ... # Do something useful with the predictions
```



Now when you're familiar with the output formats, let's see how to use them.
To start, it's useful to take a look at the values of the predictions with a naked eye:



```python
num_predictions, pred_boxes, pred_scores, pred_poses = result
num_predictions
```




    array([[9]], dtype=int64)




```python
np.set_printoptions(threshold=50, edgeitems=3)
pred_boxes, pred_boxes.shape
```




    (array([[[182.49644 , 249.07802 , 305.27576 , 530.3644  ],
             [ 34.52883 , 247.74242 , 175.7783  , 544.1926  ],
             [438.808   , 251.08049 , 587.11865 , 552.69336 ],
             ...,
             [ 67.20265 , 248.3974  , 122.415375, 371.65637 ],
             [625.7083  , 306.74194 , 639.4926  , 501.08337 ],
             [450.61108 , 386.74622 , 556.77325 , 523.2412  ]]], dtype=float32),
     (1, 1000, 4))




```python
np.set_printoptions(threshold=50, edgeitems=5)
pred_scores, pred_scores.shape
```




    (array([[0.84752125, 0.826281  , 0.82436883, 0.7830074 , 0.19943413, ...,
             0.00850207, 0.00849411, 0.00848398, 0.00848269, 0.00848123]],
           dtype=float32),
     (1, 1000))




```python
np.set_printoptions(threshold=50, edgeitems=10)
pred_poses, pred_poses.shape
```




    (array([[[[2.62617737e+02, 2.75986389e+02, 7.74692297e-01],
              [2.63401123e+02, 2.70397522e+02, 3.57395113e-01],
              [2.57980499e+02, 2.70888336e+02, 7.75521040e-01],
              [2.55243866e+02, 2.73192688e+02, 1.17698282e-01],
              [2.44570236e+02, 2.74372101e+02, 7.43144035e-01],
              [2.49452560e+02, 3.06677490e+02, 6.88611269e-01],
              [2.34771881e+02, 3.11479401e+02, 8.93006802e-01],
              [2.59407349e+02, 3.47042847e+02, 6.79081917e-01],
              [2.28826141e+02, 3.61830658e+02, 8.90955150e-01],
              [2.73077789e+02, 3.78529633e+02, 7.46283650e-01],
              [2.31264404e+02, 4.02306763e+02, 8.83945823e-01],
              [2.39841583e+02, 3.89617126e+02, 9.04788017e-01],
              [2.39531204e+02, 3.92390350e+02, 9.17376935e-01],
              [2.20032333e+02, 4.48412506e+02, 9.05503273e-01],
              [2.58518188e+02, 4.50223969e+02, 9.40084636e-01],
              [2.01152466e+02, 5.02089630e+02, 8.42420936e-01],
              [2.82095978e+02, 5.06688324e+02, 8.73963714e-01]],
     
             [[1.14750252e+02, 2.75872864e+02, 8.29551518e-01],
              [1.15829544e+02, 2.70712891e+02, 4.48927283e-01],
              [1.09389343e+02, 2.70643494e+02, 8.33203077e-01],
              [1.13211357e+02, 2.73260956e+02, 9.57965851e-02],
              [9.51810608e+01, 2.73486206e+02, 7.43200541e-01],
              [1.03895226e+02, 3.08002930e+02, 6.69343054e-01],
              [8.49247360e+01, 3.08924500e+02, 8.86556387e-01],
              [1.02957947e+02, 3.50419678e+02, 3.93054008e-01],
              [9.04963760e+01, 3.54479675e+02, 8.36569667e-01],
              [1.10958115e+02, 3.80760162e+02, 4.20597672e-01],
              [1.08458832e+02, 3.84375946e+02, 7.75406122e-01],
              [1.04001587e+02, 3.86138672e+02, 8.84934366e-01],
              [8.66012573e+01, 3.88762817e+02, 9.12180185e-01],
              [1.20186020e+02, 4.47084015e+02, 8.99593532e-01],
              [7.29626541e+01, 4.55435028e+02, 9.07496691e-01],
              [1.47440369e+02, 5.05209564e+02, 8.53177905e-01],
              [5.24395561e+01, 5.16123291e+02, 8.44702840e-01]],
     
             [[5.46199341e+02, 2.83605713e+02, 6.09813333e-01],
              [5.45253479e+02, 2.78786011e+02, 1.59033239e-01],
              [5.44112183e+02, 2.78675476e+02, 5.77503145e-01],
              [5.23715332e+02, 2.79131958e+02, 1.26258403e-01],
              [5.31369812e+02, 2.79856354e+02, 6.77978814e-01],
              [5.09348022e+02, 3.05230194e+02, 7.24757075e-01],
              [5.28026428e+02, 3.09744904e+02, 8.99605513e-01],
              [4.99039612e+02, 3.44572968e+02, 3.78879875e-01],
              [5.24457764e+02, 3.55649597e+02, 7.62571633e-01],
              [5.10139526e+02, 3.65226715e+02, 3.24873060e-01],
              [5.27322510e+02, 3.88639832e+02, 5.69304049e-01],
              [5.15003845e+02, 3.89877136e+02, 9.01696205e-01],
              [5.16953247e+02, 3.92533173e+02, 9.39601123e-01],
              [5.39589539e+02, 4.55641571e+02, 8.34750414e-01],
              [5.00366119e+02, 4.57584869e+02, 8.84028912e-01],
              [5.50320129e+02, 5.21863281e+02, 7.15586364e-01],
              [4.54590271e+02, 5.17590332e+02, 7.93488443e-01]],
     
             [[3.95477234e+02, 2.89274658e+02, 7.75570869e-01],
              [3.95304504e+02, 2.84256470e+02, 3.39867800e-01],
              [3.90781616e+02, 2.84641663e+02, 7.73032904e-01],
              [3.87255829e+02, 2.86732422e+02, 1.02017552e-01],
              [3.78526611e+02, 2.88418518e+02, 7.23391712e-01],
              [3.78563995e+02, 3.17696136e+02, 5.57907283e-01],
              [3.71781860e+02, 3.21135010e+02, 8.96694720e-01],
              [3.77163239e+02, 3.69956848e+02, 2.86054850e-01],
              [3.81310852e+02, 3.72657684e+02, 9.02382135e-01],
              [3.94692657e+02, 4.02302185e+02, 3.00682962e-01],
              [4.01326813e+02, 4.06579620e+02, 8.09893429e-01],
              [3.78791962e+02, 4.02112335e+02, 8.89372826e-01],
              [3.66032654e+02, 4.04255096e+02, 9.60101604e-01],
              [3.94984253e+02, 4.56911652e+02, 8.59754562e-01],
              [3.50547211e+02, 4.59072144e+02, 8.89475405e-01],
              [4.15519073e+02, 5.11703705e+02, 7.49074817e-01],
              [3.20821442e+02, 5.17565125e+02, 7.18492508e-01]],
     
             [[4.84677582e+02, 2.24471405e+02, 4.86074597e-01],
              [4.86033966e+02, 2.23227020e+02, 4.69703048e-01],
              [4.84380615e+02, 2.23190399e+02, 3.11850667e-01],
              [4.89061676e+02, 2.24047882e+02, 4.72676337e-01],
              [4.84933594e+02, 2.24044022e+02, 1.94730312e-01],
              [4.90376617e+02, 2.31114029e+02, 7.74827123e-01],
              [4.87172882e+02, 2.31243103e+02, 5.22870839e-01],
              [4.92284210e+02, 2.40522079e+02, 6.27111852e-01],
              [4.86155182e+02, 2.40626282e+02, 2.84491181e-01],
              [4.89158051e+02, 2.47762543e+02, 5.62235475e-01],
              [4.83380157e+02, 2.48899887e+02, 3.21294904e-01],
              [4.88469818e+02, 2.50025833e+02, 7.21766710e-01],
              [4.86471069e+02, 2.49770172e+02, 5.98737955e-01],
              [4.87357697e+02, 2.63648315e+02, 6.99281693e-01],
              [4.87669373e+02, 2.63486328e+02, 5.70817828e-01],
              [4.88168732e+02, 2.76446533e+02, 6.58615112e-01],
              [4.89197388e+02, 2.75976959e+02, 5.55222452e-01]],
     
             [[6.36717590e+02, 2.51208191e+02, 2.12087035e-01],
              [6.36672424e+02, 2.50077515e+02, 1.99229538e-01],
              [6.37246399e+02, 2.49852295e+02, 1.97700143e-01],
              [6.34123657e+02, 2.50509720e+02, 2.98916340e-01],
              [6.36376038e+02, 2.50242935e+02, 3.13487828e-01],
              [6.31871460e+02, 2.57533508e+02, 5.86982608e-01],
              [6.37110168e+02, 2.57340302e+02, 4.47703123e-01],
              [6.28766052e+02, 2.67072784e+02, 4.61719364e-01],
              [6.36492249e+02, 2.66546448e+02, 3.16147625e-01],
              [6.29501099e+02, 2.74682373e+02, 3.83048326e-01],
              [6.36805420e+02, 2.74100159e+02, 2.63563633e-01],
              [6.34455627e+02, 2.75272491e+02, 5.40961742e-01],
              [6.38066833e+02, 2.75237091e+02, 4.58729565e-01],
              [6.36313782e+02, 2.88538422e+02, 5.08250773e-01],
              [6.37799011e+02, 2.88380219e+02, 3.93619925e-01],
              [6.37243591e+02, 3.00429749e+02, 4.48986799e-01],
              [6.36930481e+02, 3.00298950e+02, 3.70416939e-01]],
     
             [[4.85152618e+02, 2.24845276e+02, 4.79349971e-01],
              [4.86152771e+02, 2.23562973e+02, 5.12778938e-01],
              [4.84754913e+02, 2.23552597e+02, 3.28223199e-01],
              [4.88753723e+02, 2.23973236e+02, 4.59801853e-01],
              [4.84796692e+02, 2.23940887e+02, 2.17209488e-01],
              [4.89951935e+02, 2.31293716e+02, 7.01054573e-01],
              [4.87081268e+02, 2.30910217e+02, 5.27912199e-01],
              [4.92560669e+02, 2.40431274e+02, 5.70250571e-01],
              [4.85739075e+02, 2.40336624e+02, 3.40047777e-01],
              [4.87803558e+02, 2.44864410e+02, 4.88935739e-01],
              [4.83851227e+02, 2.47046936e+02, 3.15269023e-01],
              [4.88669495e+02, 2.49567139e+02, 6.54390395e-01],
              [4.86886505e+02, 2.49332306e+02, 5.78326464e-01],
              [4.88596924e+02, 2.63882385e+02, 6.07191145e-01],
              [4.87096771e+02, 2.63732697e+02, 5.18960536e-01],
              [4.90393372e+02, 2.77110718e+02, 5.59345722e-01],
              [4.88872559e+02, 2.76716248e+02, 4.86189365e-01]],
     
             [[5.34085327e+02, 3.54973755e+02, 6.29425645e-02],
              [5.25682129e+02, 3.50349976e+02, 3.49148512e-02],
              [5.31193054e+02, 3.50817444e+02, 2.92686522e-02],
              [5.23321350e+02, 3.47386475e+02, 5.74792027e-02],
              [5.38499756e+02, 3.47819885e+02, 5.25604784e-02],
              [5.14988953e+02, 3.59584564e+02, 1.63533330e-01],
              [5.45641357e+02, 3.61579987e+02, 1.23190343e-01],
              [5.00512726e+02, 3.95821960e+02, 3.09125751e-01],
              [5.44541504e+02, 3.96118866e+02, 1.54896080e-01],
              [4.82409119e+02, 4.39998688e+02, 4.11001146e-01],
              [5.37386047e+02, 4.40229431e+02, 2.60766208e-01],
              [5.14716248e+02, 4.06201416e+02, 4.37605500e-01],
              [5.36000793e+02, 4.07051453e+02, 4.62512881e-01],
              [4.93844604e+02, 4.60759338e+02, 7.02160478e-01],
              [5.41572754e+02, 4.59306824e+02, 7.30659008e-01],
              [4.56344788e+02, 5.17132751e+02, 7.68323183e-01],
              [5.50762573e+02, 5.26608093e+02, 8.12822223e-01]],
     
             [[4.85128510e+02, 2.24573792e+02, 4.49509919e-01],
              [4.86325043e+02, 2.23426971e+02, 4.56321508e-01],
              [4.84612244e+02, 2.23425217e+02, 3.22477281e-01],
              [4.88920288e+02, 2.23974472e+02, 4.50538874e-01],
              [4.84553711e+02, 2.24078705e+02, 2.35897660e-01],
              [4.89817444e+02, 2.30877258e+02, 7.57300436e-01],
              [4.86150330e+02, 2.30926620e+02, 6.08168483e-01],
              [4.90756927e+02, 2.40545914e+02, 5.52712023e-01],
              [4.84597351e+02, 2.40188354e+02, 3.77925187e-01],
              [4.88303528e+02, 2.48004913e+02, 4.86442685e-01],
              [4.82454742e+02, 2.48152771e+02, 3.76741141e-01],
              [4.88135956e+02, 2.49587601e+02, 7.44444489e-01],
              [4.85826813e+02, 2.49573761e+02, 6.41330898e-01],
              [4.88307281e+02, 2.63924133e+02, 7.00867116e-01],
              [4.87772095e+02, 2.63844971e+02, 5.53154111e-01],
              [4.89414215e+02, 2.76725952e+02, 6.40206695e-01],
              [4.89250122e+02, 2.76087646e+02, 5.22767425e-01]],
     
             [[2.62897675e+02, 2.75270264e+02, 8.17279518e-01],
              [2.63719330e+02, 2.69972534e+02, 3.53590310e-01],
              [2.57903656e+02, 2.70411011e+02, 7.89806545e-01],
              [2.56239288e+02, 2.73104980e+02, 1.26096725e-01],
              [2.43735504e+02, 2.74078979e+02, 7.17853844e-01],
              [2.49559097e+02, 3.06182343e+02, 7.88639188e-01],
              [2.35288086e+02, 3.11189423e+02, 9.01687264e-01],
              [2.59721069e+02, 3.46684631e+02, 7.22490489e-01],
              [2.28704590e+02, 3.61947662e+02, 8.87249708e-01],
              [2.74032990e+02, 3.78454346e+02, 8.30342770e-01],
              [2.32086899e+02, 4.03276672e+02, 8.71515155e-01],
              [2.40708374e+02, 3.88985168e+02, 9.22950149e-01],
              [2.39727539e+02, 3.92127899e+02, 9.41203475e-01],
              [2.20081482e+02, 4.48501831e+02, 9.01696205e-01],
              [2.58718811e+02, 4.50506653e+02, 9.42129970e-01],
              [2.00505875e+02, 5.03255737e+02, 8.64027262e-01],
              [2.79800537e+02, 5.07109741e+02, 9.09489870e-01]],
     
             ...,
     
             [[4.61209198e+02, 5.00869293e+02, 1.03250116e-01],
              [4.60878571e+02, 4.98928223e+02, 9.74662304e-02],
              [4.63814453e+02, 4.98858643e+02, 9.01672840e-02],
              [4.61473694e+02, 4.97081299e+02, 1.11838251e-01],
              [4.66341888e+02, 4.98344086e+02, 8.95042121e-02],
              [4.62850220e+02, 5.00182098e+02, 1.57806307e-01],
              [4.64795868e+02, 4.99315094e+02, 1.49133205e-01],
              [4.57825165e+02, 5.20207153e+02, 2.29635060e-01],
              [4.61742828e+02, 5.17086426e+02, 2.29862571e-01],
              [4.50936462e+02, 5.30673828e+02, 2.75914669e-01],
              [4.54896027e+02, 5.28634338e+02, 2.74364412e-01],
              [4.57495605e+02, 5.26224792e+02, 2.37078875e-01],
              [4.61555908e+02, 5.25614807e+02, 2.40101725e-01],
              [4.60925446e+02, 4.96987122e+02, 4.87953067e-01],
              [4.60925262e+02, 4.96787628e+02, 4.85726297e-01],
              [4.53465698e+02, 5.25413025e+02, 6.21427596e-01],
              [4.53733032e+02, 5.24209717e+02, 6.24894500e-01]],
     
             [[2.05005539e+02, 4.50580750e+02, 1.31070495e-01],
              [2.06915314e+02, 4.46343109e+02, 1.21572196e-01],
              [2.01894440e+02, 4.46064514e+02, 1.21150881e-01],
              [2.01684692e+02, 4.40815399e+02, 1.40165716e-01],
              [2.01872391e+02, 4.42573273e+02, 1.60495758e-01],
              [2.03456421e+02, 4.41541779e+02, 1.95060015e-01],
              [1.99143585e+02, 4.41433228e+02, 2.67008513e-01],
              [1.99477829e+02, 4.69973999e+02, 1.63713366e-01],
              [1.92531204e+02, 4.67465607e+02, 2.48498559e-01],
              [1.99401413e+02, 4.93351898e+02, 2.44933963e-01],
              [1.88049957e+02, 4.91161194e+02, 3.20274264e-01],
              [1.98843552e+02, 4.76965668e+02, 2.19146311e-01],
              [1.96542709e+02, 4.77562500e+02, 2.62409449e-01],
              [2.07360992e+02, 4.54596161e+02, 2.66694158e-01],
              [2.06813599e+02, 4.55067657e+02, 3.14766556e-01],
              [1.99306931e+02, 4.96951843e+02, 3.17928433e-01],
              [1.91966995e+02, 4.95296967e+02, 3.60845208e-01]],
     
             [[1.17189377e+02, 4.36203522e+02, 1.07620925e-01],
              [1.24635544e+02, 4.28187042e+02, 1.13621294e-01],
              [1.14065216e+02, 4.28010742e+02, 1.01120114e-01],
              [1.03587090e+02, 4.22528473e+02, 1.47127748e-01],
              [1.09290840e+02, 4.23242645e+02, 1.31708324e-01],
              [1.14252174e+02, 4.20988678e+02, 2.51191586e-01],
              [1.06915726e+02, 4.24170441e+02, 1.50790274e-01],
              [1.32476715e+02, 4.58789917e+02, 3.43669653e-01],
              [1.18565887e+02, 4.62118622e+02, 1.91581368e-01],
              [1.45139618e+02, 5.03220154e+02, 3.17180187e-01],
              [1.24895592e+02, 5.02303436e+02, 2.16081768e-01],
              [1.31573227e+02, 4.72023376e+02, 3.18301201e-01],
              [1.19495789e+02, 4.74669159e+02, 2.69022077e-01],
              [1.22083450e+02, 4.42408264e+02, 4.50313896e-01],
              [1.16150276e+02, 4.41128998e+02, 3.77744079e-01],
              [1.49492691e+02, 5.04421417e+02, 5.07762611e-01],
              [1.34144836e+02, 5.07860443e+02, 4.48754340e-01]],
     
             [[2.14000473e+02, 4.86687683e+02, 7.86419809e-02],
              [2.15655548e+02, 4.83943726e+02, 8.02902877e-02],
              [2.14233398e+02, 4.83678711e+02, 7.63061047e-02],
              [2.05150635e+02, 4.81322968e+02, 8.51484835e-02],
              [2.13763138e+02, 4.82678192e+02, 7.99615383e-02],
              [2.09407318e+02, 4.83091736e+02, 6.66974783e-02],
              [2.10882462e+02, 4.83498657e+02, 6.88760877e-02],
              [2.07825256e+02, 5.03304626e+02, 1.05907768e-01],
              [2.03670380e+02, 5.00577301e+02, 1.13199592e-01],
              [2.10730148e+02, 5.11463593e+02, 1.50998890e-01],
              [2.01818771e+02, 5.09126038e+02, 1.56455725e-01],
              [2.09329880e+02, 5.07587494e+02, 1.25299037e-01],
              [2.10325302e+02, 5.07958649e+02, 1.29461318e-01],
              [2.15592712e+02, 4.70666565e+02, 3.47643852e-01],
              [2.06730087e+02, 4.70157043e+02, 3.50907385e-01],
              [2.10237106e+02, 5.05089752e+02, 5.42780340e-01],
              [2.02875366e+02, 5.03588440e+02, 5.48115134e-01]],
     
             [[1.05191460e+01, 3.04227753e+02, 1.59859300e-01],
              [9.91621780e+00, 2.97300903e+02, 1.51621729e-01],
              [7.29246521e+00, 2.97828491e+02, 1.26756042e-01],
              [4.04202652e+00, 2.99062836e+02, 1.91092819e-01],
              [8.76972675e+00, 2.98623962e+02, 1.43800229e-01],
              [7.87711525e+00, 3.18044312e+02, 2.54323125e-01],
              [4.54593849e+00, 3.18904205e+02, 1.96629196e-01],
              [9.86588669e+00, 3.56114166e+02, 2.55557209e-01],
              [4.26294327e+00, 3.58117035e+02, 2.07813501e-01],
              [1.00224953e+01, 3.82647644e+02, 2.99376070e-01],
              [9.63525200e+00, 3.85407532e+02, 2.38256633e-01],
              [6.62776756e+00, 3.82030975e+02, 2.98733532e-01],
              [6.29885483e+00, 3.81703369e+02, 2.40455717e-01],
              [1.09976912e+01, 4.15991547e+02, 3.44002962e-01],
              [9.20233154e+00, 4.14822021e+02, 2.98721015e-01],
              [1.90578060e+01, 4.53060577e+02, 3.28865886e-01],
              [1.60912189e+01, 4.54475342e+02, 3.00892353e-01]],
     
             [[2.46807098e+02, 4.36069702e+02, 8.19541812e-02],
              [2.39908157e+02, 4.23463287e+02, 5.91132939e-02],
              [2.44816879e+02, 4.23787323e+02, 1.07136816e-01],
              [2.29443649e+02, 4.23795959e+02, 1.57157421e-01],
              [2.56962738e+02, 4.21870117e+02, 2.15095520e-01],
              [2.19442352e+02, 4.38299896e+02, 1.04281247e-01],
              [2.59759827e+02, 4.37226196e+02, 2.97334015e-01],
              [2.17983383e+02, 4.76868866e+02, 9.02867913e-02],
              [2.75180176e+02, 4.77196716e+02, 2.56656975e-01],
              [2.39149460e+02, 5.04646332e+02, 1.72955483e-01],
              [2.87564484e+02, 5.05277557e+02, 3.09806854e-01],
              [2.21052811e+02, 4.91340271e+02, 1.72795206e-01],
              [2.58252136e+02, 4.90655029e+02, 2.91952282e-01],
              [2.16645493e+02, 4.59269470e+02, 2.43554443e-01],
              [2.62750854e+02, 4.60674896e+02, 4.19801265e-01],
              [2.24770477e+02, 5.03889374e+02, 3.55323315e-01],
              [2.80369080e+02, 5.05997070e+02, 5.03326595e-01]],
     
             [[1.17960297e+02, 2.89125092e+02, 2.57734239e-01],
              [1.21017349e+02, 2.84405029e+02, 2.02260256e-01],
              [1.14742531e+02, 2.84067810e+02, 2.46854722e-01],
              [1.17946815e+02, 2.85183044e+02, 1.60633355e-01],
              [1.09795914e+02, 2.85077118e+02, 2.82426566e-01],
              [1.18548523e+02, 3.02608826e+02, 3.53369653e-01],
              [1.04250267e+02, 3.02424713e+02, 4.10820127e-01],
              [1.21699966e+02, 3.35773773e+02, 2.39603907e-01],
              [1.04185631e+02, 3.36030792e+02, 3.28706235e-01],
              [1.20722572e+02, 3.56416412e+02, 2.73619235e-01],
              [1.09332932e+02, 3.57116699e+02, 3.48541379e-01],
              [1.17005905e+02, 3.57139191e+02, 3.36251974e-01],
              [1.07722672e+02, 3.57518280e+02, 3.67341101e-01],
              [1.17648331e+02, 3.51363220e+02, 2.25729465e-01],
              [1.09447929e+02, 3.52017517e+02, 2.54720926e-01],
              [1.17475136e+02, 3.85022736e+02, 2.04644471e-01],
              [1.07938232e+02, 3.86506287e+02, 2.23846316e-01]],
     
             [[1.13875908e+02, 2.76212708e+02, 7.35527277e-01],
              [1.16164986e+02, 2.70696411e+02, 4.00955290e-01],
              [1.08107491e+02, 2.70656555e+02, 7.91907310e-01],
              [1.15435028e+02, 2.74475403e+02, 8.26110542e-02],
              [9.37413864e+01, 2.74376068e+02, 6.82844341e-01],
              [1.08141586e+02, 3.07976593e+02, 5.38383245e-01],
              [8.50265427e+01, 3.10244720e+02, 7.12180495e-01],
              [1.12603233e+02, 3.46686218e+02, 2.69838393e-01],
              [8.59471817e+01, 3.53580048e+02, 4.93328601e-01],
              [1.14810791e+02, 3.46329071e+02, 2.76691556e-01],
              [1.05267067e+02, 3.53531494e+02, 4.08906013e-01],
              [1.08032372e+02, 3.76369720e+02, 4.52529192e-01],
              [9.14567490e+01, 3.78959900e+02, 4.70449597e-01],
              [1.10643852e+02, 4.05414001e+02, 3.21711093e-01],
              [9.75953293e+01, 4.07489868e+02, 3.45197320e-01],
              [1.01579475e+02, 4.40818176e+02, 2.17337132e-01],
              [9.04172211e+01, 4.44152771e+02, 2.28111655e-01]],
     
             [[6.42500244e+02, 3.39081055e+02, 1.75797671e-01],
              [6.42386841e+02, 3.34906342e+02, 1.55016124e-01],
              [6.41675354e+02, 3.34820374e+02, 1.29657656e-01],
              [6.41743713e+02, 3.37194366e+02, 2.45585531e-01],
              [6.42665283e+02, 3.37119873e+02, 1.85871810e-01],
              [6.40878662e+02, 3.55136444e+02, 1.79604352e-01],
              [6.41734558e+02, 3.55543427e+02, 2.08183020e-01],
              [6.40419678e+02, 3.78958618e+02, 1.67493135e-01],
              [6.42994446e+02, 3.80664490e+02, 1.80074334e-01],
              [6.39848328e+02, 3.82542816e+02, 2.73867249e-01],
              [6.40997681e+02, 3.84963409e+02, 2.18453676e-01],
              [6.40253662e+02, 4.01003357e+02, 1.71083063e-01],
              [6.43156189e+02, 4.01335419e+02, 2.02334285e-01],
              [6.39791016e+02, 4.17747009e+02, 1.88098043e-01],
              [6.40000122e+02, 4.15383392e+02, 2.22081602e-01],
              [6.37456421e+02, 4.40941406e+02, 2.00318485e-01],
              [6.39243164e+02, 4.41459686e+02, 2.33620048e-01]],
     
             [[5.17478271e+02, 4.09209961e+02, 1.95783913e-01],
              [5.21710632e+02, 4.01950928e+02, 1.90346301e-01],
              [5.12909302e+02, 4.02274841e+02, 1.88751698e-01],
              [5.23178894e+02, 3.94687897e+02, 2.45400369e-01],
              [5.10056122e+02, 3.96813751e+02, 2.50065386e-01],
              [5.24025024e+02, 4.02232819e+02, 4.11179900e-01],
              [5.08445953e+02, 4.01063019e+02, 3.96910936e-01],
              [5.30093384e+02, 4.37817413e+02, 3.31987023e-01],
              [5.01253967e+02, 4.33004089e+02, 3.27533305e-01],
              [5.25303284e+02, 4.75992004e+02, 3.07230651e-01],
              [4.89163544e+02, 4.70396912e+02, 3.01515013e-01],
              [5.20892578e+02, 4.50428101e+02, 4.54319179e-01],
              [5.09630005e+02, 4.49830933e+02, 4.60015267e-01],
              [5.25981995e+02, 4.57243317e+02, 4.48488414e-01],
              [4.98697205e+02, 4.55512695e+02, 4.54110742e-01],
              [5.19384705e+02, 5.21536316e+02, 4.20579553e-01],
              [4.83649933e+02, 5.19510498e+02, 4.25356269e-01]]]],
           dtype=float32),
     (1, 1000, 17, 3))



### Visualizing predictions

For sake of this tutorial we will use a simple visualization function that is tailored for batch_size=1 only.
You can use it as a starting point for your own visualization code.


```python
from super_gradients.training.utils.visualization.pose_estimation import PoseVisualization
import matplotlib.pyplot as plt

def show_predictions_from_batch_format(image, predictions):
    # In this tutorial we are using batch size of 1, therefore we are getting only first element of the predictions
    image_index, pred_boxes, pred_scores, pred_joints = next(iter(iterate_over_batch_predictions(predictions, 1)))

    image = PoseVisualization.draw_poses(
        image=image, poses=pred_joints, scores=pred_scores, boxes=pred_boxes,
        edge_links=None, edge_colors=None, keypoint_colors=None, is_crowd=None
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.tight_layout()
    plt.show()

```


```python
show_predictions_from_batch_format(image, result)
```


    
![png](models_export_pose_files/models_export_pose_26_0.png)
    


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
      (1): ChannelSelect(channels_indexes=tensor([2, 1, 0]))
      (2): ApplyMeanStd(mean=[0.], scale=[255.])
    )
    
    
    Exported model contains postprocessing (NMS) step with the following parameters:
        num_pre_nms_predictions=1000
        max_predictions_per_image=1000
        nms_threshold=0.7
        confidence_threshold=0.05
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
    
    Exported model can also be used with TensorRT
    To run inference with TensorRT, please see TensorRT deployment documentation
    You can benchmark the model using the following code snippet:
    
        trtexec --onnx=yolo_nas_s.onnx --fp16 --avgRuns=100 --duration=15
    
    
    Exported model has predictions in flat format:
    
    # flat_predictions is a 2D array of [N,K] shape
    # Each row represents (image_index, x_min, y_min, x_max, y_max, confidence, joints...)
    # Please note all values are floats, so you have to convert them to integers if needed
    
    [flat_predictions] = predictions
    pred_bboxes = flat_predictions[:, 1:5]
    pred_scores = flat_predictions[:, 5]
    pred_joints = flat_predictions[:, 6:].reshape((len(pred_bboxes), -1, 3))
    for i in range(len(pred_bboxes)):
        confidence = pred_scores[i]
        x_min, y_min, x_max, y_max = pred_bboxes[i]
        print(f"Detected pose with confidence={{confidence}}, x_min={{x_min}}, y_min={{y_min}}, x_max={{x_max}}, y_max={{y_max}}")
        for joint_index, (x, y, confidence) in enumerate(pred_joints[i]):")
            print(f"Joint {{joint_index}} has coordinates x={{x}}, y={{y}}, confidence={{confidence}}")
    



Now we exported a model that produces predictions in `flat` format. Let's run the model like before and see the result:


```python
session = onnxruntime.InferenceSession(export_result.output,
                                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
inputs = [o.name for o in session.get_inputs()]
outputs = [o.name for o in session.get_outputs()]
result = session.run(outputs, {inputs[0]: image_bchw})
result[0].shape
```




    (9, 57)




```python
def show_predictions_from_flat_format(image, predictions):
    image_index, pred_boxes, pred_scores, pred_joints = next(iter(iterate_over_flat_predictions(predictions, 1)))

    image = PoseVisualization.draw_poses(
        image=image, poses=pred_joints, scores=pred_scores, boxes=pred_boxes,
        edge_links=None, edge_colors=None, keypoint_colors=None, is_crowd=None
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.tight_layout()
    plt.show()
        
```


```python
show_predictions_from_flat_format(image, result)
```


    
![png](models_export_pose_files/models_export_pose_32_0.png)
    


### Changing postprocessing settings

You can control a number of parameters in the NMS settings as well as maximum number of detections per image before and after NMS step:

* IOU threshold for NMS - `nms_iou_threshold`
* Score threshold for NMS - `nms_score_threshold`
* Maximum number of detections per image before NMS - `max_detections_before_nms`
* Maximum number of detections per image after NMS - `max_detections_after_nms`

For sake of demonstration, let's export a model that would produce at most one detection per image with confidence threshold above 0.8 and NMS IOU threshold of 0.5. Let's use at most 100 predictions per image before NMS step:


```python
export_result = model.export(
    "yolo_nas_s_pose_top_1.onnx",
    confidence_threshold=0.8,
    nms_threshold=0.5,
    num_pre_nms_predictions=100,
    max_predictions_per_image=1,
    output_predictions_format=DetectionOutputFormatMode.FLAT_FORMAT
)

session = onnxruntime.InferenceSession(export_result.output,
                                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
inputs = [o.name for o in session.get_inputs()]
outputs = [o.name for o in session.get_outputs()]
result = session.run(outputs, {inputs[0]: image_bchw})

show_predictions_from_flat_format(image, result)
```


    
![png](models_export_pose_files/models_export_pose_34_0.png)
    


As expected, the predictions contains exactly one detection with the highest confidence score.

### Export of quantized model

You can export a model with quantization to FP16 or INT8. To do so, you need to specify the `quantization_mode` argument of `export()` method.

Important notes:
* Quantization to FP16 requires CUDA / MPS device available and would not work on CPU-only machines.

Let's see how it works:


```python
from super_gradients.conversion.conversion_enums import ExportQuantizationMode

export_result = model.export(
    "yolo_nas_pose_s_int8.onnx",
    confidence_threshold=0.5,
    output_predictions_format=DetectionOutputFormatMode.FLAT_FORMAT,
    quantization_mode=ExportQuantizationMode.INT8  # or ExportQuantizationMode.FP16
)

session = onnxruntime.InferenceSession(export_result.output,
                                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
inputs = [o.name for o in session.get_inputs()]
outputs = [o.name for o in session.get_outputs()]
result = session.run(outputs, {inputs[0]: image_bchw})

show_predictions_from_flat_format(image, result)
```


    
![png](models_export_pose_files/models_export_pose_37_0.png)
    


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
    "yolo_nas_pose_s_int8_with_calibration.onnx",
    confidence_threshold=0.5,
    output_predictions_format=DetectionOutputFormatMode.FLAT_FORMAT,
    quantization_mode=ExportQuantizationMode.INT8,
    calibration_loader=dummy_calibration_loader
)

session = onnxruntime.InferenceSession(export_result.output,
                                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
inputs = [o.name for o in session.get_inputs()]
outputs = [o.name for o in session.get_outputs()]
result = session.run(outputs, {inputs[0]: image_bchw})

show_predictions_from_flat_format(image, result)
```

     25%|█████████████████████████████████                                                                                                   | 4/16 [00:12<00:36,  3.02s/it]
    


    
![png](models_export_pose_files/models_export_pose_39_1.png)
    


### Limitations

* Dynamic batch size / input image shape is not supported yet. You can only export a model with a fixed batch size and input image shape.
* TensorRT of version 8.5.2 or higher is required.
* Quantization to FP16 requires CUDA / MPS device available.

## Conclusion

This concludes the export tutorial for YoloNAS-Pose pose estimation model. 
We hope you found it useful and will be able to use it to export your own models to ONNX format.

In case you have any questions or issues, please feel free to reach out to us at https://github.com/Deci-AI/super-gradients/issues.
