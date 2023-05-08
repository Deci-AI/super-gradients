# Prediction Set-Up

To make accurate predictions on images, several parameters must be provided:
- Class names: The model predicts class IDs, but to visualize results, the class names from the training dataset are needed.
- Processing parameters: The model requires input data in a specific format.
- Task-specific parameters: For instance, in the case of Detection, this includes `IoU` and `Confidence` thresholds.

SuperGradients manages all of these within its `model.predict()` method, but in certain scenarios, you might need to set these parameters explicitly first.

### 1. Training your model on a custom dataset
If you trained a model on a dataset that **does not** inherit from any of the SuperGradients datasets, you will need to set the processing parameters explicitly. To do this, use the `model.set_dataset_processing_params()` method. Once you've set the parameters, you can run `model.predict()`.

### 2. Using pretrained weights or training on a SuperGradient's dataset
All necessary information is automatically saved during training within the model checkpoint, so you can run `model.predict()` **without** calling `model.set_dataset_processing_params()`.

*For more details about `model.predict()`, please refer to the [related tutorial](DetectionPrediction.md).*


## Set-up parameters
### Class Names
This is straightforward as it corresponds to the list of classes used during training. For instance, if you're loading the weights of a model fine-tuned on a new dataset, use the classes from that dataset.

```python
class_names = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    ...
]
```
Ensure that the class order remains the same as during training.

### Processing 

Processing steps are necessary for making predictions.
- **Image preprocessing** prepares the input data for the model by applying various transformations, such as resizing, normalization, and channel reordering. These transformations ensure the input data is compatible with the model.
- **Image postprocessing** processes the model's output and converts it into a human-readable and interpretable format. This step may include tasks like converting class probabilities into class labels, applying non-maximum suppression to eliminate duplicate detections, and rescaling results to the original image size.

The `super_gradients.training.processing` module contains a wide range of `Processing` transformations responsible for both image preprocessing and postprocessing. 

For example, `DetectionCenterPadding` applies center padding to the image while also handling the reverse transformation to remove padding from the prediction.

Multiple processing transformations can be combined using `ComposeProcessing`:
```python
from super_gradients.training.processing import DetectionCenterPadding, StandardizeImage, NormalizeImage, ImagePermute, ComposeProcessing, DetectionLongestMaxSizeRescale

image_processor = ComposeProcessing(
    [
        DetectionLongestMaxSizeRescale(output_shape=(636, 636)),
        DetectionCenterPadding(output_shape=(640, 640), pad_value=114),
        StandardizeImage(max_value=255.0),
        ImagePermute(permutation=(2, 0, 1)),
    ]
)
```

### Task Specific parameters

#### Detection
Default `iou` and `conf` values can be set, which will be used when calling `model.predict()`.
- `iou`: IoU threshold for the non-maximum suppression (NMS) algorithm. If None, the default value associated with training is used.
- `conf`: Confidence threshold. Predictions below this threshold are discarded. If None, the default value associated with training is used.

## Saving your processing parameters to your model
After defining all parameters, call `model.set_dataset_processing_params()` and then use `model.predict()`.
```python
from super_gradients.common.object_names import Models
from super_gradients.training import models

model = models.get(Models.YOLO_NAS_L, checkpoint_path="/path/to/checkpoint")

model.set_dataset_processing_params(
    class_names=class_names,
    image_processor=image_processor,
    iou=0.35, conf=0.25,
)

IMAGES = [...]

images_predictions = model.predict(IMAGES)
```

*For more information about the `model.predict()`, please check out the [following tutorial](DetectionPrediction.md).*
