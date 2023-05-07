# Image Preprocessing and Postprocessing in Deep Learning
In order to run a prediction on an image, a few parameters are required:
- Class names: the model is trained to predict class ids, but to visualize results you will need to know the class names used to train the model.
- Processing parameters: the model requires specific input format.
- Task specific parameters: For Detection, this includes `IoU` and `Confidence` thresholds for instance.

SuperGradients handles all of this inside it's `model.predict()` method, but depending on the case, you might need to first explicitly set these parameters.


### 1. You trained your model a custom Dataset
In this case, the model is unaware you processing parameters, so you will need to set all of them using `model.set_dataset_processing_params()`.
Then you will be able to run `model.predict()`.

### 2. You are using pretrained weights, or trained on a SuperGradient's Datasets
Everything is automatically saved during training inside the model checkpoint, so you can run `model.predict()` **without** calling 
 `model.set_dataset_processing_params()` 


*For more information about the `model.predict()`, please check out the [following tutorial](DetectionPrediction.md)*

## Defining the parameters
### Class Names
This is very straight forward, as it simply consists of the list of classes used during the training. If you are loading the weights of a model fine-tuned on a new dataset for instance, this should be the classes of the dataset you fine-tuned on.

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
Just make sure to keep the same class order as during training.

### Processing 

Processing steps are required when running predictions. 
- **Image preprocessing** is required to prepare the input data for the model by applying various transformations, 
such as resizing, normalization, data augmentation, and channel reordering. 
These transformations ensure that the input data is in a format compatible with the model.
- **Image postprocessing** is required to process the model's output and convert it into a human-readable and interpretable format. 
This step may involve tasks such as converting class probabilities into class labels, in case of Object Detection, 
applying non-maximum suppression to remove duplicate detections, and rescaling the results to the original image size.

The module `super_gradients.training.processing` includes a wide range of `Processing` transformations,
which are responsible to both pre-processing the image and post-processing the predictions. 

For instance:`DetectionCenterPadding` will apply center padding to the image, 
but will also handle the reverse transformation on the image to remove the padding from the prediction.

You can combine multiple processing transformations using `ComposeProcessing`:
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
You can set default `iou` and `conf` values, which will be used when calling `model.predict()`.
- `iou`: IoU threshold for the non-maximum suppression (NMS) algorithm. If None, the default value associated with the training is used.
- `conf`: Confidence threshold. Predictions below this threshold are discarded. If None, the default value associated with the training is used.

## Saving your processing parameters to your model
Now that you defined all your parameters, you call `model.set_dataset_processing_params()` and then you'll be ready to use `model.predict()` 
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
