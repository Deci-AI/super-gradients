# DetectionOutputAdapter

The DetectionOutputAdapter is a class that converts the output of a detection model into a user-appropriate format. 
For instance, it can be used to convert the format of bounding boxes from CYXHW to XYXY, or to change the layout of the elements 
in the output tensor from [X1, Y1, X2, Y2, Confidence, Class] to [Class, Confidence, X1, Y1, X2, Y2].

## Features

* Easy rearrangement of the elements in the output tensor
* Easy conversion of the bounding box format
* Support of JIT Tracing & Scripting
* Support of ONNX export


## Usage

We start by introducing the concept of a `format`. A `format` represents a specific layout of the elements in the output tensor.
Currently, there is only one type of formats supported - `ConcatenatedTensorFormat` which represents a layout where all predictions concatenated into a single tensor.
Additional formats can be added in the future (Like `DictionaryOfTensorsFormat`).

`ConcatenatedTensorFormat` requires that input is a tensor and has the following shape: 
* Tensor of shape [N, Elements] - `N` is the number of predictions, `Elements` is the concatenated vector of attributes per box.
* Tensor of shape [B, N, Elements] - `B` is the batch dimension, `N` and `Elements` as above. 

To instantiate the `DetectionOutputAdapter` we have to describe the input and output formats for our predictions:

Let's imagine model emits predictions in the following format:

```python
# [N, 10] (cx, cy, w, h, class, confidence, attributes..)
example_input = [
    #      cx          cy        w          h     class, confidence,   attribute a, attribute b, attribute c, attribute d
    [0.465625,  0.5625,    0.13125,   0.125,          0,      0.968,         0.350,       0.643,       0.640,       0.453],
    [0.103125,  0.1671875, 0.10625,   0.134375,       1,      0.897,         0.765,       0.654,       0.324,       0.816],
    [0.078125,  0.078125,  0.15625,   0.15625,        2.,     0.423,         0.792,       0.203,       0.653,       0.777],
    ...
]
```

The corresponding format definition would look like this:

```python
from super_gradients.training.datasets.data_formats import ConcatenatedTensorFormat, BoundingBoxesTensorSliceItem, TensorSliceItem, NormalizedCXCYWHCoordinateFormat

input_format = ConcatenatedTensorFormat(
    layout=(
        BoundingBoxesTensorSliceItem(name="bboxes", format=NormalizedCXCYWHCoordinateFormat()),
        TensorSliceItem(name="class", length=1),
        TensorSliceItem(name="confidence", length=1),
        TensorSliceItem(name="attributes", length=4),
    )
)
```

For sake of demonstration, let's assume that we want to convert the output to the following format:

```python
# [N, 10] (class, attributes, x1, y1, x2, y2)
[
    # class, attribute a, attribute b, attribute c, attribute d,     x1,   y1,   x2,    y2
    [     0,       0.350,       0.643,       0.640,       0.453,    256,  320,  340,   400],
    [     1,       0.765,       0.654,       0.324,       0.816,     32,   64,  100,   150],
    [     2,       0.792,       0.203,       0.653,       0.777,      0,    0,  100,   100],
    ...
]
```

* The `class` and `attributes` are the same as in the input format but comes first
* The format of bounding boxes is changed from `NormalizedCXCYWHCoordinateFormat` to `XYXYCoordinateFormat`
* The `confidence` is removed from the output

The corresponding format definition would look like this:

```python
from super_gradients.training.datasets.data_formats import ConcatenatedTensorFormat, BoundingBoxesTensorSliceItem, TensorSliceItem, XYXYCoordinateFormat

output_format = ConcatenatedTensorFormat(
    layout=(
        TensorSliceItem(name="class", length=1),
        TensorSliceItem(name="attributes", length=4),
        BoundingBoxesTensorSliceItem(name="bboxes", format=XYXYCoordinateFormat()),
    )
)
```

Now we can construct the `DetectionOutputAdapter` and attach it to the model:

```python
from super_gradients.training.datasets.data_formats import DetectionOutputAdapter

output_adapter = DetectionOutputAdapter(input_format, output_format, image_shape=(640,640))

model = nn.Sequential(
    create_model(),
    create_nms(),
    output_adapter
)
```

To test how the output adapter transforms dummy input one can easily run it alone:

```python
output = output_adapter(torch.from_numpy(example_input)).numpy()
print(output)

# Prints:
[
    # class,   attribute a, attribute b, attribute c, attribute d,     x1,   y1,   x2,    y2
    [     0,         0.350,       0.643,       0.640,       0.453,    256,  320,  340,   400], 
    [     1,         0.765,       0.654,       0.324,       0.816,     32,   64,  100,   150], 
    [     2,         0.792,       0.203,       0.653,       0.777,      0,    0,  100,   100]
]
```


## Not supported features

Currently `DetectionOutputAdapter` does not support the following features:

* `argmax` operation over a slice of confidences for [C] classes (Useful to compute `argmax(class confidences)`)
* Multiplication of two slices (Useful to compute `confidence * class`)
