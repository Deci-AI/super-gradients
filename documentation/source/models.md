# Models

SuperGradients provides an extensive collection of state-of-the-art (SOTA) models in its [model zoo](http://bit.ly/3EGfKD4).
These models are implemented as `torch.nn.Module` and can be used, customized, and trained like any other torch module.

The 3 main use cases of the Model Zoo are to
- Train a model from scratch
- Fine-tune a pre-trained model
- Use a model (pre-trained or not) as the backbone of a larger architecture.

## Instantiating a model

To instantiate a model, specify the model name and the number of classes desired.
```python
from super_gradients.training import models

# Instantiate resnet18 with head supporting 100 classes
default_resnet18 = models.get(model_name="resnet18", num_classes=100)
```

All model names are available in the [model zoo](http://bit.ly/3EGfKD4),but can also be dynamically accessed through `super_gradients.common.object_names` for autocompletion
```python
from super_gradients.training import models
from super_gradients.common import object_names

# instantiate default pretrained resnet18
default_resnet18 = models.get(model_name=object_names.Models.RESNET18, num_classes=100)
```


## Instantiating a pretrained model
When loading a pre-trained model, SuperGradients also provides a pre-trained head by default. 
The head's dimension is determined by the number of classes in the dataset used for training.

If you're using a different dataset, you'll need to change the number of classes in the head. 
This keeps all the pre-trained weights of the model intact, except for the head which will be new and untrained. 
The model will not be able to predict accurately until fine-tuned.

**With pretrained head**
```python
from super_gradients.training import models

# Will reproduce the model zoo metrics on imagenet
model = models.get(model_name="resnet18", pretrained_weights="imagenet")
```
You can find the datasets used for pretraining our models in the [model zoo](http://bit.ly/3EGfKD4), and specify it in the `pretrained_weights`.

**With new head**
```python
from super_gradients.training import models

# Can be trained on a dataset of 94 classes
model = models.get(model_name="resnet18", num_classes=94, pretrained_weights="imagenet")
```



## Loading a Backbone
In deep learning, a backbone is a pre-trained neural network that serves as a starting point to build a larger architecture. 
It is typically a feature extractor trained on a large dataset and meant to capture important features of the data. 

When loading a model as a backbone in SuperGradients, you will get the model without the global pooling stage and the classifier head.

```python
from super_gradients.training import models

# instantiate pretrained resnet18, without classifier head. Output will be from the last stage before global pooling
backbone_resnet18 = models.get(model_name="resnet18", arch_params={"backbone_mode": True}, pretrained_weights="imagenet")
```

This backbone model can later be used as part of another model
```python
import torch

class CustomModel(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self._backbone = backbone
        self._head = ...

    def forward(self, x):
        out = self._backbone(x)
        out = self._head(out)
        return out

model = CustomModel(backbone=backbone_resnet18)
```


## Playing with the model architecture parameters 

All of SuperGradients model architectures can be parametrized using `arch_params`.
You can find the documentation about parameters of every architecture, and their default values, in the [recipes](https://github.com/Deci-AI/super-gradients/tree/master/src/super_gradients/recipes/arch_params).


In this example, we override the default params of [efficientnet_b0](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/recipes/arch_params/efficientnet_b0_arch_params.yaml)
```python
from super_gradients.training import models

arch_params = {
    "drop_connect_rate": 0.3,
    "image_size": 500,
}

yolox_custom = models.get(model_name="efficientnet_b0", arch_params=arch_params, num_classes=15)
```
