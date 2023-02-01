## Model Zoo

SuperGradients provides an extensive collection of state-of-the-art (SOTA) models in its [model zoo](http://bit.ly/3EGfKD4).
These models are implemented as `torch.nn.Module` and can be used, customized, and trained like any other torch module.

The 3 main use cases of the Model Zoo are to
- Train a model from scratch
- Fine-tune a pre-trained model
- Use a model (pre-trained or not) as the backbone of a larger architecture.

### I. Instantiate a model

To instantiate a model, specify the model name and the number of classes desired.
```python
from super_gradients.training import models

# Instantiate resnet18 with head supporting 100 classes
default_resnet18 = models.get(model_name="resnet18", num_classes=100)
```

Every model name is accessible in to [model zoo](http://bit.ly/3EGfKD4),but can also be dynamically accessed through `super_gradients.common.object_names` for autocompletion
```python
from super_gradients.training import models
from super_gradients.common import object_names

# instantiate default pretrained resnet18
default_resnet18 = models.get(model_name=object_names.Models.RESNET18, num_classes=100)
```


### II. Instantiate a pretrained model
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

**With new head**
```python
from super_gradients.training import models

# Can be trained on a dataset of 94 classes
model = models.get(model_name="resnet18", num_classes=94, pretrained_weights="imagenet")
```



### III. Loading a Backbone
SuperGradients allows you to load a model as a backbone, i.e. without the global pooling stage and the classifier head. 

This is useful when using the model as the backbone for a larger architecture.

[CHECKUT BACKONE](...)
```python
from super_gradients.training import models

# instantiate pretrained resnet18, without classifier head. Output will be from the last stage before global pooling
backbone_resnet18 = models.get(model_name="resnet18", arch_params={"backbone_mode": True}, pretrained_weights="imagenet")
```








---
```python
from super_gradients.training import models

# instantiate default pretrained resnet18
default_resnet18 = models.get(model_name="resnet18", num_classes=100, pretrained_weights="imagenet")

# instantiate pretrained resnet18, without classifier head. Output will be from the last stage before global pooling
backbone_resnet18 = models.get(model_name="resnet18", arch_params={"backbone_mode": True}, pretrained_weights="imagenet")
```
