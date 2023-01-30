## Model Zoo
SuperGradients provides an extensive [SOTA model zoo](http://bit.ly/3EGfKD4).
Each of these model and checkpoint can be seamlessly loaded using the `super_gradients.training.models` api.  

### Instantiate a model
You can instantiate a model just by specifying the model name and the number of classes you want.
```python
from super_gradients.training import models

# instantiate default pretrained resnet18
default_resnet18 = models.get(model_name="resnet18", num_classes=100)
```

You can find the model names in to model zoo, but if you prefer you can always access to the model names dynamically using `super_gradients.common.object_names`.
```python
from super_gradients.training import models
from super_gradients.common import object_names

# instantiate default pretrained resnet18
default_resnet18 = models.get(model_name=object_names.Models.RESNET18, num_classes=100)
```
*Note: This allows autocomplete!*


### Instantiate a pretrained model
Loading a pretrained model includes by default the pretrained head. 
When working on the dataset used for pretraining - for instance if you are just testing the model - you might want to keep these heads to get the expected accuracy.
When working on another dataset you will need to set num_classes=100 which will load all the model weights except for the head which will be initialized randomly.


#### Same head
```python
from super_gradients.training import models

# Usually for testing
model = models.get(model_name="resnet18", pretrained_weights="imagenet")
```
#### Replace head
```python
from super_gradients.training import models

# Can be trained on a dataset of 100 classes
model = models.get(model_name="resnet18", num_classes=100, pretrained_weights="imagenet")
```



### Load backbone
SuperGradients allows you to load a model as a backbone, i.e. before the global pooling stage and the classifier head.
This comes handy if you want to use it as the backbone of another model.
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
