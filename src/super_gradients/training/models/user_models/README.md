<div "center">
  <img src="documentation/assets/SG_img/SG - Horizontal.png" width="600"/>
 <br/><br/>

## Introduction
This page demonstrates how you can register your own models, so that SuperGradients can access it with a name `str`, for
example, when training from a recipe config `architecture: my_custom_model`.

## Usage
1. Create a new Python module in this folder (e.g. `.../user_models/my_model.py`).
2. Define your PyTorch model (`torch.nn.Module`) in the new module.
3. Import the `@register` decorator 
`from super_gradients.training.models.model_registry import register` and apply it to your model.
   * The decorator can be applied directly to the class or to a function returning the class.
   * The decorator takes an optional `name: str` argument. If not specified, the decorated class/function name will be registered.
   
## Example

```python
import torch.nn as nn
import torch.nn.functional as F

from super_gradients.training.utils.registry import register_model


@register_model('my_conv_net')  # will be registered as "my_conv_net"
class MyConvNet(nn.Module):
   def __init__(self, num_classes):
      super().__init__()
      self.conv1 = nn.Conv2d(3, 6, 5)
      self.pool = nn.MaxPool2d(2, 2)
      self.conv2 = nn.Conv2d(6, 16, 5)
      self.fc1 = nn.Linear(16 * 5 * 5, 120)
      self.fc2 = nn.Linear(120, 84)
      self.fc3 = nn.Linear(84, num_classes)

   def forward(self, x):
      x = self.pool(F.relu(self.conv1(x)))
      x = self.pool(F.relu(self.conv2(x)))
      x = torch.flatten(x, 1)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      return x
```
or
```python
@register_model()
def myconvnet_for_cifar10(): # will be registered as "myconvnet_for_cifar10"
    return MyConvNet(num_classes=10)
```




  
