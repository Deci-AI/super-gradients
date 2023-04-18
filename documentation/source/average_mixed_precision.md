# Automatic Mixed Precision (AMP)
Automatic mixed precision (AMP) is a feature in PyTorch that enables the use of lower-precision data types, such as float16, in deep learning models for improved memory and computation efficiency. 
It automatically casts the model's parameters and buffers to a lower-precision data type, and dynamically rescales the activations to prevent underflow or overflow. 


## Set up AMP
To use `AMP` in SuperGradients, you simply need to set `mixed_precision=True` in your training_params.

**In python script**
```python
from super_gradients import Trainer

trainer = Trainer("experiment_name")
model = ...

training_params = {"mixed_precision": True, ...:...}

trainer.train(model=model, training_params=training_params, ...)
```

**In recipe**
```yaml
# my_training_hyperparams.yaml

mixed_precision: True # Whether to use mixed precision or not.
```
