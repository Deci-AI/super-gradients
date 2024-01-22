# Assigning Learning Rates in SG
The `initial_lr` training hyperparameter allows you to specify different learning rates for different layers or groups of parameters in your neural network. This can be particularly useful for fine-tuning pre-trained models or when different parts of your model require different learning rate settings for optimal training.

## Using `initial_lr` as a Scalar:
When `initial_lr` is a single floating-point number, it sets a uniform learning rate for all model parameters. For example, `initial_lr` = 0.01:

```python
# Define training parameters
training_params = {
    "initial_lr": 0.01,
    "loss": "cross_entropy",
    # ... other training parameters
}

# Initialize the Trainer
trainer = Trainer("simple_net_training")

# Define model
model = 

# Define data loaders
train_dataloader = ...
test_dataloader = ...

# Train the model
trainer.train(model, training_params, train_dataloader, test_dataloader)
```


## Using `initial_lr` as a Mapping:
`initial_lr` can also be a mapping where keys are the prefixes of the named parameters of the model, and values are the learning rates
for those specific groups. This approach offers granular control over the learning rates for different parts of the model.

* Each key in the `initial_lr` dictionary acts as a prefix to match the named parameters in the model. The learning rate associated with a key is applied to all parameters whose names start with that prefix.

* The "default" key is essential, as it provides a fallback learning rate for any parameter that does not match other specified prefixes.
  
* Freezing parameters can be done by assigning a learning rate of 0 to a specific prefix. By doing so, you will be preventing them from being updated during training.


For example, in the below snippet `conv1` and `conv2` will be frozen, and `fc1` and `fc2` will be trained with an initial learning rate of 0.001:

```python
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(-1, 50 * 4 * 4)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

trainer = Trainer("simple_net_training")

# Define model
model = SimpleNet()

# Define data loaders
train_dataloader = ...
test_dataloader = ...

# Define training parameters
training_params = {
    "initial_lr": {"conv": 0.001, "default": 0.},
    "loss": "cross_entropy",
    # ... other training parameters
}

# Train the model
trainer.train(model, training_params, train_dataloader, test_dataloader)
```


## Fine-Tuning with the `finetune` Feature

The `finetune` parameter in SG adds another layer of control for model training. When set to `True`, it enables selective freezing of parts of the model, a technique often used in fine-tuning pre-trained models.
This feature is supported for all models in the SG model zoo that implement the `get_finetune_lr_dict` method. It is useful when one is not familiar with the different parts of the network.

For example, in the below the detection heads of YoloNAS will be trained with an initial learning rate of 0.01 while the rest of the network is frozen:

```python
trainer = Trainer("simple_net_training")

# Define model
model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco", num_classes=2)

# Define data loaders
train_dataloader = ...
test_dataloader = ...

# Define training parameters
training_params = {
    "initial_lr": 0.01,
    "finetune": True
    # ... other training parameters
}

# Train the model
trainer.train(model, training_params, train_dataloader, test_dataloader)
```




### How `finetune` Works

- When `finetune` is set to `True`, the model automatically freezes a part of itself based on the definitions in the `get_finetune_lr_dict` method.
- The `get_finetune_lr_dict` method returns a dictionary mapping learning rates to the unfrozen part of the network, in the same fashion as when `initial_lr` is used as a mapping.
  For example, the implementation for YoloNAS:
  ```python

        def get_finetune_lr_dict(self, lr: float):
        return {"heads": lr, "default": 0}
    ```
- If `initial_lr` is already a mapping, using `finetune` will raise an error. It's designed to work when `initial_lr` is unset or a float.
