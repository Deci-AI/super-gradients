# Third-party experiment monitoring

SuperGradients supports out-of-the-box Weights & Biases (wandb) and ClearML. 


Simply update your train parameters to specify your preferred third-party logging tool, and the Trainer will 
take care of the rest, handling all the necessary logic and integration.

### Weights & Biases
**requirements**:
- Install wandb
- Set up wandb according to the [official documentation](https://docs.wandb.ai/quickstart#1.-set-up-wandb)
- Adapt your code like in the following example

```python
from super_gradients import Trainer

# create a trainer object, look the declaration for more parameters
trainer = Trainer("experiment_name")
model = ...

training_params = {
    ...                             # Your training params
    "sg_logger": "wandb_sg_logger", # Weights&Biases Logger, see class super_gradients.common.sg_loggers.wandb_sg_logger.WandBSGLogger for details
    "sg_logger_params":             # Params that will be passes to __init__ of the logger super_gradients.common.sg_loggers.wandb_sg_logger.WandBSGLogger
      {
        "project_name": "project_name", # W&B project name
        "save_checkpoints_remote": True,
        "save_tensorboard_remote": True,
        "save_logs_remote": True,
      }
}

trainer.train(model=model, training_params=training_params, ...)
```


### ClearML
**requirements**:
- Install clearml
- Set up cleaml according to the [official documentation](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps#install-clearml)
- Adapt your code like in the following example

```python
from super_gradients import Trainer

# create a trainer object, look the declaration for more parameters
trainer = Trainer("experiment_name")
model = ...

training_params = {
    ...                                 # Your training params
    "sg_logger": "clearml_sg_logger",   # ClearML Logger, see class super_gradients.common.sg_loggers.wandb_sg_logger.ClearMLSGLogger for details
    "sg_logger_params":                 # Params that will be passes to __init__ of the logger super_gradients.common.sg_loggers.wandb_sg_logger.ClearMLSGLogger 
      {
        "project_name": "project_name", # ClearML project name
        "save_checkpoints_remote": True,
        "save_tensorboard_remote": True,
        "save_logs_remote": True,
      } 
}

trainer.train(model=model, training_params=training_params, ...)
```


## Uploading custom objects with a callback
Callbacks are the way to go when it comes to insert small pieces of code into the training/validation loop of SuperGradients.
For more information, please check out our tutorial on [how to use callbacks in SuperGradients](TODO:add_link)

In this example, we create a callback to upload images with our `sg_logger`.
If you didn't specify a `sg_logger` in your training params, these images will only be added to your tensorboard. 
If you are working with WandB or ClearML, this will also be uploaded to the monitoring service you chose.

```python
from typing import List

import cv2
import numpy as np

from super_gradients.training.utils.callbacks.base_callbacks import PhaseContext, Callback
from super_gradients.training.utils.detection_utils import DetectionVisualization


class DetectionVisualizationCallback(Callback):
    """Visualize the last batch of each validation epoch"""

    def __init__(self, classes: list, n_images: int = 10):
        super(Callback, self).__init__()
        self.classes = classes
        self.n_images = n_images

    def on_validation_loader_end(self, context: PhaseContext):
        # preds = (context.preds[0].clone(), None)

        batch_imgs: List[np.ndarray] = DetectionVisualization.visualize_batch(
            image_tensor=context.inputs[: self.n_images],
            pred_boxes=context.preds[: self.n_images],
            target_boxes=context.target[: self.n_images],
            batch_name='Validation epoch',
            class_names=self.classes,
        )
        batch_imgs: np.ndarray = np.stack([cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in batch_imgs])
        
        # The important part for this example: uploading an image
        context.sg_logger.add_images(
            tag="val_epoch_end_images",
            images=batch_imgs,
            global_step=context.epoch,
            data_format="NHWC",
        )
```

The sg_logger can also be used to upload files, text, scalars (such as metrics), checkpoints, ...

We encourage you to check out the documentation of `super_gradients.common.sg_loggers.base_sg_logger.BaseSGLogger` since every `sg_logger` inherits from it.
