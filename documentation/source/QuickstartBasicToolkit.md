# The Basic Skills of SG

Learn the basics of model development with SuperGradients. Researchers and machine learning engineers should start here.

##1. Train a Model

0. Imports:

```python
from super_gradients.common.object_names import Models
from super_gradients.training import Trainer, models
from super_gradients.training.metrics.classification_metrics import Accuracy, Top5
from super_gradients.training.dataloaders.dataloaders import cifar10_train, cifar10_val
from super_gradients.training.utils.distributed_training_utils import setup_device
```

1. Call [setup_device()](https://github.com/Deci-AI/super-gradients/blob/master/documentation/source/device.md) according to your available hardware and needs:

```python
setup_device("cpu")
```

2. Instantiate a [Trainer]() object #TODO: ADD TRAINER API LINK
```python

trainer = Trainer(experiment_name="my_cifar_experiment", ckpt_root_dir="/path/to/checkpoints_directory/")
```

3. [Instantiate a model](https://github.com/Deci-AI/super-gradients/blob/master/documentation/source/models.md):
```python
model = models.get(Models.RESNET18, num_classes=10)
```

4. Define [metrics](https://github.com/Deci-AI/super-gradients/blob/master/documentation/source/Metrics.md) and other [training parameters](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/recipes/training_hyperparams/default_train_params.yaml):
```
training_params = {
    "max_epochs": 20,
    "initial_lr": 0.1,
    "loss": "cross_entropy",
    "train_metrics_list": [Accuracy(), Top5()],
    "valid_metrics_list": [Accuracy(), Top5()],
    "metric_to_watch": "Accuracy",
    "greater_metric_to_watch_is_better": True,
}
```

5. Instantiate [PyTorch data loaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#preparing-your-data-for-training-with-dataloaders) for training and validation:
```python
train_loader=cifar10_train()
valid_loader=cifar10_val()
```

6. Launch training:
```python
trainer.train(model=model, training_params=training_params, train_loader=train_loader, valid_loader=valid_loader)
```
##2. Test a Model

0. Imports:

```python
from super_gradients.common.object_names import Models
from super_gradients.training import Trainer, models
from super_gradients.training.metrics.classification_metrics import Accuracy, Top5
from super_gradients.training.dataloaders.dataloaders import cifar10_val
from super_gradients.training.utils.distributed_training_utils import setup_device
```
1. Call [setup_device()](https://github.com/Deci-AI/super-gradients/blob/master/documentation/source/device.md) according to your available hardware and needs:

```python
setup_device("cpu")
```

2. Instantiate a [Trainer]() object #TODO: ADD TRAINER API LINK
```python
trainer = Trainer(experiment_name="test_my_cifar_experiment", ckpt_root_dir="/path/to/checkpoints_directory/")

```

3. [Instantiate a model](https://github.com/Deci-AI/super-gradients/blob/master/documentation/source/models.md) and load weights to it. 
   
Learn more about the different options for loading model weights from our [checkpoints tutorial](https://github.com/Deci-AI/super-gradients/blob/master/documentation/source/Checkpoints.md).
```python
model = models.get(Models.RESNET18, num_classes=10, checkpoint_path="/path/to/checkpoints_directory/my_cifar_experiment/ckpt_best.pth")
```



4. Define [metrics](https://github.com/Deci-AI/super-gradients/blob/master/documentation/source/Metrics.md) for test:
```python
test_metrics = [Accuracy(), Top5()]
```

5. Instantiate a [PyTorch data loader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#preparing-your-data-for-training-with-dataloaders) for testing:

```python
test_data_loader = cifar10_val()
```

6. Launch test:

```python
accuracy, top5 = trainer.test(model=model, test_loader=test_data_loader, test_metrics_list=test_metrics)
print(f"Test results: Accuracy: {accuracy}, Top5: {top5}")
```
##3. Use Pre-trained Models

0. Imports:

```python
from super_gradients.common.object_names import Models
from super_gradients.training import Trainer, models
from super_gradients.training.metrics.classification_metrics import Accuracy, Top5
from super_gradients.training.dataloaders.dataloaders import cifar10_train, cifar10_val

```
1. Call [setup_device()](https://github.com/Deci-AI/super-gradients/blob/master/documentation/source/device.md) according to your available hardware and needs:

```python
setup_device("cpu")
```

2. Instantiate a pre-trained model from SGs [model zoo](http://bit.ly/3EGfKD4):

```python
model = models.get(Models.RESNET18, num_classes=10, pretrained_weights="imagenet")
```

Or use your own local weights to instantiate a pre-trained model:

```python
model = models.get(Models.RESNET18, num_classes=10, checkpoint_path="/path/to/imagenet_checkpoint.pth", checkpoint_num_classes=1000)
```

Finetune or test your pre-trained model as done in the previous sections.



##4. Predict

0. Imports:

```python
from PIL import Image
import numpy as np
import requests
from super_gradients.training import models
from super_gradients.common.object_names import Models
import torchvision.transforms as T
import torch
from super_gradients.training.utils.distributed_training_utils import setup_device

```
1. Call [setup_device()](https://github.com/Deci-AI/super-gradients/blob/master/documentation/source/device.md) according to your available hardware and needs:

```python
setup_device("cpu")
```

2. [Instantiate a model](https://github.com/Deci-AI/super-gradients/blob/master/documentation/source/models.md), load weights to it and put it in `eval` mode: 

```python

# Load the best model that we trained
best_model = models.get(Models.RESNET18, num_classes=10,checkpoint_path="/path/to/checkpoints_directory/my_cifar_experiment/ckpt_best.pth")
best_model.eval()
```

3. Create input data and preprocess it:
```python
url = "https://www.aquariumofpacific.org/images/exhibits/Magnificent_Tree_Frog_900.jpg"
image = np.array(Image.open(requests.get(url, stream=True).raw))

transforms = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    T.Resize((32, 32))
    ])
input_tensor = transforms(image).unsqueeze(0).to(next(best_model.parameters()).device)
```

4. Predict and visualize results:
```python
predictions = best_model(input_tensor)

classes = train_dataloader.dataset.classes
plt.xlabel(classes[torch.argmax(predictions)])
plt.imshow(image)
```


<img src="./images/frog_prediction.png" width="500">
