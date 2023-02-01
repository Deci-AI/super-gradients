# Optimizers

Optimization is a critical step in the deep learning process as it determines how well the network will learn from the training data.
SuperGradients supports out-of-the-box [pytorch optimizers](https://pytorch.org/docs/stable/optim.html#base-class) SGD, Adam and AdamW, but also 
[RMSpropTF](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) and 
[Lamb](https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/pytorch/lamb.py).

### How to use ?
Optimizers should be part of the training parameters.
SuperGradients takes care of associating the optimizer to the model and uses it to train the model.

```py
from super_gradients import Trainer

trainer = Trainer(...)

trainer.train(
    training_params={"optimizer": "Adam", "optimizer_params": {"eps": 1e-3}, ...}, 
    ...
)
```
