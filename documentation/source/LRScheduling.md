# Learning Rate Scheduling in SG

When training deep neural networks, it is often useful to reduce learning rate as the training progresses. This can be done by using pre-defined learning rate schedules or adaptive learning rate methods.
Learning rate scheduling type is controlled by the training parameter `lr_mode`. From `Trainer.train(...)` docs:

    `lr_mode` : str
        Learning rate scheduling policy, one of ['step','poly','cosine','function'].

        'step' refers to constant updates at epoch numbers passed through `lr_updates`. Each update decays the learning rate by `lr_decay_factor`.

        'cosine' refers to the Cosine Anealing policy as mentioned in https://arxiv.org/abs/1608.03983. The final learning rate ratio is controlled by `cosine_final_lr_ratio` training parameter.

        'poly' refers to the polynomial decrease: in each epoch iteration `self.lr = self.initial_lr * pow((1.0 - (current_iter / max_iter)), 0.9)`

        'function' refers to a user-defined learning rate scheduling function, that is passed through `lr_schedule_function`.

Examples:

```python

from super_gradients.training import Trainer
...

trainer = Trainer("my_custom_dataset_training_experiment")

train_dataloader = ...
valid_dataloader = ...
model = ...
train_params = {...
                "initial_lr": 0.1,
                "lr_mode":"step",
                "lr_updates": [100, 150, 200],
                "lr_decay_factor": 0.1,
                ...}

trainer.train(model=model, training_params=train_params, train_loader=train_dataloader, valid_loader=valid_dataloader)
   
```
