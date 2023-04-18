# Learning Rate Scheduling

When training deep neural networks, it is often useful to reduce learning rate as the training progresses. This can be done by using pre-defined learning rate schedules or adaptive learning rate methods.
Learning rate scheduling type is controlled by the training parameter `lr_mode`. From `Trainer.train(...)` docs:

    `lr_mode` : str
        Learning rate scheduling policy, one of ['step','poly','cosine','function'].

        'step' refers to constant updates at epoch numbers passed through `lr_updates`. Each update decays the learning rate by `lr_decay_factor`.

        'cosine' refers to the Cosine Anealing policy as mentioned in https://arxiv.org/abs/1608.03983. The final learning rate ratio is controlled by `cosine_final_lr_ratio` training parameter.

        'poly' refers to the polynomial decrease: in each epoch iteration `self.lr = self.initial_lr * pow((1.0 - (current_iter / max_iter)), 0.9)`

        'function' refers to a user-defined learning rate scheduling function, that is passed through `lr_schedule_function`.

For example, the training code below will start with an initial learning rate of 0.1 and decay by 0.1 at epochs 100,150 and 200:

```python

from super_gradients.training import Trainer
...

trainer = Trainer("my_custom_scheduler_training_experiment")

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

## Using Custom LR Schedulers

Prerequisites: [phase callbacks](PhaseCallbacks.md), [training with configuration files](configuration_files.md).


In SG, learning rate schedulers are implemented as [phase callbacks](PhaseCallbacks.md).
They read the learning rate from the `PhaseContext` in their `__call__` method, calculate the new learning rate according to the current state of training, and update the optimizer's param groups.

For example, the code snippet from the previous section translates "lr_mode":"step" to a `super_gradients.training.utils.callbacks.callbacks.StepLRCallback` instance, which is added to the phase callbacks list.

### Implementing Your Own Scheduler
A custom learning rate scheduler should inherit from `LRCallbackBase`, so let's take a look at it:

```python

class LRCallbackBase(PhaseCallback):
    """
    Base class for hard coded learning rate scheduling regimes, implemented as callbacks.
    """

    def __init__(self, phase, initial_lr, update_param_groups, train_loader_len, net, training_params, **kwargs):
        super(LRCallbackBase, self).__init__(phase)
        self.initial_lr = initial_lr
        self.lr = initial_lr
        self.update_param_groups = update_param_groups
        self.train_loader_len = train_loader_len
        self.net = net
        self.training_params = training_params

    def __call__(self, context: PhaseContext, **kwargs):
        if self.is_lr_scheduling_enabled(context):
            self.perform_scheduling(context)

    def is_lr_scheduling_enabled(self, context: PhaseContext):
        """
        Predicate that controls whether to perform lr scheduling based on values in context.

        @param context: PhaseContext: current phase's context.
        @return: bool, whether to apply lr scheduling or not.
        """
        raise NotImplementedError

    def perform_scheduling(self, context: PhaseContext):
        """
        Performs lr scheduling based on values in context.

        @param context: PhaseContext: current phase's context.
        """
        raise NotImplementedError

    def update_lr(self, optimizer, epoch, batch_idx=None):
        if self.update_param_groups:
            param_groups = self.net.module.update_param_groups(optimizer.param_groups, self.lr, epoch, batch_idx, self.training_params, self.train_loader_len)
            optimizer.param_groups = param_groups
        else:
            # UPDATE THE OPTIMIZERS PARAMETER
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.lr
```

So when writing a custom scheduler, we need to override two methods:

1. `perform_scheduling`: This is where the new learning rate is calculated. The `lr` attribute is updated according. Then, in order to update the optimizer's parameter groups a call for `update_lr` should also be done (or update the optimizers parameter groups with your own logic explicitly).
2. `is_lr_scheduling_enabled`: Predicate that controls whether to perform lr scheduling based on values in context.

We will demonstrate how this is done by implementing a simple scheduler that decays the learning rate by a user-defined rate at user-defined epoch numbers.

```python
from super_gradients.training.utils.callbacks import LRCallbackBase, Phase
from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)

class UserStepLRCallback(LRCallbackBase):
    def __init__(self, lr_updates: list, lr_decay_factors: list, **kwargs):
        super(UserStepLRCallback, self).__init__(Phase.TRAIN_EPOCH_END, **kwargs)
        assert len(lr_updates) == len(lr_decay_factors)
        self.lr_updates = lr_updates
        self.lr_decay_factors = lr_decay_factors

    def perform_scheduling(self, context):
        curr_lr = self.initial_lr
        for epoch_idx, epoch_decay_rate in zip(self.lr_updates, self.lr_decay_factors):
            if epoch_idx <= context.epoch:
                curr_lr *= epoch_decay_rate
        self.lr = curr_lr
        self.update_lr(context.optimizer, context.epoch, None)

    def is_lr_scheduling_enabled(self, context):
        return self.training_params.lr_warmup_epochs <= context.epoch

```

Notes

- We specified that scheduling is enabled only after `lr_warmup_epochs`, this means that during lr warmup no updates will be done, even if such epoch is specifed!
- Notice the Phase.TRAIN_EPOCH_END which we pass to the constructor, this means that our `__call__` is triggered inside `on_train_loader_end(self, context)` (see [new callbacks API mapping between `Phase` to `Callback` methods](https://github.com/Deci-AI/super-gradients/blob/9d65cbbe5efc80b1db04d0aae081608dd91bce03/src/super_gradients/training/utils/callbacks/base_callbacks.py#L141).)

Now, we need to register our new scheduler so we can pass it through the `lr_mode` training parameter.
First we decorate our class with the `register_lr_scheduler`.
```python
# myscheduler.py

from super_gradients.training.utils.callbacks import LRCallbackBase, Phase
from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.registry import register_lr_scheduler
logger = get_logger(__name__)

@register_lr_scheduler("user_step")
class UserStepLRCallback(LRCallbackBase):
    def __init__(self, user_lr_updates: list, user_lr_decay_factors: list, **kwargs):
        super(UserStepLRCallback, self).__init__(Phase.TRAIN_EPOCH_END, **kwargs)
        assert len(user_lr_updates) == len(user_lr_decay_factors)
        self.lr_updates = user_lr_updates
        self.lr_decay_factors = user_lr_decay_factors

    def perform_scheduling(self, context):
        curr_lr = self.initial_lr
        for epoch_idx, epoch_decay_rate in zip(self.lr_updates, self.lr_decay_factors):
            if epoch_idx <= context.epoch:
                curr_lr *= epoch_decay_rate
        self.lr = curr_lr
        self.update_lr(context.optimizer, context.epoch, None)

    def is_lr_scheduling_enabled(self, context):
        return self.training_params.lr_warmup_epochs <= context.epoch

```

Next, simply import it (even if the class itself isn't used on the training script code page) to trigger the registry.

```python
# my_train_script.py

from super_gradients.training import Trainer
from myscheduler import UserStepLRCallback # triggers registry, now we can pass "lr_mode": "user_step"
...

```

And finally, use your new scheduler just as any other one supported by SG.
```python


trainer = Trainer("my_custom_scheduler_training_experiment")

train_dataloader = ...
valid_dataloader = ...
model = ...
train_params = {...
                "initial_lr": 0.1,
                "lr_mode": "user_step",
                "user_lr_updates": [100, 150, 200], # WILL BE PASSED TO UserStepLRCallback CONSTRUCTOR
                "user_lr_decay_factors": [0.1, 0.01, 0.001] # WILL BE PASSED TO UserStepLRCallback CONSTRUCTOR
                ...}

trainer.train(model=model, training_params=train_params, train_loader=train_dataloader, valid_loader=valid_dataloader)
   
```

Note that internally, Trainer unpacks [training_params to the scheduler callback constructor](https://github.com/Deci-AI/super-gradients/blob/537a0f0afe7bcf28d331fe2c0fa797fa10f54b99/src/super_gradients/training/sg_trainer/sg_trainer.py#L1078), so we pass scheduler related parameters through training_params as well.

### Using PyTorchs Native LR Schedulers (torch.optim.lr_scheduler)

PyTorch offers a [wide variety of learning rate schedulers](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate).
They can all be easily used by wrapping them up with `LRSchedulerCallback` and passing them as phase_callbacks.
For example:

```python
...

train_dataloader = ...
valid_dataloader = ...
model = ...

lr = 2.5e-4
optimizer = SGD(model.parameters(), lr=lr, weight_decay=0.0001)
step_lr_scheduler = MultiStepLR(optimizer, milestones=[0, 150, 200], gamma=0.1)

# Define phase callbacks
phase_callbacks = [
    LRSchedulerCallback(scheduler=step_lr_scheduler, phase=Phase.TRAIN_EPOCH_END),
]

# Bring everything together with Trainer and start training
trainer = Trainer("torch_schedulers_experiment")

train_params = {
    ...
    "phase_callbacks": phase_callbacks,
    "initial_lr": lr,
    "optimizer": optimizer,
    ...
}

trainer.train(model=net, training_params=train_params, train_loader=train_loader, valid_loader=valid_loader)

```
