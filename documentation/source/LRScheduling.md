# Learning Rate Scheduling

When training deep neural networks, it is often useful to reduce learning rate as the training progresses. This can be done by using pre-defined learning rate schedules or adaptive learning rate methods.
Learning rate scheduling type is controlled by the training parameter `lr_mode`. From `Trainer.train(...)` docs:

    `lr_mode` : Union[str, Mapping]

        When str:

        Learning rate scheduling policy, one of ['StepLRScheduler','PolyLRScheduler','CosineLRScheduler','FunctionLRScheduler'].

        'StepLRScheduler' refers to constant updates at epoch numbers passed through `lr_updates`. Each update decays the learning rate by `lr_decay_factor`.

        'CosineLRScheduler' refers to the Cosine Anealing policy as mentioned in https://arxiv.org/abs/1608.03983. The final learning rate ratio is controlled by `cosine_final_lr_ratio` training parameter.

        'PolyLRScheduler' refers to the polynomial decrease: in each epoch iteration `self.lr = self.initial_lr * pow((1.0 - (current_iter / max_iter)), 0.9)`

        'FunctionLRScheduler' refers to a user-defined learning rate scheduling function, that is passed through `lr_schedule_function`.

For example, the training code below will start with an initial learning rate of 0.1 and decay by 0.1 at epochs 100,150 and 200:

```python
from super_gradients.training import Trainer


trainer = Trainer("my_custom_scheduler_training_experiment")

train_dataloader = ...
valid_dataloader = ...
model = ...
train_params = {
    "initial_lr": 0.1,
    "lr_mode":"StepLRScheduler",
    "lr_updates": [100, 150, 200],
    "lr_decay_factor": 0.1,
    ...,
}

trainer.train(model=model, training_params=train_params, train_loader=train_dataloader, valid_loader=valid_dataloader)
```

<details>
<summary>Equivalent in a <code>.yaml</code> configuration file:</summary>

```yaml
training_hyperparams:
    initial_lr: 0.1
    lr_mode: StepLRScheduler
    user_lr_updates:
      - 100
      - 150
      - 200
    lr_decay_factor: 0.1
    ...

...
```
</details>


## Using Custom LR Schedulers

Prerequisites: [phase callbacks](PhaseCallbacks.md), [training with configuration files](configuration_files.md).


In SG, learning rate schedulers are implemented as [phase callbacks](PhaseCallbacks.md).
They read the learning rate from the `PhaseContext` in their `__call__` method, calculate the new learning rate according to the current state of training, and update the optimizer's param groups.

For example, the code snippet from the previous section translates "lr_mode":"StepLRScheduler" to a `super_gradients.training.utils.callbacks.callbacks.StepLRScheduler` instance, which is added to the phase callbacks list.

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

# The following code sections marked with '...' are placeholders 
# indicating additional necessary code that is not shown for simplicity.
train_dataloader = ...
valid_dataloader = ...
model = ...

train_params = {
    "initial_lr": 0.1,
    "lr_mode": "user_step",
    "user_lr_updates": [100, 150, 200], # WILL BE PASSED TO UserStepLRCallback CONSTRUCTOR
    "user_lr_decay_factors": [0.1, 0.01, 0.001], # WILL BE PASSED TO UserStepLRCallback CONSTRUCTOR
    ...
}

trainer.train(model=model, training_params=train_params, train_loader=train_dataloader, valid_loader=valid_dataloader)
```

Note that internally, Trainer unpacks [training_params to the scheduler callback constructor](https://github.com/Deci-AI/super-gradients/blob/537a0f0afe7bcf28d331fe2c0fa797fa10f54b99/src/super_gradients/training/sg_trainer/sg_trainer.py#L1078), so we pass scheduler related parameters through training_params as well.


<details>
<summary>Equivalent in a <code>.yaml</code> configuration file:</summary>

```yaml
training_hyperparams:
    initial_lr: 0.1
    lr_mode: user_step
    user_lr_updates: # WILL BE PASSED TO UserStepLRCallback CONSTRUCTOR
      - 100
      - 150
      - 200
    user_lr_decay_factors: # WILL BE PASSED TO UserStepLRCallback CONSTRUCTOR
      - 0.1
      - 0.01
      - 0.001
    ...

...
```
</details>

### Using PyTorchs Native LR Schedulers (torch.optim.lr_scheduler)

PyTorch offers a [wide variety of learning rate schedulers](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate).
They can all be easily used by passing a Mapping through the lr_mode parameter, following aa simple API.
From `Trainer.train(...)` docs:

    When Mapping, refers to a torch.optim.lr_scheduler._LRScheduler, following the below API:
    
        lr_mode = {LR_SCHEDULER_CLASS_NAME: {**LR_SCHEDULER_KWARGS, "phase": XXX, "metric_name": XXX)
    
        Where "phase" (of Phase type) controls when to call torch.optim.lr_scheduler._LRScheduler.step().
        For instance, in order to:
        - Update LR on each batch: Use phase: Phase.TRAIN_BATCH_END
        - Update LR after each epoch: Use phase: Phase.TRAIN_EPOCH_END
    
        The "metric_name" refers to the metric to watch (See docs for "metric_to_watch" in train(...)
         https://docs.deci.ai/super-gradients/docstring/training/sg_trainer.html) when using
          ReduceLROnPlateau. In any other case this kwarg is ignored.
    
        **LR_SCHEDULER_KWARGS are simply passed to the torch scheduler's __init__.

    
    
        For example:
            lr_mode = {"StepLR": {"gamma": 0.1, "step_size": 1, "phase": Phase.TRAIN_EPOCH_END}}
            is equivalent to following training code:
            
                from torch.optim.lr_scheduler import StepLR
                ...
                optimizer = ....
                scheduler = StepLR(optimizer=optimizer, gamma=0.1, step_size=1)
    
                for epoch in num_epochs:
                    train_epoch(...)
                    scheduler.step()
                    ....

### Examples
Using `StepLR`

```python
trainer = Trainer("torch_Scheduler_example")

# The following code sections marked with '...' are placeholders 
# indicating additional necessary code that is not shown for simplicity.
train_dataloader = ...
valid_dataloader = ...
model = ...

train_params = {
    "max_epochs": 2,
    "lr_mode": {"StepLR": {"gamma": 0.1, "step_size": 1, "phase": Phase.TRAIN_EPOCH_END}},
    "lr_warmup_epochs": 0,
    "initial_lr": 0.1,
    "loss": torch.nn.CrossEntropyLoss(),
    "optimizer": "SGD",
    "criterion_params": {},
    "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
    "train_metrics_list": [Accuracy()],
    "valid_metrics_list": [Accuracy()],
    "metric_to_watch": "Accuracy",
    "greater_metric_to_watch_is_better": True,
}
trainer.train(model=model, training_params=train_params, train_loader=dataloader, valid_loader=dataloader)
```

<details>
<summary>Equivalent in a <code>.yaml</code> configuration file:</summary>

```yaml
training_hyperparams:
    # Setting up LR Scheduler
    lr_mode:
      StepLR:
        gamma: 0.1
        step_size: 1
        phase: TRAIN_EPOCH_END
    
    # Setting up other parameters
    max_epochs: 2
    lr_warmup_epochs: 0
    initial_lr: 0.1
    loss: CrossEntropyLoss
    optimizer: SGD
    criterion_params: {}
    optimizer_params:
      weight_decay: 1e-4
      momentum: 0.9
    train_metrics_list:
      - Accuracy
    valid_metrics_list:
      - Accuracy
    metric_to_watch: Accuracy
    greater_metric_to_watch_is_better: true

...
```
</details>
  

**Using `ReduceLROnPlateau`**

If you choose to use `ReduceLROnPlateau` as the learning rate scheduler, you need to specify a `metric_name`. 
This parameter follows the same guidelines as `metric_to_watch`. 

For an in-depth understanding of these metrics, 
see the [metrics guide](Metrics.md).


```python
trainer = Trainer("torch_ROP_Scheduler_example")
train_dataloader = ...
valid_dataloader = ...
model = ...
train_params = {
    "max_epochs": 2,
    "lr_decay_factor": 0.1,
    "lr_mode": {
        "ReduceLROnPlateau": {"patience": 0, "phase": Phase.TRAIN_EPOCH_END, "metric_name": "DummyMetric"}},
    "lr_warmup_epochs": 0,
    "initial_lr": 0.1,
    "loss": torch.nn.CrossEntropyLoss(),
    "optimizer": "SGD",
    "criterion_params": {},
    "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
    "train_metrics_list": [Accuracy()],
    "valid_metrics_list": [Accuracy()],
    "metric_to_watch": "DummyMetric",
    "greater_metric_to_watch_is_better": True,
}
trainer.train(model=model, training_params=train_params, train_loader=dataloader, valid_loader=dataloader)
```

The scheduler's `state_dict` is saved under `torch_scheduler_state_dict` entry inside the checkpoint during training,
allowing us to resume from the same state of the scheduling.

<details>
<summary>Equivalent in a <code>.yaml</code> configuration file:</summary>

```yaml
training_hyperparams:
    # Setting up LR Scheduler
    lr_mode:
      ReduceLROnPlateau:
        patience: 0
        phase: TRAIN_EPOCH_END
        metric_name: DummyMetric

    # Setting up other parameters
    max_epochs: 2
    lr_decay_factor: 0.1
    lr_warmup_epochs: 0
    initial_lr: 0.1
    loss: CrossEntropyLoss
    optimizer: SGD
    criterion_params: {}
    optimizer_params:
      weight_decay: 1e-4
      momentum: 0.9
    train_metrics_list:
      - Accuracy
    valid_metrics_list:
      - Accuracy
    metric_to_watch: DummyMetric
    greater_metric_to_watch_is_better: true

...
```
</details>
