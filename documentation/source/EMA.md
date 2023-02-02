# EMA

Exponential Moving Average or EMA is a technique used during training to smooth the noise in the training process and improve the generalization of the model.
It is a simple technique that can be used with any model and optimizer. Here's a recap how EMA works: 

- At the start of training, the model parameters are copied to the EMA parameters.
- At each gradient update step, the EMA parameters are updated using the following formula:
    ```py
    ema_param = ema_param * decay + param * (1 - decay)
    ```
- On start of validation epoch the model parameters are replaced by the EMA parameters and reverted back on the end of validation epoch.  
- At the end of training, the model parameters are replaced by the EMA parameters.


To enable use of EMA is SuperGradients one should add following parameters to the `training_params`:

```py
from super_gradients import Trainer
trainer = Trainer(...)
trainer.train(
    training_params={"ema": True, "ema_params": {"decay": 0.9999, "decay_type": "constant"}, ...}, 
    ...
)
```

The `decay` is a hyperparameter that controls the speed of the EMA update. It's value must be in `(0,1)` range.
Larger values of `decay` will result in slower EMA model update.

It is usually beneficial to have smaller decay values at the start of training and increase it as the training progresses. 
In SuperGradients we support several types of changing decay value over time:

- `constant`:  `"ema_params": {"decay": 0.9999, "decay_type": "constant"}`
- `threshold`: `"ema_params": {"decay": 0.9999, "decay_type": "threshold"}`
- `exp`:       `"ema_params": {"decay": 0.9999, "decay_type": "exp", "beta": 15}`

![EMA Decay schedules](images/ema_decay_schedules.png)

## Knowledge Distillation

EMA is also supported in knowledge distillation. To enable it one should add following parameters to the `training_params` similar to the above example:

```py
from super_gradients import KDTrainer
trainer = KDTrainer(...)
trainer.train(
    training_params={"ema": True, "ema_params": {"decay": 0.9999, "decay_type": "constant"}, ...}, 
    ...
)
```
