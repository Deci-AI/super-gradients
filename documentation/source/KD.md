# Knowledge Distillation (KD)

Pre-requisites: [Training in SG](Example_Classification.md), [Training with Configuration Files](configuration_files.md)

Knowledge distillation is a technique in deep learning that aims to transfer the knowledge of a large, pre-trained neural network model (the "teacher") to a smaller, more computationally efficient model (the "student"). This is accomplished by training the student to mimic the teacher's predictions and the ground-truth labels. The student network can also be designed to have a different architecture from the teacher, making it possible to distill the knowledge of a complex teacher network into a lighter and faster student network for deployment in real-world applications.


The training flow with Knowledge distillation in SG is similar to regular training. For standard training, we used SGs `Trainer` class - which was in charge of training the model, evaluating test data, making predictions, and saving checkpoints.
Equivalently, for knowledge distillation, we use the `KDTrainer` class which inherits from `Trainer`.

If for regular training with `Trainer`, the general flow is:

```python
...
trainer = Trainer("my_experiment")
train_dataloader = ...
valid_dataloader = ...
model = ...

train_params = {...}
trainer.train(model=model, training_params=train_params, train_loader=train_dataloader, valid_loader=valid_dataloader)
```


Then for training with knowledge distillation, the general flow is:
```python

from super_gradients.training.kd_trainer import KDTrainer
...
kd_trainer = KDTrainer("my_experiment")
train_dataloader = ...
valid_dataloader = ...

student_model = ...
teacher_model = ...

train_params = {...}
kd_trainer.train(student=student_model, teacher=teacher_model, training_params=train_params, train_loader=train_dataloader, valid_loader=valid_dataloader)
```

Check out our [knowledge distillation tutorial notebook](https://bit.ly/3BLA5oR) to see a practical example.


## Knowledge Distillation Training: Key Components


### [KDModule](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/training/models/kd_modules/kd_module.py)

The most apparent difference in the training flow using knowledge distillation is that it requires two networks: the "teacher" and the "student".
The relation between the two is also configurable - for example, we may decide that the teacher model should preprocess the inputs differently.
For that matter, SG introduces a new `torch.nn.Module` that wraps both the student and the teacher models: `KDModule`.
Upon calling `KDTrainer.train()`, the teacher and student models are passed along the `kd_arch_params` to initialize a `KDModule` instance.

Passing a `KDModule` instance explicitly to `KDTrainer.train()` through the `model` argument instead of student and teacher models is also possible, which gives the users the option to customize KD to their needs.

A high-level example of KD customization:

```python
import torch.nn
from super_gradients.training.kd_trainer import KDTrainer

...


class MyKDModule(KDModule):
    ...

    def forward(self, x: torch.Tensor)->KDOutput:
        intermediate_output_student = self.student.extract_intermediate_output(x, layer_ids=[1, 3, -1])
        intermediate_output_teacher = self.teacher.extract_intermediate(x, layer_ids=[1, 3, -1])
        return KDOutput(student_output=intermediate_output_student, teacher_output=intermediate_output_teacher)


class MyKDLoss(torch.nn.Module):
    ...
    def forward(self, preds: KDOutput, target: torch.Tensor):
        # does something with the intermediate outputs
        ...

kd_trainer = KDTrainer("my_customized_kd_experiment")
train_dataloader = ...
valid_dataloader = ...

student_model = ...
teacher_model = ...

kd_model = MyKDModule(student=student_model, teacher=teacher_model)

train_params = {'loss': MyKDLoss(),
                ...}

kd_trainer.train(model=kd_model, training_params=train_params,
                 train_loader=train_dataloader, valid_loader=valid_dataloader)
```
### [KDOutput](https://github.com/Deci-AI/super-gradients/blob/12a4e53a96e8608409100b5ef83971157518434b/src/super_gradients/training/models/kd_modules/kd_module.py#L7)

`KDOutput` defines the structure of the output of `KDModule` and has two self-explanatory attributes: student_output and teacher_output.
`KDTrainer` uses these attributes behind the scenes to perform the usual operations of regular training, such as metrics calculations.
This means that when customizing KD, it's essential for the custom `KDModule` to stick to this output format.

### KD Losses

Currently, [KDLogitsLoss](https://github.com/Deci-AI/super-gradients/blob/12a4e53a96e8608409100b5ef83971157518434b/src/super_gradients/training/losses/kd_losses.py#L15) is currently the only supported loss function in SGs KD losses bank, but more is to come.
Note that during KD training, the `KDModule` outputs (which are of `KDOutput` instance) are passed to the loss's forward method as predictions.

## Knowledge Distillation Training: Checkpoints

Checkpointing during KD training is generally the [same as checkpointing without KD](Checkpoints.md).
Nevertheless, there are a few differences worth mentioning:

- `ckpt_latest.pth` contains the state dict of the entire `KDModule`. 
- `ckpt_best.pth` contains the state dict of the student only.
- When training with EMA, `ckpt_best.pth`s `net` entry holds the EMA network.


## Knowledge Distillation Training with Configuration Files

As done when training without knowledge distillation, to [train with configuration files](configuration_files.md#required-hyper-parameters), we call the [`KDTrainer.train_from_config` method](https://github.com/Deci-AI/super-gradients/blob/9485f1533ff64cecb32a238d4779aafca1f0d199/src/super_gradients/training/kd_trainer/kd_trainer.py#L43), which assumes a specific [configuration structure](configuration_files.md#required-hyper-parameters).
When training with KD, the same structure and required fields hold, but we introduce a few additions:

- `arch_params` are being passed to the `KDModule` constructor. For example, in our [Resnet50 KD training on Imagenet](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/recipes/imagenet_resnet50_kd.yaml), we handle the difference in preprocessing of the teacher, which expects different normalization by passing the `KDModule` a normalization adaptor module:
```yaml
# super_gradients/recipes/imagenet_resnet50_kd.yaml
...
arch_params:
  teacher_input_adapter:
    _target_: super_gradients.training.utils.kd_trainer_utils.NormalizationAdapter
    mean_original: [0.485, 0.456, 0.406]
    std_original: [0.229, 0.224, 0.225]
    mean_required: [0.5, 0.5, 0.5]
    std_required: [0.5, 0.5, 0.5]
```
> Warning: Remember to distinguish the arch params being passed to the KDModule constructor from the student ones.

- `student_architecture`, `teacher_architecture`,` student_arch_params`, `student_checkpoint_params`, `teacher_arch_params`, and ` teacher_checkpoint_params` play the same role as `architecture`, `arch_params` and `checkpoint_params` for instantiating our model in non-KD training, and are being passed to `models.get(...)` to instantiate the teacher and the student:

```yaml
...
student_architecture: resnet50
teacher_architecture: beit_base_patch16_224

student_arch_params:
  num_classes: 1000

teacher_arch_params:
  num_classes: 1000
  image_size: [224, 224]
  patch_size: [16, 16]

teacher_checkpoint_params:
...
  pretrained_weights: imagenet

student_checkpoint_params:
...

```

Any KD recipe can be launched with our [train_from_kd_recipe_example](https://github.com/Deci-AI/super-gradients/blob/master/src/super_gradients/examples/train_from_kd_recipe_example/train_from_kd_recipe.py) script.
