# """
# This short quickstart tutorial uses SuperGradients to demonstrate knowledge distillation training, and includes:
#
# - Loading Cifar10 dataset.
#
# - Building our teacher network: BEiT base.
#
# - Building our student network: Resnet18 (Cifar10 varaint).
#
# - Train it using knowledge distillation, on the teacher's raw outputs.
#
#
# Imports:
# """
# from super_gradients.training import KDTrainer, dataloaders
# from super_gradients.training import models
#
# from torchvision.transforms import Resize
#
#
# """#Introdocing new KD supportive objects:
#
# In previous tutorials, we:
#
# Initialized an sg_model object, which is in charge of everything that happens during training- saving checkpoints, plotting etc.
# The experiment name argument will cause the checkpoints, logs and tensorboards to be saved in a directory with the same name under the checkpoints
# directory.
#
# ```
# trainer = SgModel(experiment_name='cifar10_on_resnet18',ckpt_root_dir="/home/louis.dupont/data/notebook_ckpts/")
# ```
#
# Equivalently, when training with KD we use KDModel:
#
# """
#
# trainer = KDTrainer(experiment_name="kd_cifar10_resnet", ckpt_root_dir="/home/louis.dupont/data/notebook_ckpts/")
#
# """As before, we initialize a dataset interface object that wraps the train and validation data loaders we will use during training:"""
#
# train_dataloader = dataloaders.get("cifar10_train", dataset_params={"download": False}, dataloader_params={"batch_size": 128})
# val_dataloader = dataloaders.get("cifar10_val", dataset_params={"download": False}, dataloader_params={"batch_size": 512})
#
# """**Building a KDModule**
#
# In our previous [classification quickstart tutorial](https://bit.ly/3ufnsgT),
# our call to build_model was quite simple:
#
#
# ```
# trainer.build_model("resnet18", arch_params={'num_classes': 10})
# ```
#
# But now we need to initialize both student and teacher networks in order to perform knowledge distillation.
#
# The same logic for building models in SgModel applies for both the student and the teacher. We pass the student architecture, teacher architecture, student architecture parameters, and teacher architecture parameters as done below.
#
# Note that here, we use the cifar10 pretrained checkpoint for our teacher from our model zoo, but we could have used a local trained checkpoint by passing it's path through the keyword "teacher_checkpoint_path" in checkpoint_params.
#
# Additionally, notice the key in arch_params teacher_input_adapter, which we have passed `Resize(224)` to. This is because our teacher model expects input of size 224X224 while we wish to train our student on 32X32 images.
# """
#
# student = models.get('resnet18_cifar', arch_params={'num_classes': 10})
# teacher = models.get('beit_base_patch16_224', pretrained_weights="cifar10")
#
# # from super_gradients.training.metrics import Accuracy, Top5
# # import torchvision.transforms as transform
# #
# # teacher_test_dataloader = dataloaders.get("cifar10_val", dataloader_params={"batch_size": 64}, dataset_params={"transforms": [transform.ToTensor(), transform.Resize(224)]})
# # accuracy, top5 = trainer.test(model=teacher,
# #                               test_loader=teacher_test_dataloader,
# #                               test_metrics_list=[Accuracy(), Top5()])
# # print(f"Accuracy: {accuracy}")
# # print(f"Top 5:    {top5}")
#
#
# from super_gradients.training import training_hyperparams
# training_params = training_hyperparams.get("imagenet_resnet50_kd")
# # training_params = training_hyperparams.get("cifar10_resnet")
# #
# #
# # from super_gradients.training.losses import KDLogitsLoss, LabelSmoothingCrossEntropyLoss
# # training_params["loss"] = KDLogitsLoss(distillation_loss_coeff=0.8, task_loss_fn=LabelSmoothingCrossEntropyLoss())
#
# training_params["max_epochs"] = 20
# training_params["loss_logging_items_names"] = ["Loss", "Task Loss", "Distillation Loss"]
#
# trainer.train(training_params=training_params,
#               student=student,
#               teacher=teacher,
#               kd_architecture="kd_module",
#               kd_arch_params={"teacher_input_adapter": Resize(224)},
#               run_teacher_on_eval=False,
#               train_loader=train_dataloader, valid_loader=val_dataloader)
#
#
# """**Defining Training Hyperparameters**
#
# Most of the below arguement's use has already been demonstarted in previous tutorials. Do notice the `KDLogitsLoss` we use as our loss function- which mathematically is the weighted sum of:
#
#
# (1- distillation_loss_coeff) * LabelSmoothingCrossEntropyLoss (applied on the "regular" dataset labels)  + KLDivergenceLoss (applied on the teacher's raw outputs).
# """
#
#
# """Laucnhing the tensoboard process:"""
#
# # Commented out IPython magic to ensure Python compatibility.
# # %load_ext tensorboard
# # %tensorboard --logdir /home/louis.dupont/data/notebook_ckpts/ --bind_all
#
# """Finally, call train:"""
#
# # trainer.train(kd_train_params)
#
# print("Best Checkpoint Accuracy is: "+ str(trainer.best_metric.item()))
#
# print(trainer.checkpoints_dir_path)
#
# """Once training is finished, the trained model can be accessed through *trainer.net.module.student*. All of the [architectures](https://deci-ai.github.io/super-gradients/user_guide.html#network-architectures) offered by SuperGradients compilation processes have been verified and can be uploaded to [Deci Lab](https://docs.deci.ai/docs/quickstart) for runtime optimization (including quantization and graph compilers) and benchmarking on various hardwares.  <!--Be sure to see how one can do so in our deci_platform integration tutorial <link>. -->"""




from super_gradients.training import Trainer, models, dataloaders
from super_gradients.training.metrics import Accuracy, Top5
from torchvision import transforms


trainer = Trainer(experiment_name="beit_base_patch16_224_test")
test_dataloader = dataloaders.get("cifar10_val", dataloader_params={"batch_size": 64},dataset_params={"transforms": [transforms.ToTensor(), transforms.Resize(224)]})
pretrained_beit = models.get('beit_base_patch16_224', arch_params={'num_classes': 10, "image_size": [224, 224], "patch_size": [16, 16]}, pretrained_weights="cifar10")
# accuracy, top5 = trainer.test(model=pretrained_beit, test_loader=test_dataloader, test_metrics_list=[Accuracy(), Top5()])
# print()
# print(f"Accuracy: {accuracy}")
# print(f"Top 5:    {top5}")

from super_gradients.training import KDTrainer


kd_trainer = KDTrainer(experiment_name="kd_cifar10_resnet")



from super_gradients.training import dataloaders, models


train_dataloader = dataloaders.get("cifar10_train", dataloader_params={"batch_size": 128})
val_dataloader = dataloaders.get("cifar10_val", dataloader_params={"batch_size": 512})

student_resnet18 = models.get('resnet18_cifar', num_classes=10)




from super_gradients.training import training_hyperparams
from super_gradients.training.losses import KDLogitsLoss, LabelSmoothingCrossEntropyLoss


kd_params = {
    "max_epochs": 5,  # We will stop after 4 epochs because it is slow to train on google collab
    "loss": KDLogitsLoss(distillation_loss_coeff=0.8, task_loss_fn=LabelSmoothingCrossEntropyLoss()),
    "loss_logging_items_names": ["Loss", "Task Loss", "Distillation Loss"]}

training_params = training_hyperparams.get("imagenet_resnet50_kd", overriding_params=kd_params)


arch_params={"teacher_input_adapter": transforms.Resize(224)}


kd_trainer.train(training_params=training_params,
                 student=student_resnet18,
                 teacher=pretrained_beit,
                 kd_architecture="kd_module",
                 kd_arch_params=arch_params,
                 train_loader=train_dataloader, valid_loader=val_dataloader)


from super_gradients.training.metrics import Accuracy, Top5

accuracy, top5 = trainer.test(model=student_resnet18, test_loader=val_dataloader, test_metrics_list=[Accuracy(), Top5()])
print()
print(f"Accuracy: {accuracy}")
print(f"Top 5:    {top5}")