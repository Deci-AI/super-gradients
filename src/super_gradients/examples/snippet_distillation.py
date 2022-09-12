#
# from super_gradients.training import KDTrainer, dataloaders, models, training_hyperparams
# from torchvision.transforms import Resize
#
# # Instantiate the trainer
# trainer = KDTrainer(experiment_name="imagenet_resnet50_kd", ckpt_root_dir="/home/louis.dupont/data/notebook_ckpts/")
#
# # Get the data loaders
# train_dataloader = dataloaders.get("imagenet_train")
# val_dataloader = dataloaders.get("imagenet_val")
#
# # Get the models
# student = models.get('resnet50', num_classes=1000)
# teacher = models.get('beit_base_patch16_224', num_classes=1000, pretrained_weights="cifar10", arch_params={"image_size": [224, 224], "patch_size": [16, 16]})
#
# # Get predefined training hyperparameters
# training_params = training_hyperparams.get("imagenet_resnet50_kd")
#
# # Define an adapter to allow the teacher to run on this image size (224x224)
# arch_params = {"teacher_input_adapter": Resize(224)}
#
# # Train
# trainer.train(training_params=training_params,
#               student=student,
#               teacher=teacher,
#               kd_architecture="kd_module",
#               kd_arch_params=arch_params,
#               run_teacher_on_eval=True,
#               train_loader=train_dataloader, valid_loader=val_dataloader)


from super_gradients.training import KDTrainer, dataloaders, models, training_hyperparams

# Instantiate the trainer
trainer = KDTrainer(experiment_name="imagenet_resnet50_kd")

# Get the data loaders
train_dataloader = dataloaders.get("imagenet_train")
val_dataloader = dataloaders.get("imagenet_val")

# Get the models
student = models.get('resnet50', num_classes=1000)
teacher = models.get('beit_base_patch16_224', num_classes=1000, pretrained_weights="imagenet")

# Get predefined training hyperparameters
training_params = training_hyperparams.get("imagenet_resnet50_kd")

# Train
trainer.train(training_params=training_params,
              student=student, teacher=teacher,
              train_loader=train_dataloader, valid_loader=val_dataloader)