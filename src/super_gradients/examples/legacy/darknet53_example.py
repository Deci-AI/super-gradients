# Darknet53 Backbone Training on HAM10000 Dataset
from super_gradients.training import MultiGPUMode
from super_gradients.training import SgModel
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import ClassificationDatasetInterface

# Define Parameters
train_params = {"max_epochs": 110, "lr_updates": [30, 60, 90, 100], "lr_decay_factor": 0.1, "lr_mode": "step",
                "lr_warmup_epochs": 0, "initial_lr": 0.1, "loss": "cross_entropy", "optimizer": "SGD",
                "criterion_params": {}, "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9}}
arch_params = {'backbone_mode': False, 'num_classes': 7}
dataset_params = {"batch_size": 16, "test_batch_size": 16, 'dataset_dir': '/data/HAM10000'}

# Define Model
model = SgModel("Darknet53_Backbone_HAM10000",
                model_checkpoints_location='local',
                device='cuda',
                multi_gpu=MultiGPUMode.DATA_PARALLEL)

# Connect Dataset
dataset = ClassificationDatasetInterface(normalization_mean=(0.7483, 0.5154, 0.5353),
                                         normalization_std=(0.1455, 0.1691, 0.1879),
                                         resolution=416,
                                         dataset_params=dataset_params)

model.connect_dataset_interface(dataset, data_loader_num_workers=8)

# Build Model
model.build_model("darknet53", arch_params=arch_params)

# Start Training
model.train(training_params=train_params)
