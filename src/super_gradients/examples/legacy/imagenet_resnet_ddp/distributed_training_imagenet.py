#!/usr/bin/env python
""" Single node distributed training.

    The program will dispatch distributed training on all available GPUs residing in a single node.

    Usage:
    python -m torch.distributed.launch --nproc_per_node=n distributed_training_imagenet.py
    where n is the number of GPUs required, e.g., n=8

    Important note: (1) in distributed training it is customary to specify learning rates and batch sizes per GPU.
    Whatever learning rate and schedule you specify will be applied to the each GPU individually.
    Since gradients are passed and summed (reduced) from all to all GPUs, the effective batch size is the
    batch you specify times the number of GPUs. In the literature there are several "best practices" to set
    learning rates and schedules for large batch sizes.
    Should be checked with. (2) The training protocol specified in this file for 8 GPUs are far from optimal.
    The best protocol should use cosine schedule.

    In the example below: for ImageNetDataset training using Resnet50, when applied with n=8 should compute an Eopch in about
    5min20sec with 8 V100 GPUs.

    Todo: (1) the code is more or less ready for multiple nodes, but I have not experimented with it at all.
          (2) detection and segmentation codes were not modified and should not work properly.
              Specifically, the analogue changes done in sg_classification_model should be done also in
              deci_segmentation_model and deci_detection_model

"""
import super_gradients
import torch.distributed
from super_gradients.training.sg_trainer import MultiGPUMode
from super_gradients.training import Trainer
from super_gradients.training.datasets.dataset_interfaces import ImageNetDatasetInterface
from super_gradients.common.aws_connection.aws_secrets_manager_connector import AWSSecretsManagerConnector
from super_gradients.training.metrics.classification_metrics import Accuracy, Top5

torch.backends.cudnn.benchmark = True

super_gradients.init_trainer()
# TODO - VALIDATE THE HYPER PARAMETERS WITH RAN TO FIX THIS EXAMPLE CODE
train_params = {"max_epochs": 110,
                "lr_updates": [30, 60, 90],
                "lr_decay_factor": 0.1,
                "initial_lr": 0.6,
                "loss": "cross_entropy",
                "lr_mode": "step",
                # "initial_lr": 0.05 * 2,
                "lr_warmup_epochs": 5,
                # "criterion_params":{"smooth_eps":0.1}}
                "mixed_precision": True,
                # "mixed_precision_opt_level": "O3",
                "optimizer_params": {"weight_decay": 0.000, "momentum": 0.9},
                # "optimizer_params": {"weight_decay": 0.0001, "momentum": 0.9}
                "train_metrics_list": [Accuracy(), Top5()], "valid_metrics_list": [Accuracy(), Top5()],
                "loss_logging_items_names": ["Loss"], "metric_to_watch": "Accuracy",
                "greater_metric_to_watch_is_better": True}
dataset_params = {"batch_size": 128}

model_repo_bucket_name = AWSSecretsManagerConnector.get_secret_value_for_secret_key(aws_env='research',
                                                                                    secret_name='training_secrets',
                                                                                    secret_key='S3.MODEL_REPOSITORY_BUCKET_NAME')
trainer = Trainer("test_checkpoints_resnet_8_gpus",
                  model_checkpoints_location='s3://' + model_repo_bucket_name,
                  multi_gpu=MultiGPUMode.DISTRIBUTED_DATA_PARALLEL
                  )
# FOR AWS
dataset = ImageNetDatasetInterface(data_dir="/data/Imagenet", dataset_params=dataset_params)
trainer.connect_dataset_interface(dataset, data_loader_num_workers=8)
trainer.build_model("resnet50")
trainer.train(training_params=train_params)
