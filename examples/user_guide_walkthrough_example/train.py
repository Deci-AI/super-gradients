from super_gradients.training import SgModel
from super_gradients.training import MultiGPUMode
from dataset import UserDataset
from model import ResNet, BasicBlock
from loss import LabelSmoothingCrossEntropyLoss
from metrics import Accuracy, Top5


def main():
    # ------------------ Loading The Model From Model.py----------------
    arch_params = {'num_classes': 10}
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=arch_params['num_classes'])

    deci_classification_model = SgModel('client_model_training',
                                        model_checkpoints_location='local',
                                        multi_gpu=MultiGPUMode.OFF)

    # if a torch.nn.Module is provided when building the model, the model will be integrated into deci model class
    deci_classification_model.build_model(model, arch_params=arch_params, load_checkpoint=False)

    # ------------------ Loading The Dataset From Dataset.py----------------
    dataset_params = {"batch_size": 256}
    dataset = UserDataset(dataset_params)
    deci_classification_model.connect_dataset_interface(dataset)

    # ------------------ Loading The Loss From Loss.py -----------------
    loss = LabelSmoothingCrossEntropyLoss()

    # ------------------ Defining the metrics we wish to log -----------------
    train_metrics_list = [Accuracy(), Top5()]
    valid_metrics_list = [Accuracy(), Top5()]

    # ------------------ Training -----------------
    train_params = {"max_epochs": 250,
                    "lr_updates": [100, 150, 200],
                    "lr_decay_factor": 0.1,
                    "lr_mode": "step",
                    "lr_warmup_epochs": 0,
                    "initial_lr": 0.1,
                    "loss": loss,
                    "criterion_params": {},
                    "optimizer": "SGD",
                    "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
                    "launch_tensorboard": False,
                    "train_metrics_list": train_metrics_list,
                    "valid_metrics_list": valid_metrics_list,
                    "loss_logging_items_names": ["Loss"],
                    "metric_to_watch": "Accuracy",
                    "greater_metric_to_watch_is_better": True}

    deci_classification_model.train(train_params)


if __name__ == '__main__':
    main()
