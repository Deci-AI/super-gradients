import numpy as np
import torch.utils.data

from statistics import DatasetStatistics
from super_gradients.training.utils.detection_utils import Anchors
from super_gradients import SgModel
from super_gradients.training.datasets import CoCoDetectionDatasetInterface


def build_w_interface():

    model = SgModel(experiment_name='tomer')
    print("Created SG Model")

####################################################################################

    yolo_v5_dataset_params = {"batch_size": 32,
                              "test_batch_size": 32,
                              "dataset_dir": "/data/coco/",
                              # "image_size": 128,
                              "train_image_size": 128,
                              "val_image_size": 128}

    coco_dataset_interface = CoCoDetectionDatasetInterface(dataset_params=yolo_v5_dataset_params, cache_labels=False)

    model.train_loader, model.valid_loader, model.test_loader, model.classes = coco_dataset_interface.get_data_loaders()
    # model.connect_dataset_interface(coco_dataset_interface, data_loader_num_workers=4)

    print("Connected dataset interface")

####################################################################################

    model.build_model(architecture='yolo_v5n')
    print("Built model")

####################################################################################

    coco2017_quickstart_anchors = Anchors(anchors_list=[[5, 6, 8, 15, 21, 13],
                                                        [15, 36, 32, 32, 36, 80],
                                                        [71, 55, 89, 137, 213, 167]],
                                          strides=[8, 16, 32])

    training_yolov5_params = {"max_epochs": 1, "initial_lr": 0.01,
                              "loss": "yolo_v5_loss", "optimizer": "SGD",
                              "criterion_params": {
                              "anchors": coco2017_quickstart_anchors,
                              "box_loss_gain": 0.05,
                              "cls_loss_gain": 0.5,
                              "obj_loss_gain": 0.25,
                              },
                              "dataset_statistics": True,
                              "train_metrics_list": [],
                              "loss_logging_items_names": ["GIoU", "obj", "cls", "Loss"],
                              "metric_to_watch": "GIoU",
                              "average_best_models": False
                              }
    print("Starting train")
    model.train(training_params=training_yolov5_params)
    # model.evaluate()


class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 32

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = np.zeros(shape=(64, 64, 3))
        annotations = np.array((idx % 3, 0.5, 0.5, 0.75, 0.75))
        # sample = {'image': image, 'landmarks': annotations}
        return image, annotations


def build_w_my_statistics():
    my_classes = ['Horse', 'Natan', 'IceCream']
    my_ds = CustomDataSet()
    my_dl = torch.utils.data.DataLoader(my_ds, batch_size=8)
    my_anchors = Anchors(anchors_list=[[5, 6, 8, 15, 21, 13],
                                                        [15, 36, 32, 32, 36, 80],
                                                        [71, 55, 89, 137, 213, 167]],
                                          strides=[8, 16, 32])

    my_anchors = None
    my_stat = DatasetStatistics(data_loader=my_dl, classes_names=my_classes, anchors=my_anchors)
    my_stat.analyze()


def main():
    build_w_interface()
    # build_w_my_statistics()


if __name__ == '__main__':
    main()
