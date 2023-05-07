import pickle

import torch
from torch.utils.data import Dataset, DataLoader

__all__ = ["ValTrainRescoringDataset", "TrainRescoringDataset"]


class RescoringDataset(Dataset):
    def __init__(self, json_file: str):
        pass

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()

    @classmethod
    def parse_pkl_file(self, pkl_file_path: str):
        with open(pkl_file_path, "rb") as f:
            data = pickle.load(f)
        return data


class TrainRescoringDataset(RescoringDataset):
    """
    Implementation of the dataset for training the rescoring network.
    In this implementation, the dataset is a list of individual poses and DataLoader randomly samples
    them to form a batch during training.
    """

    def __init__(self, pkl_file: str):
        super().__init__(pkl_file)
        self.pred_poses = []
        self.pred_scores = []
        self.iou = []

        for sample in self.parse_pkl_file(pkl_file):
            pred_poses = sample["pred_poses"]
            pred_scores = sample["pred_scores"]
            iou = sample["iou"]

            self.pred_poses.extend(pred_poses)
            self.pred_scores.extend(pred_scores)
            self.iou.extend(iou)

        self.num_samples = len(self.pred_poses)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        inputs = torch.tensor(self.pred_poses[index])
        targets = torch.tensor([self.iou[index]])
        return inputs, targets


class ValTrainRescoringDataset(RescoringDataset):
    """
    Implementation of the dataset for validating the rescoring model.
    It differs from the training dataset implementation. Each sample represents a single image with all the poses
    on it, this enables us to compute pose estimation metrics after rescoring.

    This dataset is intended to be used with DataLoader with batch_size=1.
    In this case we don't need to padd poses in collate_fn.
    """

    def __init__(self, pkl_file: str):
        super().__init__(pkl_file)

        self.pred_poses = []
        self.pred_scores = []
        self.extras = []
        self.gt_joints = []
        self.gt_is_crowd = []
        self.gt_area = []
        self.iou = []

        for sample in self.parse_pkl_file(pkl_file):
            pred_poses = sample["pred_poses"]
            pred_scores = sample["pred_scores"]
            extras = dict(gt_joints=sample["gt_joints"], gt_iscrowd=sample["gt_iscrowd"], gt_bboxes=sample["gt_bboxes"], gt_areas=sample["gt_areas"])
            iou = sample["iou"]

            self.pred_poses.append(pred_poses)
            self.pred_scores.append(pred_scores)
            self.extras.append(extras)
            self.iou.append(iou)

        self.num_joints = next(p.shape[1] for p in self.pred_poses if len(p))
        self.num_samples = len(self.pred_poses)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        inputs = torch.tensor(self.pred_poses[index]).reshape(-1, self.num_joints, 3)
        targets = torch.tensor(self.iou[index]).reshape(-1, 1)
        extras = self.extras[index]
        return inputs, targets, extras


if __name__ == "__main__":
    ds = TrainRescoringDataset("D:/Develop/GitHub/Deci/super-gradients/src/super_gradients/rescoring_data.json")
    print(len(ds))
    inputs, targets = next(iter(DataLoader(ds, batch_size=32)))
    print(inputs.size(), targets.size())

    ds = ValTrainRescoringDataset("D:/Develop/GitHub/Deci/super-gradients/src/super_gradients/rescoring_data.json")
    print(len(ds))
    inputs, targets, extras = next(iter(DataLoader(ds, batch_size=1)))
    print(inputs.size(), targets.size())
