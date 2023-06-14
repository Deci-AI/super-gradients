import collections
import os.path
import pickle
from pprint import pprint
from typing import Optional

import hydra
import numpy as np
import pkg_resources
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from super_gradients import init_trainer, setup_device
from super_gradients.training import utils as core_utils, models, dataloaders
from super_gradients.training.metrics import PoseEstimationMetrics
from super_gradients.training.metrics.pose_estimation_utils import compute_oks
from super_gradients.training.models.pose_estimation_models.dekr_hrnet import DEKRHorisontalFlipWrapper
from super_gradients.training.utils import get_param


def remove_starting_module(key: str):
    if key.startswith("module."):
        return key[7:]
    return key


def process_loader(model, loader, post_prediction_callback, sigmas, metric: Optional[PoseEstimationMetrics] = None):
    samples = []
    for inputs, targets, extras in tqdm(loader):
        with torch.no_grad(), torch.cuda.amp.autocast(True):
            predictions = model(inputs.cuda(non_blocking=True))
            all_poses, all_scores = post_prediction_callback(predictions)

        if metric is not None:
            metric.update(predictions, targets, **extras)

        batch_size = len(inputs)
        for image_index in range(batch_size):
            pred_poses = all_poses[image_index]  # [M, J, 3]
            pred_scores = all_scores[image_index]  # [M]

            gt_iscrowd = extras["gt_iscrowd"][image_index]
            gt_keypoints = extras["gt_joints"][image_index]  # [N, J, 3]
            gt_bboxes = extras["gt_bboxes"][image_index]  # [N, 4]
            gt_areas = extras["gt_areas"][image_index]  # [N]

            # Filter out poses with no visible keypoints
            if len(gt_keypoints) > 0 and len(pred_poses) > 0:
                gt_keypoints_xy = gt_keypoints[:, :, 0:2]
                gt_keypoints_visibility = gt_keypoints[:, :, 2]

                iou = compute_oks(
                    pred_joints=torch.from_numpy(pred_poses),
                    gt_joints=torch.from_numpy(gt_keypoints_xy),
                    gt_keypoint_visibility=torch.from_numpy(gt_keypoints_visibility),
                    sigmas=sigmas,
                    gt_bboxes=torch.from_numpy(gt_bboxes),
                )

                # Here we are not interested in solving the MxN matching problem, but rather
                # in getting the largest IoU for each predicted pose with the ground truth poses.
                max_iou = iou.max(axis=1).values.numpy()  # [M]
            else:
                max_iou = np.zeros(len(pred_poses), dtype=np.float32)

            sample = {
                "pred_poses": pred_poses,
                "pred_scores": pred_scores,
                # Targets
                "iou": max_iou,
                #
                "gt_bboxes": gt_bboxes,
                "gt_joints": gt_keypoints,
                "gt_iscrowd": gt_iscrowd,
                "gt_areas": gt_areas,
            }
            samples.append(sample)
    return samples


@hydra.main(
    config_path=pkg_resources.resource_filename("super_gradients.recipes", ""), config_name="script_generate_rescoring_data_dekr_coco2017", version_base="1.2"
)
def main(cfg: DictConfig) -> None:
    setup_device(
        device=core_utils.get_param(cfg, "device"),
        multi_gpu=core_utils.get_param(cfg, "multi_gpu"),
        num_gpus=core_utils.get_param(cfg, "num_gpus"),
    )

    sigmas = torch.from_numpy(np.array(cfg.dataset_params.oks_sigmas))

    cfg.dataset_params.train_dataset_params.transforms = cfg.dataset_params.val_dataset_params.transforms
    cfg = instantiate(cfg)

    # Temporary hack to remove "module." from model state dict saved in checkpoint
    if cfg.checkpoint_params.checkpoint_path is not None:
        checkpoint = torch.load(cfg.checkpoint_params.checkpoint_path, map_location="cpu")
        if "ema_net" in checkpoint:
            checkpoint["ema_net"] = collections.OrderedDict((remove_starting_module(k), v) for k, v in checkpoint["ema_net"].items())
        if "net" in checkpoint:
            checkpoint["net"] = collections.OrderedDict((remove_starting_module(k), v) for k, v in checkpoint["net"].items())
        torch.save(checkpoint, cfg.checkpoint_params.checkpoint_path)

    # BUILD NETWORK
    model = models.get(
        model_name=cfg.architecture,
        num_classes=cfg.arch_params.num_classes,
        arch_params=cfg.arch_params,
        strict_load=cfg.checkpoint_params.strict_load,
        pretrained_weights=cfg.checkpoint_params.pretrained_weights,
        checkpoint_path=cfg.checkpoint_params.checkpoint_path,
    )

    # model = DEKRWrapper(model, apply_sigmoid=True).cuda().eval()
    model = DEKRHorisontalFlipWrapper(model, cfg.dataset_params.flip_indexes, apply_sigmoid=True).cuda().eval()

    post_prediction_callback = cfg.post_prediction_callback

    pose_estimation_metric = PoseEstimationMetrics(
        post_prediction_callback=post_prediction_callback,
        max_objects_per_image=post_prediction_callback.max_num_people,
        num_joints=cfg.dataset_params.num_joints,
        oks_sigmas=cfg.dataset_params.oks_sigmas,
    )

    os.makedirs(cfg.rescoring_data_dir, exist_ok=True)

    val_dataloader = dataloaders.get(
        name=get_param(cfg, "val_dataloader"),
        dataset_params=cfg.dataset_params.val_dataset_params,
        dataloader_params=cfg.dataset_params.val_dataloader_params,
    )
    valid_samples = process_loader(model, val_dataloader, post_prediction_callback, sigmas, metric=pose_estimation_metric)

    with open(os.path.join(cfg.rescoring_data_dir, "rescoring_data_valid.pkl"), "wb") as f:
        pickle.dump(valid_samples, f)

    print("Pose estimation metrics on validation set:")
    pprint(pose_estimation_metric.compute())

    train_dataloader = dataloaders.get(
        name=get_param(cfg, "train_dataloader"),
        dataset_params=cfg.dataset_params.train_dataset_params,
        dataloader_params=cfg.dataset_params.train_dataloader_params,
    )
    train_samples = process_loader(model, train_dataloader, post_prediction_callback, sigmas)
    with open(os.path.join(cfg.rescoring_data_dir, "rescoring_data_train.pkl"), "wb") as f:
        pickle.dump(train_samples, f)

    print(f"Train data for rescoring saved to {cfg.rescoring_data_dir}")


def run():
    init_trainer()
    main()


if __name__ == "__main__":
    run()
