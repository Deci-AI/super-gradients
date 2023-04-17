import hydra
import json_tricks as json
import numpy as np
import pkg_resources
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from super_gradients import init_trainer, setup_device
from super_gradients.training import utils as core_utils, models, dataloaders
from super_gradients.training.metrics.pose_estimation_utils import compute_oks
from super_gradients.training.utils import get_param


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""), config_name="script_generate_rescoring_data", version_base="1.2")
def main(cfg: DictConfig) -> None:
    setup_device(
        device=core_utils.get_param(cfg, "device"),
        multi_gpu=core_utils.get_param(cfg, "multi_gpu"),
        num_gpus=core_utils.get_param(cfg, "num_gpus"),
    )

    sigmas = torch.from_numpy(np.array(cfg.dataset_params.oks_sigmas))

    cfg = instantiate(cfg)
    # BUILD NETWORK
    model = (
        models.get(
            model_name=cfg.architecture,
            num_classes=cfg.arch_params.num_classes,
            arch_params=cfg.arch_params,
            strict_load=cfg.checkpoint_params.strict_load,
            pretrained_weights=cfg.checkpoint_params.pretrained_weights,
            checkpoint_path=cfg.checkpoint_params.checkpoint_path,
        )
        .cuda()
        .eval()
    )

    # INSTANTIATE DATA LOADERS

    val_dataloader = dataloaders.get(
        name=get_param(cfg, "val_dataloader"),
        dataset_params=cfg.dataset_params.val_dataset_params,
        dataloader_params=cfg.dataset_params.val_dataloader_params,
    )

    post_prediction_callback = cfg.post_prediction_callback
    samples = []

    for inputs, targets, extras in tqdm(val_dataloader):
        with torch.no_grad(), torch.cuda.amp.autocast(True):
            predictions = model(inputs.cuda(non_blocking=True))
        all_poses, all_scores = post_prediction_callback(predictions)
        print(all_poses)

        batch_size = len(inputs)
        for image_index in range(batch_size):
            pred_poses = all_poses[image_index]  # [M, J, 3]
            pred_scores = all_scores[image_index]  # [M]

            gt_keypoints = extras["gt_joints"][image_index]  # [N, J, 3]
            gt_bboxes = extras["gt_bboxes"][image_index]  # [N, 4]

            gt_keypoints_xy = gt_keypoints[:, :, 0:2]
            gt_keypoints_visibility = gt_keypoints[:, :, 2]

            # Filter out poses with no visible keypoints
            if len(gt_keypoints_xy) == 0 or len(pred_poses) == 0:
                continue

            iou = compute_oks(
                pred_joints=torch.from_numpy(pred_poses),
                gt_joints=torch.from_numpy(gt_keypoints_xy),
                gt_keypoint_visibility=torch.from_numpy(gt_keypoints_visibility),
                sigmas=sigmas,
                gt_bboxes=torch.from_numpy(gt_bboxes),
            )

            # Here we are not interested in solving the MxN matching problem, but rather
            # in getting the largest IoU for each predicted pose with the ground truth poses.
            max_iou = iou.max(axis=1).values  # [M]

            sample = {
                "image_path": extras["image_path"][image_index],
                "pred_poses": pred_poses,
                "pred_scores": pred_scores,
                "gt_iou": max_iou,
            }
            samples.append(sample)

    with open("rescoring_data.json", "w") as f:
        json.dump(samples, f, indent=2)


def run():
    init_trainer()
    main()


if __name__ == "__main__":
    run()
