from typing import Union, Tuple, List

import numpy as np
import torch
from torch import nn, Tensor


def get_locations(output_h: int, output_w: int, device):
    """
    Generate location map (each pixel contains its own XY coordinate)

    :param output_h: Feature map height (rows)
    :param output_w: Feature map width (cols)
    :param device: Target device to put tensor on
    :return: [H * W, 2]
    """
    shifts_x = torch.arange(0, output_w, step=1, dtype=torch.float32, device=device)
    shifts_y = torch.arange(0, output_h, step=1, dtype=torch.float32, device=device)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1)

    return locations


def get_reg_poses(offset: Tensor, num_joints: int):
    """
    Decode offset predictions into absolute locations.

    :param offset: Tensor of [num_joints*2,H,W] shape with offset predictions for each joint
    :param num_joints: Number of joints
    :return: [H * W, num_joints, 2]
    """
    _, h, w = offset.shape
    offset = offset.permute(1, 2, 0).reshape(h * w, num_joints, 2)
    locations = get_locations(h, w, offset.device)
    locations = locations[:, None, :].expand(-1, num_joints, -1)
    poses = locations - offset

    return poses


def _offset_to_pose(offset, flip=False, flip_index=None):
    """
    Decode offset predictions into absolute locations.

    :param offset: [1, 2 * num_joints, H, W]
    :param flip: True to decode offsets for flipped keypoints (WHen horisontal flip TTA is used)
    :param flip_index: Indexes of keypoints that must be swapped to account for horizontal flip
    :return: [1, 2 * num_joints, H, W]
    """
    num_offset, h, w = offset.shape[1:]
    num_joints = int(num_offset / 2)
    reg_poses = get_reg_poses(offset[0], num_joints)

    if flip:
        reg_poses = reg_poses[:, flip_index, :]
        reg_poses[:, :, 0] = w - reg_poses[:, :, 0] - 1

    reg_poses = reg_poses.contiguous().view(h * w, 2 * num_joints).permute(1, 0)
    reg_poses = reg_poses.contiguous().view(1, -1, h, w).contiguous()

    return reg_poses


def _hierarchical_pool(heatmap, pool_threshold1=300, pool_threshold2=200):
    pool1 = torch.nn.MaxPool2d(3, 1, 1)
    pool2 = torch.nn.MaxPool2d(5, 1, 2)
    pool3 = torch.nn.MaxPool2d(7, 1, 3)
    map_size = (heatmap.shape[1] + heatmap.shape[2]) / 2.0
    if map_size > pool_threshold1:
        maxm = pool3(heatmap[None, :, :, :])
    elif map_size > pool_threshold2:
        maxm = pool2(heatmap[None, :, :, :])
    else:
        maxm = pool1(heatmap[None, :, :, :])

    return maxm


def _get_maximum_from_heatmap(heatmap, max_num_people: int, pose_center_score_threshold: float) -> Tuple[Tensor, Tensor]:
    """

    :param heatmap: [1, H, W] Single-channel heatmap
    :param max_num_people: (int) Maximum number of poses to return
    :param pose_center_score_threshold: (float) A minimum score of a pose center keypoint for pose to be considered as a potential candidate
    :return: Tuple of (indexes of poses, scores)
    """
    maxm = _hierarchical_pool(heatmap)
    maxm = torch.eq(maxm, heatmap).float()
    heatmap = heatmap * maxm
    scores = heatmap.view(-1)
    scores, pos_ind = scores.topk(max_num_people)

    select_ind = (scores > pose_center_score_threshold).nonzero()
    scores = scores[select_ind][:, 0]
    pos_ind = pos_ind[select_ind][:, 0]

    return pos_ind, scores


def _up_interpolate(x, size):
    H = x.size(2)
    W = x.size(3)
    scale_h = int(size[0] / H)
    scale_w = int(size[1] / W)
    inter_x = torch.nn.functional.interpolate(x, size=[size[0] - scale_h + 1, size[1] - scale_w + 1], align_corners=True, mode="bilinear")
    padd = torch.nn.ReplicationPad2d((0, scale_w - 1, 0, scale_h - 1))
    return padd(inter_x)


def _cal_area_2_torch(v):
    w = torch.max(v[:, :, 0], -1)[0] - torch.min(v[:, :, 0], -1)[0]
    h = torch.max(v[:, :, 1], -1)[0] - torch.min(v[:, :, 1], -1)[0]
    return w * w + h * h


def _nms_core(pose_coord, heat_score, nms_threshold: float, nms_num_threshold: int):
    """
    Non-maximum suppression for predicted poses.
    Removes poses that has certain number of joints that are too close to each other.

    :param pose_coord: Array of shape [num_people, num_joints, 2] with pose coordinates
    :param heat_score: Scores of each joint
    :param float nms_threshold: The maximum distance between two joints for them to be considered as belonging to the same pose.
                          Given in terms of a percentage of a square root of the area of the pose bounding box.
    :param int nms_num_threshold: Number of joints that must pass the NMS check for the pose to be considered as a valid one.
    :return: Indexes of poses that should be kept
    """
    num_people, num_joints, _ = pose_coord.shape
    pose_area = _cal_area_2_torch(pose_coord)[:, None].repeat(1, num_people * num_joints)
    pose_area = pose_area.reshape(num_people, num_people, num_joints)

    pose_diff = pose_coord[:, None, :, :] - pose_coord
    pose_diff.pow_(2)
    pose_dist = pose_diff.sum(3)
    pose_dist.sqrt_()
    pose_thre = nms_threshold * torch.sqrt(pose_area)
    pose_dist = (pose_dist < pose_thre).sum(2)
    nms_pose = pose_dist > nms_num_threshold

    ignored_pose_inds = []
    keep_pose_inds = []
    for i in range(nms_pose.shape[0]):
        if i in ignored_pose_inds:
            continue
        keep_inds = nms_pose[i].nonzero().cpu().numpy()
        keep_inds = [list(kind)[0] for kind in keep_inds]
        if len(keep_inds) == 0:
            continue
        keep_scores = heat_score[keep_inds]
        ind = torch.argmax(keep_scores)
        keep_ind = keep_inds[ind]
        if keep_ind in ignored_pose_inds:
            continue
        keep_pose_inds += [keep_ind]
        ignored_pose_inds += list(set(keep_inds) - set(ignored_pose_inds))

    return keep_pose_inds


def _get_heat_value(pose_coord, heatmap):
    _, h, w = heatmap.shape
    heatmap_nocenter = heatmap[:-1].flatten(1, 2).transpose(0, 1)

    y_b = torch.clamp(torch.floor(pose_coord[:, :, 1]), 0, h - 1).long()
    x_l = torch.clamp(torch.floor(pose_coord[:, :, 0]), 0, w - 1).long()
    heatval = torch.gather(heatmap_nocenter, 0, y_b * w + x_l).unsqueeze(-1)
    return heatval


def pose_nms(
    heatmap_avg, poses, max_num_people: int, nms_threshold: float, nms_num_threshold: int, pose_score_threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    NMS for the regressed poses results.

    :param Tensor heatmap_avg: Avg of the heatmaps at all scales (1, 1+num_joints, w, h)
    :param List poses: Gather of the pose proposals [(num_people, num_joints, 3)]
    :param int max_num_people: Maximum number of decoded poses
    :param float nms_threshold: The maximum distance between two joints for them to be considered as belonging to the same pose.
                          Given in terms of a percentage of a square root of the area of the pose bounding box.
    :param int nms_num_threshold: Number of joints that must pass the NMS check for the pose to be considered as a valid one.
    :param float pose_score_threshold: Minimum confidence threshold for pose. Pose with confidence lower than this threshold will be discarded.

    :return Tuple of (poses, scores)
    """
    assert len(poses) == 1

    pose_score = torch.cat([pose[:, :, 2:] for pose in poses], dim=0)
    pose_coord = torch.cat([pose[:, :, :2] for pose in poses], dim=0)

    num_people, num_joints, _ = pose_coord.shape

    if num_people == 0:
        return np.zeros((0, num_joints, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    heatval = _get_heat_value(pose_coord, heatmap_avg[0])
    heat_score = (torch.sum(heatval, dim=1) / num_joints)[:, 0]

    pose_score = pose_score * heatval
    poses = torch.cat([pose_coord.cpu(), pose_score.cpu()], dim=2)

    keep_pose_inds = _nms_core(pose_coord, heat_score, nms_threshold=nms_threshold, nms_num_threshold=nms_num_threshold)
    poses = poses[keep_pose_inds]
    heat_score = heat_score[keep_pose_inds]

    if len(keep_pose_inds) > max_num_people:
        heat_score, topk_inds = torch.topk(heat_score, max_num_people)
        poses = poses[topk_inds]

    poses = poses.numpy()
    if len(poses):
        scores = poses[:, :, 2].mean(axis=1)

        mask = scores >= pose_score_threshold
        poses = poses[mask]
        scores = scores[mask]
    else:
        return np.zeros((0, num_joints, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return poses, scores


def aggregate_results(
    heatmap: Tensor, posemap: Tensor, output_stride: int, pose_center_score_threshold: float, max_num_people: int
) -> Tuple[Tensor, List[Tensor]]:
    """
    Get initial pose proposals and aggregate the results of all scale.
    Not this implementation works only for batch size of 1.

    :param heatmap: Heatmap at this scale (B, 1+num_joints, w, h)
    :param posemap: Posemap at this scale (B, 2*num_joints, w, h)
    :param output_stride: Ratio of input size / predictions size
    :param pose_center_score_threshold: (float) A minimum score of a pose center keypoint for pose to be considered as a potential candidate
    :param max_num_people: (int)

    :return:
        - heatmap_sum: Sum of the heatmaps (1, 1+num_joints, w, h)
        - poses (List): Gather of the pose proposals [B, (num_people, num_joints, 3)]
    """

    poses = []

    h, w = heatmap[0].size(-1), heatmap[0].size(-2)

    heatmap_sum = _up_interpolate(heatmap, size=(int(output_stride * w), int(output_stride * h)))
    center_heatmap = heatmap[0, -1:]
    pose_ind, ctr_score = _get_maximum_from_heatmap(center_heatmap, pose_center_score_threshold=pose_center_score_threshold, max_num_people=max_num_people)
    posemap = posemap[0].permute(1, 2, 0).view(h * w, -1, 2)
    pose = output_stride * posemap[pose_ind]
    ctr_score = ctr_score[:, None].expand(-1, pose.shape[-2])[:, :, None]
    poses.append(torch.cat([pose, ctr_score], dim=2))

    return heatmap_sum, poses


class DEKRPoseEstimationDecodeCallback(nn.Module):
    """
    Class that implements decoding logic of DEKR's model predictions into poses.
    """

    def __init__(
        self,
        output_stride: int,
        max_num_people: int,
        keypoint_threshold: float,
        nms_threshold: float,
        nms_num_threshold: int,
        apply_sigmoid: bool,
        min_confidence: float = 0.0,
    ):
        """

        :param output_stride: Output stride of the model
        :param int max_num_people: Maximum number of decoded poses
        :param float keypoint_threshold: (float) A minimum score of a pose center keypoint for pose to be considered as a potential candidate
        :param float nms_threshold: The maximum distance between two joints for them to be considered as belonging to the same pose.
                              Given in terms of a percentage of a square root of the area of the pose bounding box.
        :param int nms_num_threshold: Number of joints that must pass the NMS check for the pose to be considered as a valid one.
        :param bool apply_sigmoid: If True, apply the sigmoid activation on heatmap. This is needed when heatmap is not
                              bound to [0..1] range and trained with logits (E.g focal loss)
        :param float min_confidence: Minimum confidence threshold for pose
        """
        super().__init__()
        self.keypoint_threshold = keypoint_threshold
        self.max_num_people = max_num_people
        self.output_stride = output_stride
        self.nms_threshold = nms_threshold
        self.nms_num_threshold = nms_num_threshold
        self.apply_sigmoid = apply_sigmoid
        self.min_confidence = min_confidence

    @torch.no_grad()
    def forward(self, predictions: Union[Tensor, Tuple[Tensor, Tensor]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """

        :param predictions: Tuple (heatmap, offset):
            heatmap - [BatchSize, NumJoints+1,H,W]
            offset - [BatchSize, NumJoints*2,H,W]

        :return: Tuple
        """
        all_poses = []
        all_scores = []

        heatmap, offset = predictions
        batch_size = len(heatmap)
        for i in range(batch_size):
            poses, scores = self.decode_one_sized_batch(predictions=(heatmap[i : i + 1], offset[i : i + 1]))
            all_poses.append(poses)
            all_scores.append(scores)
        return all_poses, all_scores

    def decode_one_sized_batch(self, predictions: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        heatmap, offset = predictions
        posemap = _offset_to_pose(offset)  # [1, 2 * num_joints, H, W]

        if heatmap.size(0) != 1:
            raise RuntimeError("Batch size of 1 is required")

        if self.apply_sigmoid:
            heatmap = heatmap.sigmoid()

        heatmap_sum, poses_sum = aggregate_results(
            heatmap,
            posemap,
            pose_center_score_threshold=self.keypoint_threshold,
            max_num_people=self.max_num_people,
            output_stride=self.output_stride,
        )

        poses, scores = pose_nms(
            heatmap_sum,
            poses_sum,
            max_num_people=self.max_num_people,
            nms_threshold=self.nms_threshold,
            nms_num_threshold=self.nms_num_threshold,
            pose_score_threshold=self.min_confidence,
        )

        if len(poses) != len(scores):
            raise RuntimeError("Decoding error detected. Returned mismatching number of poses/scores")

        return poses, scores
