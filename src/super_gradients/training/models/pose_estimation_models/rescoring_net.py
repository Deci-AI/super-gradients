import copy
from typing import List, Tuple

import torch
from torch import nn, Tensor

__all__ = ["PoseRescoringNet"]

from super_gradients.common.object_names import Models
from super_gradients.common.registry import register_model
from super_gradients.training.models import SgModule
from super_gradients.training.utils import HpmStruct


@register_model(Models.POSE_RESCORING)
class PoseRescoringNet(SgModule):
    """
    Rescoring network for pose estimation. It takes input features and predicts the single scalar score
    which is the multiplication factor for original score prediction. This model learns what are the reasonable/possible
    joint configurations. So it may downweight confidence of impossible joint configurations.

    The model is a simple 3-layer MLP with ReLU activation. The input is the concatenation of the predicted poses and prior
    information in the form of the joint links. See RescoringNet.get_feature() for details.
    The output is a single scalar value.
    """

    def __init__(self, num_classes: int, hidden_channels: int, num_layers: int, edge_links: List[Tuple[int, int]]):
        super(PoseRescoringNet, self).__init__()
        in_channels = len(edge_links) * 2 + len(edge_links) + num_classes  # [joint_relate, joint_length, visibility]
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(in_channels, hidden_channels, bias=True))
            layers.append(nn.ReLU())
            in_channels = hidden_channels
        self.layers = nn.Sequential(*layers)
        self.final = nn.Linear(hidden_channels, 1, bias=True)
        self.edge_links = torch.tensor(edge_links).long()

    def forward(self, poses: Tensor) -> Tuple[Tensor, Tensor]:
        """

        :param x: Predicted poses or shape [N, J, 3] or [B, N, J, 3]
        :return: Tuple of input poses and corresponding scores
        """

        x = self.get_feature(poses, self.edge_links)
        x = self.layers(x)
        y_pred = self.final(x)
        return poses, y_pred

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    @classmethod
    def get_feature(cls, poses: Tensor, edge_links: Tensor) -> Tensor:
        """
        Compute the feature vector input to the rescoring network.

        :param poses: [N, J, 3] Predicted poses
        :param edge_links: [L,2] List of joint indices
        :return: [N, L*2+L+J] Feature vector
        """
        joint_xy = poses[..., :2]
        visibility = poses[..., 2]

        joint_1 = edge_links[:, 0]
        joint_2 = edge_links[:, 1]

        # To get the Delta x Delta y
        joint_relate = joint_xy[..., joint_1, :] - joint_xy[..., joint_2, :]  # [N, L, 2]
        joint_length = ((joint_relate**2)[..., 0] + (joint_relate**2)[..., 1]) ** (0.5)  # [N, L]

        # To use the torso distance to normalize
        normalize = (joint_length[..., 9] + joint_length[..., 11]) / 2  # [N] # NOTE: THIS IS COCO-SPECIFIC
        normalize_tiled = torch.tile(normalize, (len(joint_1), 2, 1)).permute(2, 0, 1)
        normalize_tiled = normalize_tiled.clamp_min(1)

        joint_length = joint_length / normalize_tiled[..., 0]
        joint_relate = joint_relate / normalize_tiled
        joint_relate = torch.flatten(joint_relate, start_dim=-2)  # .reshape((-1, len(joint_1) * 2))

        feature = [joint_relate, joint_length, visibility]
        feature = torch.cat(feature, dim=-1)
        return feature


@register_model(Models.POSE_RESCORING_COCO)
class COCOPoseRescoringNet(PoseRescoringNet):
    def __init__(self, arch_params):
        from super_gradients.training.models import get_arch_params

        RESCORING_POSE_DEKR_ARCH_PARAMS = get_arch_params("pose_dekr_coco_rescoring_arch_params")
        merged_arch_params = HpmStruct(**copy.deepcopy(RESCORING_POSE_DEKR_ARCH_PARAMS))
        merged_arch_params.override(**arch_params.to_dict())
        super().__init__(
            num_classes=merged_arch_params.num_classes,
            hidden_channels=merged_arch_params.hidden_channels,
            num_layers=merged_arch_params.num_layers,
            edge_links=merged_arch_params.edge_links,
        )
