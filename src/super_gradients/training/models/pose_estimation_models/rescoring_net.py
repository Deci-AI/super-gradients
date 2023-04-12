from torch import nn, Tensor
import numpy as np


class RescoringNet(nn.Module):
    """
    Rescoring network for pose estimation. It takes input features and predicts the single scalar score
    which is the multiplication factor for original score prediction. This model learns what are the reasonable/possible
    joint configurations. So it may downweight confidence of impossible joint configurations.

    The model is a simple 3-layer MLP with ReLU activation. The input is the concatenation of the predicted poses and prior
    information in the form of the joint links. See RescoringNet.get_feature() for details.
    The output is a single scalar value.
    """

    def __init__(self, in_channels: int, hidden_channels):
        super(RescoringNet, self).__init__()
        self.l1 = nn.Linear(in_channels, hidden_channels, bias=True)
        self.l2 = nn.Linear(hidden_channels, hidden_channels, bias=True)
        self.l3 = nn.Linear(hidden_channels, 1, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.relu(self.l1(x))
        x2 = self.relu(self.l2(x1))
        y_pred = self.l3(x2)
        return y_pred

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    @classmethod
    def get_feature(cls, poses: np.ndarray, joint_links: np.ndarray) -> np.ndarray:
        """
        Compute the feature vector input to the rescoring network.

        :param poses: [N, J, 3] Predicted poses
        :param joint_links: [L,2] List of joint indices
        :return: [N, L*2+L+J] Feature vector
        """
        joint_xy = poses[:, :, :2]
        visibility = poses[:, :, 2]

        joint_1 = joint_links[:, 0]
        joint_2 = joint_links[:, 1]

        # To get the Delta x Delta y
        joint_relate = joint_xy[:, joint_1] - joint_xy[:, joint_2]
        joint_length = ((joint_relate**2)[:, :, 0] + (joint_relate**2)[:, :, 1]) ** (0.5)

        # To use the torso distance to normalize
        normalize = (joint_length[:, 9] + joint_length[:, 11]) / 2
        normalize = np.tile(normalize, (len(joint_1), 2, 1)).transpose(2, 0, 1)
        normalize[normalize < 1] = 1

        joint_length = joint_length / normalize[:, :, 0]
        joint_relate = joint_relate / normalize
        joint_relate = joint_relate.reshape((-1, len(joint_1) * 2))

        feature = [joint_relate, joint_length, visibility]
        feature = np.concatenate(feature, axis=1)
        return feature
