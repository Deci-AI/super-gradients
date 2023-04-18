import torch

from super_gradients.training.models.pose_estimation_models import PoseRescoringNet


def test_rescoring_net():
    links = [
        [0, 1],
        [0, 2],
        [1, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6],
        [5, 6],
        [5, 7],
        [5, 11],
        [6, 8],
        [6, 12],
        [7, 9],
        [8, 10],
        [11, 12],
        [11, 13],
        [12, 14],
        [13, 15],
        [14, 16],
    ]

    # net = PoseRescoringNet(num_classes=17, hidden_channels=128, num_layers=2, joint_links=links)
    # x = torch.randn(32, 17, 3)
    # y = net(x)
    # print(y.shape)

    net = PoseRescoringNet(num_classes=17, hidden_channels=128, num_layers=2, joint_links=links)
    x = torch.randn(1, 32, 17, 3)
    y = net(x)
    print(y.shape)
