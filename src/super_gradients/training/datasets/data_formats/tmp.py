import torch
from torch import Tensor, nn

ten = torch.tensor(
    [
        [
            [
                1,
                2,
                3,
                4,
            ],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ],
        [
            [
                1,
                2,
                3,
                4,
            ],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ],
        [
            [
                1,
                2,
                3,
                4,
            ],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ],
        [
            [
                1,
                2,
                3,
                4,
            ],
            [1, 2, 3, 4],
            [1, 2, 3, 4],
        ],
    ]
)


class IndexedConcatenatedOutput(nn.Module):
    def __init__(self, batch_size: int, num_predictions: int):
        super().__init__()
        self._indices = torch.range(0, batch_size - 1).repeat(1, 1).T.repeat(1, num_predictions)[:, :, None]
        self._batch_size = batch_size
        self._num_predictions = num_predictions

    def forward(self, input: Tensor) -> Tensor:
        concatenated_tensor = torch.cat([self._indices, input], dim=-1)
        return torch.reshape(concatenated_tensor, [self._batch_size * self._num_predictions, -1])


f = IndexedConcatenatedOutput(4, 3)

res = f(ten)
print(res)
