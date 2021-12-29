from __future__ import print_function, absolute_import

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

from super_gradients.training.utils import convert_to_tensor


class RSquaredLoss(_Loss):

    def forward(self, output, target):
        # FIXME - THIS NEEDS TO BE CHANGED SUCH THAT THIS CLASS INHERETS FROM _Loss (TAKE A LOOK AT YoLoV3DetectionLoss)
        """Computes the R-squared for the output and target values
        :param output: Tensor / Numpy / List
            The prediction
        :param target: Tensor / Numpy / List
            The corresponding lables
        """
        # Convert to tensor
        output = convert_to_tensor(output)
        target = convert_to_tensor(target)

        criterion_mse = nn.MSELoss()
        return 1 - criterion_mse(output, target).item() / torch.var(target).item()
