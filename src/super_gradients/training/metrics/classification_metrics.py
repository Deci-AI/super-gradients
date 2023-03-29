import torch
import torchmetrics
from torchmetrics import Metric

from super_gradients.common.object_names import Metrics
from super_gradients.common.registry.registry import register_metric
from super_gradients.training.utils import convert_to_tensor


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    :param output: Tensor / Numpy / List
        The prediction
    :param target: Tensor / Numpy / List
        The corresponding lables
    :param topk: tuple
        The type of accuracy to calculate, e.g. topk=(1,5) returns accuracy for top-1 and top-5"""
    # Convert to tensor
    output = convert_to_tensor(output)
    target = convert_to_tensor(target)

    # Get the maximal value of the accuracy measurment and the batch size
    maxk = max(topk)
    batch_size = target.size(0)

    # Get the top k predictions
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # Count the number of correct predictions only for the highest k
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # Count the number of correct prediction for the different K (the top predictions) values
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


@register_metric(Metrics.ACCURACY)
class Accuracy(torchmetrics.Accuracy):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.greater_is_better = True

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if target.shape == preds.shape:
            target = target.argmax(1)  # supports smooth labels
        super().update(preds=preds.argmax(1), target=target)


@register_metric(Metrics.TOP5)
class Top5(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.greater_is_better = True

        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if target.shape == preds.shape:
            target = target.argmax(1)  # supports smooth labels

        # Get the maximal value of the accuracy measurment and the batch size
        batch_size = target.size(0)

        # Get the top k predictions
        _, pred = preds.topk(5, 1, True, True)
        pred = pred.t()
        # Count the number of correct predictions only for the highest 5
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct5 = correct[:5].reshape(-1).float().sum(0)
        self.correct += correct5
        self.total += batch_size

    def compute(self):
        return self.correct.float() / self.total


class ToyTestClassificationMetric(Metric):
    """
    Dummy classification Mettric object returning 0 always (for testing).
    """

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        pass

    def compute(self):
        return 0
