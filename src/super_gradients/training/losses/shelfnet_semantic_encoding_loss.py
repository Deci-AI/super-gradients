import torch
from torch import nn
from torch.autograd import Variable

from super_gradients.common.object_names import Losses
from super_gradients.common.registry.registry import register_loss


@register_loss(Losses.SHELFNET_SE_LOSS)
class ShelfNetSemanticEncodingLoss(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""

    # FIXME - THIS LOSS SHOULD BE CHANGED TO SUPPORT APEX
    def __init__(self, se_weight=0.2, nclass=21, aux_weight=0.4, weight=None, ignore_index=-1):
        super().__init__(weight, None, ignore_index)
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight

        # FIXME - TEST CODE LOTEM, CHANGED IN ORDER TO WORK WITH apex.amp
        self.bcewithlogitsloss = nn.BCELoss(weight)

    def forward(self, logits, labels):
        pred1, se_pred, pred2 = logits

        batch = labels.size(0)
        se_target = Variable(torch.zeros(batch, self.nclass))
        # FIXME - THIS IS WHAT apex MIGHT BE FAILING TO WORK WITH
        for i in range(batch):
            hist = torch.histc(labels[i].cpu().data.float(), bins=self.nclass, min=0, max=self.nclass - 1)
            vect = hist > 0
            se_target[i] = vect

        loss1 = super().forward(pred1, labels)
        loss2 = super().forward(pred2, labels)
        loss3 = self.bcewithlogitsloss(torch.sigmoid(se_pred), se_target.data.cuda())  # FIXME - MAYBE CHANGE TO SIGMOID
        total_loss = loss1 + self.aux_weight * loss2 + self.se_weight * loss3
        losses = [loss1, loss2, loss3, total_loss]
        return total_loss, torch.stack(losses, dim=0).detach()

    @property
    def component_names(self):
        """
        Component names for logging during training.
        These correspond to 2nd item in the tuple returned in self.forward(...).
        See super_gradients.Trainer.train() docs for more info.
        """
        return ["loss1", "loss2", "loss3", "total_loss"]
