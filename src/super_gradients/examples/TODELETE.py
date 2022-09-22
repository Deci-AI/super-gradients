import omegaconf
import hydra
import pkg_resources

import torch
import torch.nn as nn
import torch.nn.functional as F

from super_gradients import Trainer, init_trainer
from super_gradients.common.registry import register_model


@register_model('my_conv_net')  # will be registered as "my_conv_net"
class MyConvNet(nn.Module):
   def __init__(self, num_classes: int):
      super().__init__()
      self.conv1 = nn.Conv2d(3, 6, 5)
      self.pool = nn.MaxPool2d(2, 2)
      self.conv2 = nn.Conv2d(6, 16, 5)
      self.fc1 = nn.Linear(16 * 5 * 5, 120)
      self.fc2 = nn.Linear(120, 84)
      self.fc3 = nn.Linear(84, num_classes)

   def forward(self, x):
      x = self.pool(F.relu(self.conv1(x)))
      x = self.pool(F.relu(self.conv2(x)))
      x = torch.flatten(x, 1)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      return x


@hydra.main(config_path=pkg_resources.resource_filename("super_gradients.recipes", ""), version_base="1.2")
def main(cfg: omegaconf.DictConfig) -> None:
   Trainer.train_from_config(cfg)


init_trainer()
main()
