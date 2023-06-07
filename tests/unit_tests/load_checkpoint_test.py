import unittest

import torch.nn.init
from torch import nn

from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training.utils.checkpoint_utils import transfer_weights


class LoadCheckpointTest(unittest.TestCase):
    def test_transfer_weights(self):
        class Foo(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 10)
                self.fc2 = nn.Linear(10, 10)
                torch.nn.init.zeros_(self.fc1.weight)
                torch.nn.init.zeros_(self.fc2.weight)

        class Bar(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 11)
                self.fc2 = nn.Linear(10, 10)
                torch.nn.init.ones_(self.fc1.weight)
                torch.nn.init.ones_(self.fc2.weight)

        foo = Foo()
        bar = Bar()
        self.assertFalse((foo.fc2.weight == bar.fc2.weight).all())
        transfer_weights(foo, bar.state_dict())
        self.assertTrue((foo.fc2.weight == bar.fc2.weight).all())

    def test_checkpoint_path_url(self):
        m1 = models.get(Models.YOLO_NAS_S, num_classes=80, checkpoint_path="https://sghub.deci.ai/models/yolo_nas_s_coco.pth")
        m2 = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")
        m1_state = m1.state_dict()
        m2_state = m2.state_dict()
        self.assertTrue(m1_state.keys() == m2_state.keys())
        for k in m1_state.keys():
            self.assertTrue((m1_state[k] == m2_state[k]).all())
