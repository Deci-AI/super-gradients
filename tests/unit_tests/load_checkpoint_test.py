import unittest

import torch.nn.init
from torch import nn

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
