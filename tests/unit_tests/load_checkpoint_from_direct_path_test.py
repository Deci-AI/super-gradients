import shutil
import tempfile
import unittest
import os
from super_gradients.training import Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from super_gradients.training.sg_trainer.sg_trainer import StrictLoad


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 3 * 3, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LoadCheckpointFromDirectPathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_working_file_dir = tempfile.TemporaryDirectory(prefix='load_checkpoint_test').name
        if not os.path.isdir(cls.temp_working_file_dir):
            os.mkdir(cls.temp_working_file_dir)
        cls.checkpoint_path = cls.temp_working_file_dir + '/load_checkpoint_test.pth'

        # Setup the model
        cls.original_torch_net = Net()

        # Save the model's checkpoint
        torch.save(cls.original_torch_net.state_dict(), cls.checkpoint_path)

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls.temp_working_file_dir):
            shutil.rmtree(cls.temp_working_file_dir)

    def test_external_checkpoint_loaded_correctly(self):
        # Define Model
        new_torch_net = Net()

        # Make sure we initialized a model with different weights
        assert not self.check_models_have_same_weights(new_torch_net, self.original_torch_net)

        # Build the Trainer and load the checkpoint
        trainer = Trainer("load_checkpoint_test", model_checkpoints_location='local')
        trainer.build_model(new_torch_net, arch_params={'num_classes': 10},
                          checkpoint_params={'external_checkpoint_path': self.checkpoint_path,
                                             'load_checkpoint': True,
                                             'strict_load': StrictLoad.NO_KEY_MATCHING})

        # Assert the weights were loaded correctly
        assert self.check_models_have_same_weights(trainer.net, self.original_torch_net)

    def check_models_have_same_weights(self, model_1, model_2):
        model_1, model_2 = model_1.to('cpu'), model_2.to('cpu')
        models_differ = 0
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if (key_item_1[0] == key_item_2[0]):
                    print(f'Layer names match but layers have different weights for layers: {key_item_1[0]}')
        if models_differ == 0:
            return True
        else:
            return False


if __name__ == '__main__':
    unittest.main()
