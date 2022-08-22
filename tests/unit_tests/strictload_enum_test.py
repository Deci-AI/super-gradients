import os
import shutil
import tempfile
import unittest

from super_gradients.common.sg_loggers import BaseSGLogger
from super_gradients.training import Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F

from super_gradients.training import models
from super_gradients.training.sg_trainer.sg_trainer import StrictLoad
from super_gradients.training.utils import HpmStruct


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


class StrictLoadEnumTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_working_file_dir = tempfile.TemporaryDirectory(prefix='strict_load_test').name
        if not os.path.isdir(cls.temp_working_file_dir):
            os.mkdir(cls.temp_working_file_dir)

        cls.experiment_name = 'load_checkpoint_test'

        cls.checkpoint_diff_keys_name = 'strict_load_test_diff_keys.pth'
        cls.checkpoint_diff_keys_path = cls.temp_working_file_dir + '/' + cls.checkpoint_diff_keys_name

        # Setup the trainer
        cls.original_torch_net = Net()

        # Save the trainer's state_dict checkpoint with different keys
        torch.save(cls.change_state_dict_keys(cls.original_torch_net.state_dict()), cls.checkpoint_diff_keys_path)

        # Save the trainer's state_dict checkpoint in Trainer format
        cls.sg_trainer = Trainer("load_checkpoint_test", model_checkpoints_location='local')  # Saves in /checkpoints
        cls.sg_trainer.build_model(cls.original_torch_net, arch_params={'num_classes': 10})
        # FIXME: after uniting init and build_model we should remove this
        cls.sg_trainer.sg_logger = BaseSGLogger('project_name', 'load_checkpoint_test', 'local', resumed=False,
                                                training_params=HpmStruct(max_epochs=10),
                                                checkpoints_dir_path=cls.sg_trainer.checkpoints_dir_path)
        cls.sg_trainer._save_checkpoint()

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls.temp_working_file_dir):
            shutil.rmtree(cls.temp_working_file_dir)

    @classmethod
    def change_state_dict_keys(self, state_dict):
        new_ckpt_dict = {}
        for i, (ckpt_key, ckpt_val) in enumerate(state_dict.items()):
            new_ckpt_dict[str(i)] = ckpt_val
        return new_ckpt_dict

    def check_models_have_same_weights(self, model_1, model_2):
        model_1, model_2 = model_1.to('cpu'), model_2.to('cpu')
        models_differ = 0
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if (key_item_1[0] == key_item_2[0]):
                    print('Mismtach found at', key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            return True
        else:
            return False

    def test_strict_load_on(self):
        # Define Model
        net = models.get('resnet18', arch_params={"num_classes": 1000})
        pretrained_net = models.get('resnet18', arch_params={"num_classes": 1000},
                                    pretrained_weights="imagenet")

        # Make sure we initialized a trainer with different weights
        assert not self.check_models_have_same_weights(net, pretrained_net)

        pretrained_sd_path = os.path.join(self.temp_working_file_dir, "pretrained_net_strict_load_on.pth")
        torch.save(pretrained_net.state_dict(), pretrained_sd_path)

        net = models.get('resnet18', arch_params={"num_classes": 1000},
                         checkpoint_path=pretrained_sd_path, strict_load=StrictLoad.ON)

        # Assert the weights were loaded correctly
        assert self.check_models_have_same_weights(net, pretrained_net)

    def test_strict_load_off(self):
        # Define Model
        net = models.get('resnet18', arch_params={"num_classes": 1000})
        pretrained_net = models.get('resnet18', arch_params={"num_classes": 1000},
                                    pretrained_weights="imagenet")

        # Make sure we initialized a trainer with different weights
        assert not self.check_models_have_same_weights(net, pretrained_net)

        pretrained_sd_path = os.path.join(self.temp_working_file_dir, "pretrained_net_strict_load_off.pth")
        del pretrained_net.linear
        torch.save(pretrained_net.state_dict(), pretrained_sd_path)

        with self.assertRaises(RuntimeError):
            models.get('resnet18', arch_params={"num_classes": 1000},
                       checkpoint_path=pretrained_sd_path, strict_load=StrictLoad.ON)

        net = models.get('resnet18', arch_params={"num_classes": 1000},
                         checkpoint_path=pretrained_sd_path, strict_load=StrictLoad.OFF)
        del net.linear
        # Assert the weights were loaded correctly
        assert self.check_models_have_same_weights(net, pretrained_net)

    def test_strict_load_no_key_matching_sg_checkpoint(self):
        # Define Model
        net = models.get('resnet18', arch_params={"num_classes": 1000})
        pretrained_net = models.get('resnet18', arch_params={"num_classes": 1000},
                                    pretrained_weights="imagenet")

        # Make sure we initialized a trainer with different weights
        assert not self.check_models_have_same_weights(net, pretrained_net)

        pretrained_sd_path = os.path.join(self.temp_working_file_dir, "pretrained_net_strict_load_soft.pth")
        torch.save(self.change_state_dict_keys(pretrained_net.state_dict()), pretrained_sd_path)

        with self.assertRaises(RuntimeError):
            models.get('resnet18', arch_params={"num_classes": 1000},
                       checkpoint_path=pretrained_sd_path, strict_load=StrictLoad.ON)

        net = models.get('resnet18', arch_params={"num_classes": 1000},
                         checkpoint_path=pretrained_sd_path, strict_load=StrictLoad.NO_KEY_MATCHING)
        # Assert the weights were loaded correctly
        assert self.check_models_have_same_weights(net, pretrained_net)


if __name__ == '__main__':
    unittest.main()
