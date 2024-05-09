import unittest
from super_gradients.training import models
from super_gradients.training import Trainer
from super_gradients.training.dataloaders.dataloaders import kitti2015_optical_flow_val
from super_gradients.training.metrics import EPE
from super_gradients.training.transforms import OpticalFlowInputPadder, OpticalFlowNormalize


class RAFTIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.data_dir = "/home/yael.baron/data/kitti"
        self.dl = kitti2015_optical_flow_val(
            dataset_params=dict(root=self.data_dir, transforms=[OpticalFlowInputPadder(dataset_mode="kitti", pad_factor=8), OpticalFlowNormalize()]),
            dataloader_params=dict(batch_size=1),
        )

    def test_raft_s_kitti(self):
        trainer = Trainer("test_raft_s")
        model = models.get("raft_s", num_classes=1, checkpoint_path="/home/yael.baron/checkpoints/RAFT_pretrained_weights/raft-small.pth")
        # model = models.get("raft_s", num_classes=1, pretrained_weights="flying_things")
        metric = EPE(apply_unpad=True)
        metric_values = trainer.test(model=model, test_loader=self.dl, test_metrics_list=[metric])
        self.assertAlmostEqual(metric_values["epe"], 7.672, delta=0.1)

    def test_raft_l_kitti(self):
        trainer = Trainer("test_raft_l")
        model = models.get("raft_l", num_classes=1, checkpoint_path="/home/yael.baron/checkpoints/RAFT_pretrained_weights/raft-things.pth")
        # model = models.get("raft_l", num_classes=1, pretrained_weights="flying_things")
        metric = EPE(apply_unpad=True)
        metric_values = trainer.test(model=model, test_loader=self.dl, test_metrics_list=[metric])
        self.assertAlmostEqual(metric_values["epe"], 5.044, delta=0.001)


if __name__ == "__main__":
    unittest.main()
