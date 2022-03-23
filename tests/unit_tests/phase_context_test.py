import unittest
from super_gradients.training.utils.callbacks import PhaseContextTestCallback, Phase
from super_gradients import SgModel, \
    ClassificationTestDatasetInterface
from super_gradients.training.metrics import Accuracy, Top5
from super_gradients.training.models import ResNet18
import torch
from super_gradients.training.utils.utils import AverageMeter
from torchmetrics import MetricCollection


class PhaseContextTest(unittest.TestCase):
    def context_information_in_train_test(self):
        model = SgModel("context_information_in_train_test", model_checkpoints_location='local')
        dataset_params = {"batch_size": 10}
        dataset = ClassificationTestDatasetInterface(dataset_params=dataset_params)
        model.connect_dataset_interface(dataset)

        net = ResNet18(num_classes=5, arch_params={})
        model.build_model(net, load_checkpoint=False)

        phase_callbacks = [PhaseContextTestCallback(Phase.TRAIN_BATCH_END),
                           PhaseContextTestCallback(Phase.TRAIN_BATCH_STEP),
                           PhaseContextTestCallback(Phase.TRAIN_EPOCH_END),
                           PhaseContextTestCallback(Phase.VALIDATION_BATCH_END),
                           PhaseContextTestCallback(Phase.VALIDATION_EPOCH_END)]

        train_params = {"max_epochs": 2, "lr_updates": [1], "lr_decay_factor": 0.1, "lr_mode": "step",
                        "lr_warmup_epochs": 0, "initial_lr": 0.1, "loss": "cross_entropy", "optimizer": "SGD",
                        "criterion_params": {}, "optimizer_params": {"weight_decay": 1e-4, "momentum": 0.9},
                        "train_metrics_list": [Accuracy()], "valid_metrics_list": [Top5()],
                        "loss_logging_items_names": ["Loss"], "metric_to_watch": "Top5",
                        "greater_metric_to_watch_is_better": True, "phase_callbacks": phase_callbacks}

        model.train(train_params)
        context_callbacks = list(filter(lambda cb: isinstance(cb, PhaseContextTestCallback), model.phase_callbacks))

        # CHECK THAT PHASE CONTEXES HAVE THE EXACT INFORMATION THERY'RE SUPPOSE TO HOLD
        for phase_callback in context_callbacks:
            if phase_callback.phase in [Phase.TRAIN_BATCH_END, Phase.TRAIN_BATCH_STEP, Phase.VALIDATION_BATCH_END]:
                self.assertTrue(phase_callback.context.batch_idx == 0)
                self.assertTrue(phase_callback.context.criterion is not None)
                self.assertTrue(isinstance(phase_callback.context.inputs, torch.Tensor))
                self.assertTrue(isinstance(phase_callback.context.loss_avg_meter, AverageMeter))
                self.assertTrue(isinstance(phase_callback.context.loss_log_items, torch.Tensor))
                self.assertTrue(phase_callback.context.metrics_dict is None)
                self.assertTrue(isinstance(phase_callback.context.preds, torch.Tensor))
                self.assertTrue(isinstance(phase_callback.context.target, torch.Tensor))

                if phase_callback.phase == Phase.VALIDATION_BATCH_END:
                    self.assertTrue(phase_callback.context.epoch == 2)
                    self.assertTrue(isinstance(phase_callback.context.metrics_compute_fn, MetricCollection) and hasattr(phase_callback.context.metrics_compute_fn, "Top5"))

                else:
                    self.assertTrue(phase_callback.context.epoch == 1)
                    self.assertTrue(isinstance(phase_callback.context.metrics_compute_fn, MetricCollection) and hasattr(phase_callback.context.metrics_compute_fn, "Accuracy"))

        if phase_callback.phase in [Phase.TRAIN_EPOCH_END, Phase.VALIDATION_EPOCH_END]:
            self.assertTrue(phase_callback.context.batch_idx is None)
            self.assertTrue(phase_callback.context.criterion is None)
            self.assertTrue(phase_callback.context.inputs is None)
            self.assertTrue(phase_callback.context.loss_log_items is None)
            self.assertTrue(phase_callback.context.metrics_compute_fn is None)
            self.assertTrue(phase_callback.context.optimizer is not None)
            self.assertTrue(phase_callback.context.preds is None)
            self.assertTrue(phase_callback.context.target is None)
            self.assertTrue(phase_callback.context.epoch == 1)

            # EPOCH END PHASES USE THE SAME CONTEXT, WHICH IS UPDATED- SO VALID METRICS DICT SHOULD BE PRESENT
            self.assertTrue(isinstance(phase_callback.context.metrics_dict, dict) and "Loss" in phase_callback.context.metrics_dict.keys() and "Top5" in phase_callback.context.metrics_dict.keys())


if __name__ == '__main__':
    unittest.main()
