import unittest
from super_gradients import SgModel, \
    ClassificationTestDatasetInterface
from super_gradients.training.metrics import Accuracy, Top5
from super_gradients.training.models import ResNet18
from torch.optim import SGD
from super_gradients.training.utils.callbacks import DeciLabUploadCallback, ModelConversionCheckCallback
from deci_lab_client.models import Metric, QuantizationLevel, ModelMetadata, OptimizationRequestForm


class DeciLabUploadTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = SgModel("deci_lab_export_test_model", model_checkpoints_location='local')
        dataset = ClassificationTestDatasetInterface(dataset_params={"batch_size": 10})
        self.model.connect_dataset_interface(dataset)
        net = ResNet18(num_classes=5, arch_params={})
        self.optimizer = SGD(params=net.parameters(), lr=0.1)
        self.model.build_model(net)

    def test_train_with_deci_lab_integration(self):
        model_meta_data = ModelMetadata(name='model_for_deci_lab_upload_test',
                                        primary_batch_size=1,
                                        architecture='Resnet18',
                                        framework='pytorch',
                                        dl_task='classification',
                                        input_dimensions=(3, 224, 224),
                                        primary_hardware='XEON',
                                        dataset_name='imagenet',
                                        description='ResNet18 ONNX deci.ai Test',
                                        tags=['imagenet',
                                              'resnet18'])

        optimization_request_form = OptimizationRequestForm(target_hardware='XEON',
                                                            target_batch_size=1,
                                                            target_metric=Metric.LATENCY,
                                                            optimize_model_size=True,
                                                            quantization_level=QuantizationLevel.FP16,
                                                            optimize_autonac=True)

        model_conversion_callback = ModelConversionCheckCallback(model_meta_data=model_meta_data)
        deci_lab_callback = DeciLabUploadCallback(email="trainer-tester@testcase.ai",
                                                  model_meta_data=model_meta_data,
                                                  optimization_request_form=optimization_request_form)

        train_params = {"max_epochs": 2, "lr_updates": [1], "lr_decay_factor": 0.1, "lr_mode": "step",
                        "lr_warmup_epochs": 0, "initial_lr": 0.1, "loss": "cross_entropy", "optimizer": self.optimizer,
                        "criterion_params": {},
                        "train_metrics_list": [Accuracy(), Top5()], "valid_metrics_list": [Accuracy(), Top5()],
                        "loss_logging_items_names": ["Loss"], "metric_to_watch": "Accuracy",
                        "greater_metric_to_watch_is_better": True,
                        "phase_callbacks": [model_conversion_callback, deci_lab_callback]}

        self.model.train(train_params)

        # CLEANUP

        # FIXME: MISUSE OF DECI_PLATFROM CALLBACK:
        #  https://github.com/Deci-AI/deci_trainer/pull/106/files/2ed12b78adc9886faabad9d952969ff5479e9237#r708092979
        new_model_from_repo_name = model_meta_data.name + '_1_1'

        your_model_from_repo = deci_lab_callback.platform_client.get_model_by_name(name=new_model_from_repo_name).data
        deci_lab_callback.platform_client.delete_model(your_model_from_repo.model_id)


if __name__ == '__main__':
    unittest.main()
