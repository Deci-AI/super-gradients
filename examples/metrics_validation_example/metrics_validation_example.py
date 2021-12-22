import infery
from torch.utils.data import DataLoader
import os
import shutil
from typing import Union

import torch
import logging
from super_gradients.training import SgModel
from super_gradients.training.datasets import DatasetInterface
from super_gradients.training.metrics import Accuracy, Top5, DetectionMetrics, PixelAccuracy, IoU
from super_gradients.training.utils.detection_utils import DetectionPostPredictionCallback
from super_gradients.training.datasets.dataset_interfaces import ImageNetDatasetInterface




def test_metrics(inferencer,
                 data_loader: Union[DatasetInterface, DataLoader],
                 task: str,
                 device: str = 'cpu',
                 post_predict_callback: DetectionPostPredictionCallback = None,
                 num_cls: int = None,
                 task_metrics: list = None) -> ...:
    """
    run the test() function of the model
    :param data_loader:             a deci DatasetInterface or DataLoader
    :param task:                    one of ['classification', 'detection','segmentation']
    :param device:                  device ['cpu', 'cuda'] to carry the test on
    :param post_predict_callback:   a callback which will be executed by the test() function after the model
                                    prediction. Typically, this will receive the prediction output to preform some
                                    post processing
    :param task_metrics              list metrics to be tested, if not given default metrics for the task are set.
    :return:                        the result tuple from the test() function
    """
    if task_metrics is None:
        if task == 'classification':
            task_metrics = [Accuracy(), Top5()]
        elif task == 'detection':
            task_metrics = [DetectionMetrics(post_prediction_callback=post_predict_callback,
                                             num_cls=num_cls,
                                             img_ids=data_loader.testset.get_img_ids(),
                                             height=data_loader.testset.img_size,
                                             width=data_loader.testset.img_size)]
        elif task == 'segmentation':
            task_metrics = [IoU(num_cls), PixelAccuracy()]
        else:
            raise RuntimeError(f'Metric for task {task} is not defined')

    deci_model = SgModel("inferencer_test_metrics", device=device, model_checkpoints_location='local')

    # FIXME: OTHER CASES THAN CLASSIFICATION ARE NOT CURRENTLY TESTED
    if isinstance(data_loader, DatasetInterface):
        deci_model.connect_dataset_interface(data_loader, data_loader_num_workers=8)
        # not all datasets have a testset and test_loader, use val_loader instead.
        if deci_model.test_loader is None:
            deci_model.test_loader = deci_model.valid_loader
            logging.info("test_loader of dataset interface is None, using valid_loader instead.")
    elif isinstance(data_loader, DataLoader):
        deci_model.test_loader = data_loader
    else:
        raise NotImplementedError(f'{data_loader} is not a supported input for data_loader, please'
                                  f'use either DatasetInterface or DataLoader')

    deci_model.net = InferencerTestMetricsConnector(inferencer, device)
    results_tuple = deci_model.test(test_metrics_list=task_metrics)

    # Delete the temporary checkpoint folder
    if os.path.isdir(deci_model.checkpoints_dir_path):
        shutil.rmtree(deci_model.checkpoints_dir_path)

    return results_tuple


class InferencerTestMetricsConnector:
    """
    An abstract class for connecting the BaseInferencer to the SgModel class, so that the latter's
    test methods (accuracy on real data set etc) can be used in conjunction with the Inferencer.
    """

    def __init__(self, inferencer, device: str = 'cpu'):
        self.inferencer = inferencer
        self.device = device

    def __call__(self, x):
        x = x.cpu().numpy()
        output_list = self.inferencer.predict(x)
        torch_output = [torch.from_numpy(output).to(self.device) for output in output_list]
        return torch_output[0] if len(torch_output) == 1 else torch_output

    def eval(self):
        pass  # Needed so SgModel's test() method doesn't break
dataset = ImageNetDatasetInterface(data_dir="/data/Imagenet", dataset_params={"batch_size": 128})
model = infery.load(model_path="/home/shay.aharon/decinets/decinet1_1_1.pkl", framework_type="trt", inference_hardware="gpu")
test_metrics(data_loader=dataset, inferencer=model, device="gpu", num_cls=1000, task="classification")