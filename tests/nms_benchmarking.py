import infery
import pandas as pd
import numpy as np
from enum import Enum
import time
from datetime import datetime
import os
import gc
from super_gradients.training.sg_model import SgModel

from super_gradients.training.utils.utils import HpmStruct
from super_gradients.training.models.detection_models.yolox import YoloX_S, YoloX_N, YoloX_T
from super_gradients.training.datasets import CoCoDetectionDatasetInterface
from super_gradients.training.datasets.dataset_interfaces.dataset_interface import CocoDetectionDatasetInterfaceV2
from super_gradients.training.models.detection_models.yolov5_base import YoloV5PostPredictionCallback
from super_gradients.training.utils.detection_utils import DetectionCollateFN

THROUGHPUT_BATCH_SIZES = [8, 16] #[8, 16, 32, 64]
LATENCY_BATCH_SIZES = [1]
WARMUP_REPETITIONS = 100
NMS_CONFIDENCE_THRESHOLD = 0.001
NMS_THRESHOLD = 0.0
NMS_IOU = 0.65


class ModelArchs(Enum):
    YOLOX_N = 'yolox_n'
    YOLOX_T = 'yolox_t'
    # YOLOX_S = 'yolox_s'


class NMSTypes(Enum):
    FIRST_100 = 'first_100'
    NO_NMS = 'no_nms'
    BATCHED_NMS = 'batched_nms'
    # TODO: MATRIX_NMS = 'matrix_nms'


class NMSBenchmarker:
    def __init__(self, _model_directory_base):
        self._model_directory_base = _model_directory_base
        """
        Expected Model Directory Structure:
        
        |
        -- yolox_s
        |    |
        |    ---- first_100.engine
        |    ---- no_nms.engine
        |    ---- batched_nms.engine
        -- yolox_t
        |    |
        |    ---- first_100.engine
        |    ---- no_nms.engine
        |    ---- batched_nms.engine
        -- yolox_n
             |
             ---- first_100.engine
             ---- no_nms.engine
             ---- batched_nms.engine
        """

    def _benchmark_internal(self, model_archs, batch_sizes,  nms_type, nms_device='cpu'):
        self.cleanup()
        results_dict = {}

        # If NO_NMS is compiled, use the Torch NMS module with the passed NMS device
        nms = YoloV5PostPredictionCallback(conf=NMS_CONFIDENCE_THRESHOLD, iou=NMS_IOU, classes=80) if \
            nms_type == NMSTypes.NO_NMS else (lambda y: y)

        for arch in model_archs:
            # Prep model and results bookkeeping
            results_dict[arch] = {}
            model_path = self.path_for_model(model_arch=arch, nms_type=nms_type)
            loaded_model = infery.load(model_path=model_path, framework_type='trt')

            # Start benchmarking different batch sizes
            for bs in batch_sizes:
                # FOR TESTING INPUT
                input_shape = [bs, 3, 416, 416]
                model = YoloX_N(HpmStruct()).eval()
                model.prep_model_for_conversion(input_shape)

                print(f'{arch} --- {bs}')
                data_loader = self.get_coco_data_loader(loaded_model=loaded_model, batch_size=bs)
                results_dict[arch][bs] = {}

                # Warmup
                dummy_input = np.random.rand(bs, *(loaded_model.input_dims[0]))
                for _ in range(WARMUP_REPETITIONS):
                    x = loaded_model.predict(dummy_input, output_device=nms_device)

                # Benchmark
                times = []

                for x in data_loader:
                    x = x[0].numpy()

                    # We're loaded and the input is converted - now time to benchmark (E2E - i.e., CPU -> CPU)
                    start = time.perf_counter()
                    x = loaded_model.predict(x, output_device=nms_device)
                    # x = nms(x)
                    times.append(time.perf_counter() - start)

                results_dict[arch][bs]['latency'] = sum(times)/len(times)
                results_dict[arch][bs]['throughput'] = len(times)*bs/sum(times)
                results_dict[arch][bs]['date'] = datetime.now()

        self.cleanup()
        return results_dict

    def no_nms_first_100_benchmarks(self, model_archs=None, batch_sizes=None):
        model_archs, batch_sizes = self._valid_archs_and_batchs(model_archs, batch_sizes)

        return self._benchmark_internal(model_archs=model_archs, batch_sizes=batch_sizes, nms_type=NMSTypes.FIRST_100)

    def trt_batched_nms(self, model_archs=None, batch_sizes=None):
        model_archs, batch_sizes = self._valid_archs_and_batchs(model_archs, batch_sizes)

        return self._benchmark_internal(model_archs=model_archs, batch_sizes=batch_sizes, nms_type=NMSTypes.BATCHED_NMS)

    def native_torch_on_cpu(self, model_archs=None, batch_sizes=None):
        model_archs, batch_sizes = self._valid_archs_and_batchs(model_archs, batch_sizes)

        return self._benchmark_internal(model_archs=model_archs, batch_sizes=batch_sizes, nms_device='cpu',
                                        nms_type=NMSTypes.NO_NMS)

    def native_torch_on_gpu(self, model_archs=None, batch_sizes=None):
        model_archs, batch_sizes = self._valid_archs_and_batchs(model_archs, batch_sizes)

        return self._benchmark_internal(model_archs=model_archs, batch_sizes=batch_sizes, nms_device='gpu',
                                        nms_type=NMSTypes.NO_NMS)

    def path_for_model(self, model_arch, nms_type):
        return os.path.join(self._model_directory_base, model_arch.value, f'{nms_type.value}.engine')

    @staticmethod
    def persist_result_dict_to_csv(benchmark_type, results_dict, export_path, append_to_existing=True):
        df_dict = {
            'benchmark_type': [],
            'model': [],
            'latency': [],
            'throughput': [],
            'date': [],
            'batch_size': []
        }

        for arch in results_dict:
            for bs in results_dict[arch]:
                df_dict['latency'].append(results_dict[arch][bs]['latency'])
                df_dict['throughput'].append(results_dict[arch][bs]['throughput'])
                df_dict['date'].append(results_dict[arch][bs]['date'])
                df_dict['benchmark_type'].append(benchmark_type)
                df_dict['model'].append(arch.value)
                df_dict['batch_size'].append(bs)

        new_results_df = pd.DataFrame(data=df_dict)
        if append_to_existing and os.path.exists(export_path):
            old_results_df = pd.read_csv(export_path)
            new_results_df = new_results_df.append(old_results_df, ignore_index=True)

        new_results_df.to_csv(export_path, index=False)

    @staticmethod
    def get_coco_data_loader(loaded_model, batch_size):
        import yaml
        from pathlib import Path

        dataset_params = yaml.safe_load(Path(
            '/home/naveassaf/Workspace/super-gradients/src/super_gradients/recipes/dataset_params/coco_detection_yolox_dataset_params.yaml').read_text())

        dataset_params['val_collate_fn'] = DetectionCollateFN()
        dataset_params['train_collate_fn'] = DetectionCollateFN()
        dataset_params['batch_size'] = batch_size
        dataset_params['val_batch_size'] = batch_size
        dataset_params['val_image_size'] = loaded_model.input_dims[0][-1]
        dataset_params['train_image_size'] = loaded_model.input_dims[0][-1]

        _, val_loader, _, _ = CocoDetectionDatasetInterfaceV2(dataset_params).get_data_loaders()

        return val_loader

    def _valid_archs_and_batchs(self, model_archs, batch_sizes):
        # If not specified, benchmark all model architectures and all batch sizes
        model_archs = model_archs or list(ModelArchs)
        batch_sizes = batch_sizes or {*THROUGHPUT_BATCH_SIZES, *LATENCY_BATCH_SIZES}

        return model_archs, batch_sizes

    @staticmethod
    def cleanup():
        gc.collect()
        del gc.garbage[:]
        gc.collect()


if __name__ == '__main__':
    import yaml
    from pathlib import Path

    dataset_params = yaml.safe_load(Path(
        '/home/naveassaf/Workspace/super-gradients/src/super_gradients/recipes/dataset_params/coco_detection_yolox_dataset_params.yaml').read_text())

    dataset_params['val_collate_fn'] = DetectionCollateFN()
    dataset_params['train_collate_fn'] = DetectionCollateFN()
    dataset_params['batch_size'] = 1
    dataset_params['val_batch_size'] = 1
    dataset_params['val_image_size'] = 640
    dataset_params['train_image_size'] = 640

    ds_interface = CocoDetectionDatasetInterfaceV2(dataset_params)

    yolox_s = SgModel(experiment_name='YoloX_S')
    yolox_s.connect_dataset_interface(ds_interface)
    yolox_s.build_model()

    # ------------- CONSTANTS ------------- #
    benchmarker = NMSBenchmarker('/home/naveassaf/Desktop/NMS_Benchmarks/A4000')
    results_path = '/home/naveassaf/Desktop/NMS_Benchmarks/results_A4000.csv'

    # ------------- BENCHMARK ------------- #
    # results_dict = benchmarker.no_nms_first_100_benchmarks()
    # benchmarker.persist_result_dict_to_csv('first_100',
    #                                        results_dict=results_dict,
    #                                        export_path=results_path,
    #                                        append_to_existing=True)
    # results_dict = benchmarker.trt_batched_nms()
    # benchmarker.persist_result_dict_to_csv('trt_batched',
    #                                        results_dict=results_dict,
    #                                        export_path=results_path,
    #                                        append_to_existing=True)
    results_dict = benchmarker.native_torch_on_cpu()
    benchmarker.persist_result_dict_to_csv('torch_cpu',
                                           results_dict=results_dict,
                                           export_path=results_path,
                                           append_to_existing=True)
    results_dict = benchmarker.native_torch_on_gpu()
    benchmarker.persist_result_dict_to_csv('torch_gpu',
                                           results_dict=results_dict,
                                           export_path=results_path,
                                           append_to_existing=True)
