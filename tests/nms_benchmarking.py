import infery
import pandas as pd
import numpy as np
from enum import Enum
import time
from datetime import datetime
import os
import sys
import gc
import torch

try:
    from super_gradients.training.models.detection_models.yolo_base import YoloPostPredictionCallback
    from super_gradients.training.utils.detection_utils import NMS_Type
    SG_IMPORTED_SUCCESSFULLY = True
except ImportError as e:
    print(f'\n\nWARNING: Failed to import super-gradients - {e}\n\n')
    SG_IMPORTED_SUCCESSFULLY = False

BATCH_SIZES = [1, 8, 16, 32]
WARMUP_REPETITIONS = 100
NMS_CONFIDENCE_THRESHOLD = 0.5
NMS_IOU = 0.65


class ModelArchs(Enum):
    YOLOX_N = 'yolox_n'
    YOLOX_T = 'yolox_t'
    YOLOX_S = 'yolox_s'


class NMSTypes(Enum):
    FIRST_100 = 'first_100'
    NO_NMS = 'no_nms'
    BATCHED_NMS = 'batched_nms'


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

    def _benchmark_internal(self, model_archs, batch_sizes,  nms_type, nms_device='cpu', torch_nms='iterative'):
        self.cleanup()
        results_dict = {}

        # If NO_NMS is compiled, use the Torch NMS module with the passed NMS device and type
        benchmark_method = self._benchmark_helper_in_model_nms
        nms_callback = None
        if nms_type == NMSTypes.NO_NMS:
            benchmark_method = self._benchmark_helper_torch_cpu if nms_device == 'cpu' else \
                self._benchmark_helper_torch_gpu
            nms_callback = YoloPostPredictionCallback(conf=NMS_CONFIDENCE_THRESHOLD,
                                                      iou=NMS_IOU, nms_type=NMS_Type(torch_nms))


        for arch in model_archs:
            # Prep model and results bookkeeping
            results_dict[arch] = {}
            model_path = self.path_for_model(model_arch=arch, nms_type=nms_type)
            loaded_model = infery.load(model_path=model_path, framework_type='trt')

            # Start benchmarking different batch sizes
            for bs in batch_sizes:
                print(f'{arch} --- {bs}')
                data_loader = self.get_coco_data_loader()
                results_dict[arch][bs] = {}

                # Warmup
                dummy_input = np.random.rand(bs, *(loaded_model.input_dims[0])).astype(np.float32)
                for _ in range(WARMUP_REPETITIONS):
                    x = loaded_model.predict(dummy_input, output_device=nms_device)

                # Benchmark
                times = []

                for x in data_loader:
                    x = (x[0].numpy())

                    # Bug with setting BS of dataloader dynamically
                    for i in range(0, 64, bs):
                        y = x[i:i+bs, :, :, :]

                        # We're loaded and the input is converted - now time to benchmark (E2E - i.e., CPU -> CPU)
                        benchmark_method(y, loaded_model, times, nms_callback)

                results_dict[arch][bs]['latency'] = sum(times)/len(times)
                results_dict[arch][bs]['throughput'] = len(times)*bs/sum(times)
                results_dict[arch][bs]['date'] = datetime.now()

        self.cleanup()
        return results_dict

    def _benchmark_helper_torch_gpu(self, x, loaded_model, times, nms_callback):
        start = time.perf_counter()
        x = loaded_model.predict(x, output_device='gpu')
        x = nms_callback(x[-1])
        times.append(time.perf_counter() - start)

    def _benchmark_helper_torch_cpu(self, x, loaded_model, times, nms_callback):
        start = time.perf_counter()
        x = loaded_model.predict(x)
        x = nms_callback(torch.from_numpy(x[-1]))
        times.append(time.perf_counter() - start)

    def _benchmark_helper_in_model_nms(self, x, loaded_model, times, nms_callback):
        start = time.perf_counter()
        x = loaded_model.predict(x)
        times.append(time.perf_counter() - start)

    def no_nms_first_100_benchmarks(self, model_archs=None, batch_sizes=None):
        model_archs, batch_sizes = self._valid_archs_and_batchs(model_archs, batch_sizes)

        return self._benchmark_internal(model_archs=model_archs, batch_sizes=batch_sizes, nms_type=NMSTypes.FIRST_100)

    def trt_batched_nms(self, model_archs=None, batch_sizes=None):
        model_archs, batch_sizes = self._valid_archs_and_batchs(model_archs, batch_sizes)

        return self._benchmark_internal(model_archs=model_archs, batch_sizes=batch_sizes, nms_type=NMSTypes.BATCHED_NMS)

    def native_torch_on_cpu(self, model_archs=None, batch_sizes=None, torch_nms='iterative'):
        model_archs, batch_sizes = self._valid_archs_and_batchs(model_archs, batch_sizes)

        return self._benchmark_internal(model_archs=model_archs, batch_sizes=batch_sizes, nms_device='cpu',
                                        nms_type=NMSTypes.NO_NMS)

    def native_torch_on_gpu(self, model_archs=None, batch_sizes=None, torch_nms='iterative'):
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
    def get_coco_data_loader():
        from super_gradients.training.dataloaders.dataloader_factory import coco2017_val_yolox

        return coco2017_val_yolox()

    def _valid_archs_and_batchs(self, model_archs, batch_sizes):
        # If not specified, benchmark all model architectures and all batch sizes
        model_archs = model_archs or list(ModelArchs)
        batch_sizes = batch_sizes or BATCH_SIZES

        return model_archs, batch_sizes

    @staticmethod
    def cleanup():
        gc.collect()
        del gc.garbage[:]
        gc.collect()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('USAGE: [PATH_TO_MODEL_DIR] [PATH_TO_RESULTS_FILE]')
        exit(1)

    # ------------- CONSTANTS ------------- #
    benchmarker = NMSBenchmarker(sys.argv[1])
    results_path = sys.argv[2]

    # ------------- BENCHMARK ------------- #
    results_dict = benchmarker.no_nms_first_100_benchmarks()
    benchmarker.persist_result_dict_to_csv('first_100',
                                           results_dict=results_dict,
                                           export_path=results_path,
                                           append_to_existing=True)
    results_dict = benchmarker.trt_batched_nms()
    benchmarker.persist_result_dict_to_csv('trt_batched',
                                           results_dict=results_dict,
                                           export_path=results_path,
                                           append_to_existing=True)
    if SG_IMPORTED_SUCCESSFULLY:
        results_dict = benchmarker.native_torch_on_cpu(torch_nms=NMS_Type.ITERATIVE)
        benchmarker.persist_result_dict_to_csv('torch_cpu',
                                               results_dict=results_dict,
                                               export_path=results_path,
                                               append_to_existing=True)
        results_dict = benchmarker.native_torch_on_gpu(torch_nms=NMS_Type.ITERATIVE)
        benchmarker.persist_result_dict_to_csv('torch_gpu',
                                               results_dict=results_dict,
                                               export_path=results_path,
                                               append_to_existing=True)

        # TODO: SUPPORT MATRIX NMS? CURRENTLY SEEMS BUGGY.
        # TODO: results_dict = benchmarker.native_torch_on_gpu(torch_nms=NMS_Type.MATRIX)
