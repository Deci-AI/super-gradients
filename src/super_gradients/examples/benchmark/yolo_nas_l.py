import torch
from typing import Callable
from super_gradients.common.object_names import Models
from super_gradients.training import models


def benchmark_model(action: Callable, benchmark_cycles: int, warmup_cycles: int):
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.no_grad():

        for _ in range(warmup_cycles):
            action()

        total_time = 0
        for i in range(benchmark_cycles):
            starter.record()
            action()
            ender.record()
            torch.cuda.synchronize()
            total_time += starter.elapsed_time(ender)

    latency = total_time / benchmark_cycles
    return latency


if __name__ == "__main__":
    model = models.get(Models.YOLO_NAS_L, pretrained_weights="coco")
    model = model.to("cuda")
    model.prep_model_for_conversion(input_size=(640, 640))

    input_t = torch.randn(1, 3, 640, 640).to("cuda")

    func = lambda: model(input_t)

    latency = benchmark_model(func, benchmark_cycles=1000, warmup_cycles=50)
    print(latency)
