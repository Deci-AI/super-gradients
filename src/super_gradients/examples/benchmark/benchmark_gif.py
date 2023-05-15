import torch
from typing import Callable
from super_gradients.common.object_names import Models
from super_gradients.training import models
from super_gradients.training.utils.media.video import load_video


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

    model = models.get(Models.YOLO_NAS_L, pretrained_weights="coco").cuda()

    video, _fps = load_video("../../../../documentation/source/images/examples/pose_elephant_flip.gif")

    predict_video = lambda: model.predict(video)
    latency = benchmark_model(predict_video, benchmark_cycles=15, warmup_cycles=5)
    print("Benchmark full video: ", latency)

    def predict_frame_by_frame():
        for frame in video:
            _ = model.predict(frame)

    latency = benchmark_model(predict_frame_by_frame, benchmark_cycles=15, warmup_cycles=5)
    print("Benchmark frame-by-frame: ", latency)
