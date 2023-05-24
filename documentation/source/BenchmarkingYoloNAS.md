# Benchmarking YoloNAS
## Introduction:

YoloNAS is a leading object detection architecture that combines accuracy and efficiency. By employing quantization aware training (QAT), YoloNAS models can be optimized for resource-constrained devices.
However, to fully tap into its potential, knowing how to export the quantized model to the INT8 TensorRT (TRT) engine is crucial.
In this blog, we emphasize the significance of this step and provide a concise guide to efficiently exporting a quantized YoloNAS model to the INT8 TRT engine.
By doing so, we learn how to properly benchmark YoloNAS and understand its full potential.

## Step 1: Export YoloNAS to ONNX

First step, is to properly export our YoloNAS model to ONNX. There are two actions that must be taken before we export our model to onnx:
1. We need to replace our layers with "fake quantized" ones - this happens when we perform post-training quantization or quantization aware training with SG.
So nothing to worry about if you performed PTQ/QAT with SG and hold your newly exported ONNX checkpoint.
2. We must call model.prep_model_for_conversion - this is essential as YoloNAS incorporates QARepVGG blocks. Without this call, the RepVGG branches will not be fused and our model's speed will decrease significantly!
Again, nothing to worry about if you have quantized your model with PTQ/QAT with SG, as this is done under the hood before exporting the ONNX checkpoints.
   
There plenty of guides on how to performs PTQ/QAT with SG:
- [Quantization-aware fine-tuning YoloNAS on custom dataset notebook](https://colab.research.google.com/drive/1yHrHkUR1X2u2FjjvNMfUbSXTkUul6o1P?usp=sharing)
- [QA/PTQ YoloNAS with configuration files](https://github.com/Deci-AI/super-gradients/blob/master/documentation/source/qat_ptq_yolo_nas.md)
- [QA/PTQ](https://github.com/Deci-AI/super-gradients/blob/master/documentation/source/ptq_qat.md)

Suppose we ran PTQ/QAT, then our PTQ/QAT checkpoints have been exported to our checkpoints directory.
If we plug them into [netron.app](https://netron.app), we can see that new blocks that were not a part of the original network were introduced: the **Quantize/Dequantize** layers - 

<div>
<img src="images/qdq_yolonas_netron.png" width="750">
</div>
This is expected and a good way to verify that our model is ready to be converted to Int8 using Nvidia's TesnorRT.

## Step 2: Create TRT Engine
First, please make sure to [install Nvidia's TensorRT](https://developer.nvidia.com/tensorrt-getting-started).
TensorRT version >= 8.4 is required.
We can now use these ONNX files to deploy our newly trained YoloNAS models to production. When building the TRT engine it is important specify that we convert to Int8 (the fake quantized layers in our models will be adapted accordingly),
this can be done by running: `trtexec --fp16 --int8 --onnx=your_yolonas_qat_model.onnx`.
## Step 3: View Model Benchmark Results

Once running `trtexec --fp16 --int8 --onnx=your_yolonas_qat_model.onnx.` your screen will look somewhat similar to the screenshot below: 
<div>
<img src="images/trtexec.png" width="750">
</div>

- The actual throughput and latency of your model are in blue. This tells you how your model is actually performing.
- Note that trtexec shows the minimum, maximum, mean, and median values. Large differences between these values can indicate that your measurements are noisy. It might be that some other process is using the GPU while benchmarking or that the GPU is not properly cooled.
- The end-to-end latency is marked in yellow. This includes the time it takes to prepare the input and pass it to the GPU, as well as the GPU compute time and the time it takes to move the output from the GPU back to the host (for the entire batch size). If you plan on running batches one by one synchronously, this is the time that affects you. But if you are using an async inference engine, you will be affected by the numbers in blue.
- High H2D (H=Host=CPU; D=Device=GPU) values indicate your input size has a crucial effect on the performance. Consider resizing the input in advance (not on the GPU), or maybe test with different batch sizes to find the optimal setting.
- High D2H values indicate your output might be too big. You can consider a task-specific method to reduce it. i.e., use top-k at the end of your detection model to limit the number of boxes coming out, or use a softmax layer at the end of your segmentation model to change the output representation to one with smaller dimensions.
