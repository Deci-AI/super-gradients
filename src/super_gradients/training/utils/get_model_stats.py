import numpy as np
import GPUtil
from typing import Union
from collections import OrderedDict
import torch
import torch.nn as nn
from super_gradients.training.utils.utils import Timer


def get_model_stats(
    model: nn.Module,
    input_dims: Union[list, tuple],
    high_verbosity: bool = True,
    batch_size: int = 1,
    device: str = "cuda",  # noqa: C901
    dtypes=None,
    iterations: int = 100,
):
    """
    return the model summary as a string
    The block(type) column represents the lines (layers) above
        :param dtypes:          The input types (list of inputs types)
        :param high_verbosity:  prints layer by layer information
    """
    dtypes = dtypes or [torch.FloatTensor] * len(input_dims)

    def register_hook(module):
        """
        add a hook (all the desirable actions) for every layer that is not nn.Sequential/nn.ModuleList
        """

        def hook(module, input, output):

            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = f"{class_name}-{module_idx + 1}"
            summary[m_key] = OrderedDict()

            # block_name refers to all layers that contains other layers
            if len(module._modules) != 0:
                summary[m_key]["block_name"] = class_name

            summary[m_key]["inference_time"] = np.round(timer.stop(), 3)
            timer.start()

            summary[m_key]["gpu_occupation"] = (round(torch.cuda.memory_allocated(0) / 1024**3, 2), "GB") if torch.cuda.is_available() else [0]
            summary[m_key]["gpu_cached_memory"] = (round(torch.cuda.memory_reserved(0) / 1024**3, 2), "GB") if torch.cuda.is_available() else [0]

            summary[m_key]["input_shape"], summary[m_key]["output_shape"] = get_input_output_shapes(batch_size=batch_size, input_dims=input, output_dims=output)

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_dims, tuple):
        input_dims = [input_dims]

    x = [torch.rand(batch_size, *input_dim).type(dtype).to(device=device) for input_dim, dtype in zip(input_dims, dtypes)]

    summary_list = []
    with torch.no_grad():
        for i in range(iterations + 10):
            # create properties
            summary = OrderedDict()
            hooks = []

            # register hook
            model.apply(register_hook)

            timer = Timer(device=device)
            timer.start()
            # make a forward pass
            model(*x)

            # remove these hooks
            for h in hooks:
                h.remove()

            # we start counting from the 10th iteration for warmup
            if i >= 10:
                summary_list.append(summary)

    summary = _average_inference_time(summary_list=summary_list, summary=summary, divisor=iterations)

    return _convert_summary_dict_to_string(summary=summary, high_verbosity=high_verbosity, input_dims=input_dims, batch_size=batch_size, device=device)


def _average_inference_time(summary_list: list, summary: OrderedDict, divisor: int = 100):
    inference_time_dict = {}
    for idx, sum in enumerate(summary_list):
        for key, _ in sum.items():
            if idx == 0:
                inference_time_dict[key] = sum[key]["inference_time"]
            else:
                inference_time_dict[key] += sum[key]["inference_time"]

    for key, _ in summary.items():
        summary[key]["inference_time"] = np.round(inference_time_dict[key] / divisor, 3)

    return summary


def get_input_output_shapes(batch_size: int, input_dims: Union[list, tuple], output_dims: Union[list, tuple]):
    """
    Returns input/output shapes for single/multiple input/s output/s
    """
    if isinstance(input_dims[0], list):
        input_shape = [i.size() for i in input_dims[0] if i is not None]
    else:
        input_shape = list(input_dims[0].size())
    input_shape[0] = batch_size
    if isinstance(output_dims, (list, tuple)):
        output_shape = [[-1] + list(o.size())[1:] for o in output_dims if o is not None]
    else:
        output_shape = list(output_dims.size())
        output_shape[0] = batch_size
    return input_shape, output_shape


def _convert_summary_dict_to_string(summary: dict, high_verbosity: bool, input_dims: Union[list, tuple], batch_size: int, device: str):
    """
    Takes summary dict and Returns summary string
    """
    summary_str = ""
    total_params = 0
    total_output = 0
    trainable_params = 0
    if high_verbosity:
        summary_str += f"{'-' * 200}\n"
        line_new = (
            f'{"block (type)":>20} {"Layer (type)":>20} {"Output Shape":>63} {"Param #":>15} '
            f'{"inference time[ms]":>25} {"gpu_cached_memory[GB]":>25} {"gpu_occupation[GB]":>25}'
        )
        summary_str += f"{line_new}\n"
        summary_str += f"{'=' * 200}\n"
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20} {:>20}  {:>63} {:>15} {:>25} {:>25} {:>25}".format(
            str(summary[layer]["block_name"]) if "block_name" in summary[layer].keys() else "",
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
            "{0:,}".format(summary[layer]["inference_time"]),
            "{0:,}".format(summary[layer]["gpu_cached_memory"][0]),
            "{0:,}".format(summary[layer]["gpu_occupation"][0]),
        )
        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"]:
                trainable_params += summary[layer]["nb_params"]
        if high_verbosity:
            summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_dims, ())) * batch_size * 4.0 / (1024**2.0))
    total_output_size = abs(2.0 * total_output * 4.0 / (1024**2.0))  # x2 for gradients
    total_params_size = abs(total_params * 4.0 / (1024**2.0))
    total_size = total_params_size + total_output_size + total_input_size

    gpus = GPUtil.getGPUs()
    gpu_memory_utilization = [gpu.memoryUtil * 100 for gpu in gpus]

    summary_str += (
        f"{'=' * 200}\n"
        f"Total params: {total_params:,}\n"
        f"Trainable params: {trainable_params:,}\n"
        f"Non-trainable params: {total_params - trainable_params:,}\n"
        f"{'-' * 200}\n"
        f"Input size (MB): {total_input_size:.2f}\n"
        f"Forward/backward pass size (MB): {total_output_size:.2f}\n"
        f"Params size (MB): {total_params_size}\n"
        f"Estimated Total Size (MB): {total_size}\n"
    )

    summary_str += str(["Memory Footprint (percentage): %0.2f" % gpu_memory_utilization[i] for i in range(4)]) + "\n"
    summary_str += f"{'-' * 200}\n" if device == "cuda" else f"{'-' * 200}\n"

    return summary_str
