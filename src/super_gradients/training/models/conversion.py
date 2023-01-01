import torch
from pandas import np
from torch.nn import Identity
from torch.nn import Sequential


def build_complete_pipeline_convertable_model(
    model: torch.nn.Module, pre_process: torch.nn.Module = None, post_process: torch.nn.Module = None, **prep_model_for_conversion_kwargs
):
    model.eval()
    pre_process = pre_process or Identity()
    post_process = post_process or Identity()
    if hasattr(model, "prep_model_for_conversion"):
        model.prep_model_for_conversion(**prep_model_for_conversion_kwargs)

    return Sequential([pre_process, model, post_process])


def convert_to_onnx(
    model: torch.nn.Module,
    out_path: str,
    input_shape: tuple,
    pre_process: torch.nn.Module = None,
    post_process: torch.nn.Module = None,
    prep_model_for_conversion_kwargs=None,
    **kwargs,
):

    prep_model_for_conversion_kwargs = prep_model_for_conversion_kwargs or dict()
    onnx_input = torch.Tensor(np.zeros([1, *input_shape]))
    if not out_path.endswith(".onnx"):
        out_path = out_path + ".onnx"
    complete_model = build_complete_pipeline_convertable_model(model, pre_process, post_process, **prep_model_for_conversion_kwargs)
    torch.onnx.export(complete_model, args=onnx_input, f=out_path, **kwargs)
    return out_path
