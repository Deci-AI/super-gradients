import torch
from pandas import np
from torch.nn import Identity
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.transforms_factory import TransformsFactory


class ConvertableCompletePipelineModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, pre_process: torch.nn.Module = None, post_process: torch.nn.Module = None, **prep_model_for_conversion_kwargs):
        super(ConvertableCompletePipelineModel, self).__init__()
        model.eval()
        pre_process = pre_process or Identity()
        post_process = post_process or Identity()
        if hasattr(model, "prep_model_for_conversion"):
            model.prep_model_for_conversion(**prep_model_for_conversion_kwargs)
        self.model = model
        self.pre_process = pre_process
        self.post_process = post_process

    def forward(self, x):
        return self.post_process(self.model(self.pre_process(x)))


@resolve_param("pre_procss", TransformsFactory())
@resolve_param("post_procss", TransformsFactory())
def convert_to_onnx(
    model: torch.nn.Module,
    out_path: str,
    input_shape: tuple,
    pre_process: torch.nn.Module = None,
    post_process: torch.nn.Module = None,
    prep_model_for_conversion_kwargs=None,
    torch_onnx_export_kwargs=None,
):
    torch_onnx_export_kwargs = torch_onnx_export_kwargs or dict()
    prep_model_for_conversion_kwargs = prep_model_for_conversion_kwargs or dict()
    onnx_input = torch.Tensor(np.zeros([1, *input_shape]))
    if not out_path.endswith(".onnx"):
        out_path = out_path + ".onnx"
    complete_model = ConvertableCompletePipelineModel(model, pre_process, post_process, **prep_model_for_conversion_kwargs)

    torch.onnx.export(model=complete_model, args=onnx_input, f=out_path, **torch_onnx_export_kwargs)
    return out_path
