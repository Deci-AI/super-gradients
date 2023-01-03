from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from pandas import np
from torch.nn import Identity

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.training.utils.checkpoint_utils import get_checkpoints_dir_path
from super_gradients.training.utils.hydra_utils import load_experiment_cfg

logger = get_logger(__name__)


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


def prepare_conversion_cfgs(cfg: DictConfig):
    """
    Builds the cfg (i.e conversion_params) and experiment_cfg (i.e recipe config according to cfg.experiment_name)
     to be used by convert_recipe_example

    :param cfg: DictConfig, converion_params config
    :return: cfg, experiment_cfg
    """
    cfg = hydra.utils.instantiate(cfg)
    # CREATE THE EXPERIMENT CFG
    experiment_cfg = load_experiment_cfg(cfg.experiment_name, cfg.ckpt_root_dir)
    hydra.utils.instantiate(experiment_cfg)
    if cfg.checkpoint_path is None:
        logger.info(
            "checkpoint_params.checkpoint_path was not provided, so the model will be converted using weights from "
            "checkpoints_dir/training_hyperparams.ckpt_name "
        )
        checkpoints_dir = Path(get_checkpoints_dir_path(experiment_name=cfg.experiment_name, ckpt_root_dir=cfg.ckpt_root_dir))
        cfg.checkpoint_path = str(checkpoints_dir / cfg.ckpt_name)
    cfg.out_path = cfg.out_path or cfg.checkpoint_path.replace(".ckpt", ".onnx")
    logger.info(f"Exporting checkpoint: {cfg.checkpoint_path} to ONNX.")
    return cfg, experiment_cfg
