from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
import numpy as np
from torch.nn import Identity

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.training import models
from super_gradients.training.utils.checkpoint_utils import get_checkpoints_dir_path
from super_gradients.training.utils.hydra_utils import load_experiment_cfg
from super_gradients.training.utils.sg_trainer_utils import parse_args
import os
import pathlib

logger = get_logger(__name__)


class ConvertableCompletePipelineModel(torch.nn.Module):
    """
    Exportable nn.Module that wraps the model, preprocessing and postprocessing.
    Args:
        model: torch.nn.Module, the main model. takes input from pre_process' output, and feeds pre_process.
        pre_process: torch.nn.Module, preprocessing module, its output will be model's input. When none (default), set to Identity().
        pre_process: torch.nn.Module, postprocessing module, its output is the final output. When none (default), set to Identity().
        **prep_model_for_conversion_kwargs: for SgModules- args to be passed to model.prep_model_for_conversion
            prior to torch.onnx.export call.
    """

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


@resolve_param("pre_process", TransformsFactory())
@resolve_param("post_process", TransformsFactory())
def convert_to_onnx(
    model: torch.nn.Module,
    out_path: str,
    input_shape: tuple,
    pre_process: torch.nn.Module = None,
    post_process: torch.nn.Module = None,
    prep_model_for_conversion_kwargs=None,
    torch_onnx_export_kwargs=None,
):
    """
    Exports model to ONNX.

    :param model: torch.nn.Module, model to export to ONNX.
    :param out_path: str, destination path for the .onnx file.
    :param input_shape: tuple, input shape, excluding batch_size (i.e (3, 224, 224)).
    :param pre_process: torch.nn.Module, preprocessing pipeline, will be resolved by TransformsFactory()
    :param post_process: torch.nn.Module, postprocessing pipeline, will be resolved by TransformsFactory()
    :param prep_model_for_conversion_kwargs: dict, for SgModules- args to be passed to model.prep_model_for_conversion
     prior to torch.onnx.export call.
    :param torch_onnx_export_kwargs: kwargs (EXCLUDING: FIRST 3 KWARGS- MODEL, F, ARGS). to be unpacked in torch.onnx.export call

    :return: out_path
    """
    if not os.path.isdir(pathlib.Path(out_path).parent.resolve()):
        raise FileNotFoundError(f"Could not find destination directory {out_path} for the ONNX file.")
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
    cfg.out_path = cfg.out_path or cfg.checkpoint_path.replace(".pth", ".onnx")
    logger.info(f"Exporting checkpoint: {cfg.checkpoint_path} to ONNX.")
    return cfg, experiment_cfg


def convert_from_config(cfg: DictConfig) -> str:
    """
    Exports model according to cfg.

    See:
     super_gradients/recipes/conversion_params/default_conversion_params.yaml for the full cfg content documentation,
     and super_gradients/examples/convert_recipe_example/convert_recipe_example.py for usage.
    :param cfg:
    :return: out_path, the path of the saved .onnx file.
    """
    cfg, experiment_cfg = prepare_conversion_cfgs(cfg)
    model = models.get(
        model_name=experiment_cfg.architecture,
        num_classes=experiment_cfg.arch_params.num_classes,
        arch_params=experiment_cfg.arch_params,
        strict_load=cfg.strict_load,
        checkpoint_path=cfg.checkpoint_path,
    )
    cfg = parse_args(cfg, models.convert_to_onnx)
    out_path = models.convert_to_onnx(model=model, **cfg)
    logger.info(f"Successfully exported model at {out_path}")
    return out_path
