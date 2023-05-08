import os
from typing import Union, Tuple

import copy
import hydra
import torch.cuda
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch import nn

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.environment.device_utils import device_config
from super_gradients.training import utils as core_utils, models, dataloaders
from super_gradients.training.sg_trainer import Trainer
from super_gradients.training.utils import get_param
from super_gradients.training.utils.distributed_training_utils import setup_device
from super_gradients.modules.repvgg_block import fuse_repvgg_blocks_residual_branches

logger = get_logger(__name__)
try:
    from super_gradients.training.utils.quantization.calibrator import QuantizationCalibrator
    from super_gradients.training.utils.quantization.export import export_quantized_module_to_onnx
    from super_gradients.training.utils.quantization.selective_quantization_utils import SelectiveQuantizer

    _imported_pytorch_quantization_failure = None

except (ImportError, NameError, ModuleNotFoundError) as import_err:
    logger.debug("Failed to import pytorch_quantization:")
    logger.debug(import_err)
    _imported_pytorch_quantization_failure = import_err


class QATTrainer(Trainer):
    @classmethod
    def train_from_config(cls, cfg: Union[DictConfig, dict]) -> Tuple[nn.Module, Tuple]:
        """
        Perform quantization aware training (QAT) according to a recipe configuration.

        This method will instantiate all the objects specified in the recipe, build and quantize the model,
        and calibrate the quantized model. The resulting quantized model and the output of the trainer.train()
        method will be returned.

        The quantized model will be exported to ONNX along with other checkpoints.

        :param cfg: The parsed DictConfig object from yaml recipe files or a dictionary.
        :return: A tuple containing the quantized model and the output of trainer.train() method.
        :rtype: Tuple[nn.Module, Tuple]

        :raises ValueError: If the recipe does not have the required key `quantization_params` or
        `checkpoint_params.checkpoint_path` in it.
        :raises NotImplementedError: If the recipe requests multiple GPUs or num_gpus is not equal to 1.
        :raises ImportError: If pytorch-quantization import was unsuccessful

        """
        if _imported_pytorch_quantization_failure is not None:
            raise _imported_pytorch_quantization_failure

        # INSTANTIATE ALL OBJECTS IN CFG
        cfg = hydra.utils.instantiate(cfg)

        # TRIGGER CFG MODIFYING CALLBACKS
        cfg = cls._trigger_cfg_modifying_callbacks(cfg)

        if "quantization_params" not in cfg:
            raise ValueError("Your recipe does not have quantization_params. Add them to use QAT.")

        if "checkpoint_path" not in cfg.checkpoint_params:
            raise ValueError("Starting checkpoint is a must for QAT finetuning.")

        num_gpus = core_utils.get_param(cfg, "num_gpus")
        multi_gpu = core_utils.get_param(cfg, "multi_gpu")
        device = core_utils.get_param(cfg, "device")
        if num_gpus != 1:
            raise NotImplementedError(
                f"Recipe requests multi_gpu={cfg.multi_gpu} and num_gpus={cfg.num_gpus}. QAT is proven to work correctly only with multi_gpu=OFF and num_gpus=1"
            )

        setup_device(device=device, multi_gpu=multi_gpu, num_gpus=num_gpus)

        # INSTANTIATE DATA LOADERS
        train_dataloader = dataloaders.get(
            name=get_param(cfg, "train_dataloader"),
            dataset_params=copy.deepcopy(cfg.dataset_params.train_dataset_params),
            dataloader_params=copy.deepcopy(cfg.dataset_params.train_dataloader_params),
        )

        val_dataloader = dataloaders.get(
            name=get_param(cfg, "val_dataloader"),
            dataset_params=copy.deepcopy(cfg.dataset_params.val_dataset_params),
            dataloader_params=copy.deepcopy(cfg.dataset_params.val_dataloader_params),
        )

        if "calib_dataloader" in cfg:
            calib_dataloader_name = get_param(cfg, "calib_dataloader")
            calib_dataloader_params = copy.deepcopy(cfg.dataset_params.calib_dataloader_params)
            calib_dataset_params = copy.deepcopy(cfg.dataset_params.calib_dataset_params)
        else:
            calib_dataloader_name = get_param(cfg, "train_dataloader")
            calib_dataloader_params = copy.deepcopy(cfg.dataset_params.train_dataloader_params)
            calib_dataset_params = copy.deepcopy(cfg.dataset_params.train_dataset_params)

            # if we use whole dataloader for calibration, don't shuffle it
            # HistogramCalibrator collection routine is sensitive to order of batches and produces slightly different results
            # if we use several batches, we don't want it to be from one class if it's sequential in dataloader
            # model is in eval mode, so BNs will not be affected
            calib_dataloader_params.shuffle = cfg.quantization_params.calib_params.num_calib_batches is not None
            # we don't need training transforms during calibration, distribution of activations will be skewed
            calib_dataset_params.transforms = cfg.dataset_params.val_dataset_params.transforms

        calib_dataloader = dataloaders.get(
            name=calib_dataloader_name,
            dataset_params=calib_dataset_params,
            dataloader_params=calib_dataloader_params,
        )

        # BUILD MODEL
        model = models.get(
            model_name=cfg.arch_params.get("model_name", None) or cfg.architecture,
            num_classes=cfg.get("num_classes", None) or cfg.arch_params.num_classes,
            arch_params=cfg.arch_params,
            strict_load=cfg.checkpoint_params.strict_load,
            pretrained_weights=cfg.checkpoint_params.pretrained_weights,
            checkpoint_path=cfg.checkpoint_params.checkpoint_path,
            load_backbone=False,
        )
        model.to(device_config.device)

        # QUANTIZE MODEL
        model.eval()
        fuse_repvgg_blocks_residual_branches(model)

        q_util = SelectiveQuantizer(
            default_quant_modules_calibrator_weights=cfg.quantization_params.selective_quantizer_params.calibrator_w,
            default_quant_modules_calibrator_inputs=cfg.quantization_params.selective_quantizer_params.calibrator_i,
            default_per_channel_quant_weights=cfg.quantization_params.selective_quantizer_params.per_channel,
            default_learn_amax=cfg.quantization_params.selective_quantizer_params.learn_amax,
            verbose=cfg.quantization_params.calib_params.verbose,
        )
        q_util.register_skip_quantization(layer_names=cfg.quantization_params.selective_quantizer_params.skip_modules)
        q_util.quantize_module(model)

        # CALIBRATE MODEL
        logger.info("Calibrating model...")
        calibrator = QuantizationCalibrator(
            verbose=cfg.quantization_params.calib_params.verbose,
            torch_hist=True,
        )
        calibrator.calibrate_model(
            model,
            method=cfg.quantization_params.calib_params.histogram_calib_method,
            calib_data_loader=calib_dataloader,
            num_calib_batches=cfg.quantization_params.calib_params.num_calib_batches or len(train_dataloader),
            percentile=get_param(cfg.quantization_params.calib_params, "percentile", 99.99),
        )
        calibrator.reset_calibrators(model)  # release memory taken by calibrators

        # VALIDATE PTQ MODEL AND PRINT SUMMARY
        logger.info("Validating PTQ model...")
        trainer = Trainer(experiment_name=cfg.experiment_name, ckpt_root_dir=get_param(cfg, "ckpt_root_dir", default_val=None))
        valid_metrics_dict = trainer.test(model=model, test_loader=val_dataloader, test_metrics_list=cfg.training_hyperparams.valid_metrics_list)
        results = ["PTQ Model Validation Results"]
        results += [f"   - {metric:10}: {value}" for metric, value in valid_metrics_dict.items()]
        logger.info("\n".join(results))

        # TRAIN
        if cfg.quantization_params.ptq_only:
            logger.info("cfg.quantization_params.ptq_only=True. Performing PTQ only!")
            suffix = "ptq"
            res = None
        else:
            model.train()
            recipe_logged_cfg = {"recipe_config": OmegaConf.to_container(cfg, resolve=True)}
            trainer = Trainer(experiment_name=cfg.experiment_name, ckpt_root_dir=get_param(cfg, "ckpt_root_dir", default_val=None))
            torch.cuda.empty_cache()

            res = trainer.train(
                model=model,
                train_loader=train_dataloader,
                valid_loader=val_dataloader,
                training_params=cfg.training_hyperparams,
                additional_configs_to_log=recipe_logged_cfg,
            )
            suffix = "qat"

        # EXPORT QUANTIZED MODEL TO ONNX
        input_shape = next(iter(val_dataloader))[0].shape
        os.makedirs(trainer.checkpoints_dir_path, exist_ok=True)

        qdq_onnx_path = os.path.join(trainer.checkpoints_dir_path, f"{cfg.experiment_name}_{'x'.join((str(x) for x in input_shape))}_{suffix}.onnx")
        # TODO: modify SG's convert_to_onnx for quantized models and use it instead
        export_quantized_module_to_onnx(
            model=model.cpu(),
            onnx_filename=qdq_onnx_path,
            input_shape=input_shape,
            input_size=input_shape,
            train=False,
        )

        logger.info(f"Exported {suffix.upper()} ONNX to {qdq_onnx_path}")

        return model, res
