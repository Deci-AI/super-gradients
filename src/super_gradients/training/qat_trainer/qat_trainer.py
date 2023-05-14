import os
from typing import Union, Tuple, Dict, Mapping, List
from torchmetrics import Metric

import copy
import hydra
import torch.cuda
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.common.environment.cfg_utils import load_recipe
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
    def quantize_from_config(cls, cfg: Union[DictConfig, dict]) -> Tuple[nn.Module, Tuple]:
        """
        Perform quantization aware training (QAT) according to a recipe configuration.

        This method will instantiate all the objects specified in the recipe, build and quantize the model,
        and calibrate the quantized model. The resulting quantized model and the output of the trainer.train()
        method will be returned.

        The quantized model will be exported to ONNX along with other checkpoints.

        The call to self.quantize (see docs in the next method) is done with the created
         train_loader and valid_loader. If no calibration data loader is passed through cfg.calib_loader,
         a train data laoder with the validation transforms is used for calibration.

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

        quantization_params = get_param(cfg, "quantization_params")

        if quantization_params is None:
            raise logger.warning("Your recipe does not include quantization_params. Using default quantization params.")

        if get_param(cfg.checkpoint_params, "checkpoint_path") is None and get_param(cfg.checkpoint_params, "pretrained_weights") is None:
            raise ValueError("Starting checkpoint / pretrained weights are a must for QAT finetuning.")

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

        recipe_logged_cfg = {"recipe_config": OmegaConf.to_container(cfg, resolve=True)}
        trainer = QATTrainer(experiment_name=cfg.experiment_name, ckpt_root_dir=get_param(cfg, "ckpt_root_dir"))

        res = trainer.quantize(
            model=model,
            quantization_params=quantization_params,
            calib_dataloader=calib_dataloader,
            val_dataloader=val_dataloader,
            train_dataloader=train_dataloader,
            training_params=cfg.training_hyperparams,
            additional_qat_configs_to_log=recipe_logged_cfg,
        )

        return model, res

    def quantize(
        self,
        calib_dataloader: DataLoader,
        model: torch.nn.Module = None,
        val_dataloader: DataLoader = None,
        train_dataloader: DataLoader = None,
        quantization_params: Mapping = None,
        training_params: Mapping = None,
        additional_qat_configs_to_log: Dict = None,
        valid_metrics_list: List[Metric] = None,
    ):
        """
        Performs post-training quantization (PTQ), and optionally quantization-aware training (QAT).
        Exports the ONNX model to the checkpoints directory.

        :param calib_dataloader: DataLoader, data loader for calibration.

        :param model: torch.nn.Module, Model to perform QAT/PTQ on. When None, will try to use self.net which is set
        in previous self.train(..) call (default=None).


        :param val_dataloader: DataLoader, data loader for validation. Used both for validating the calibrated model after PTQ and during QAT.
            When None, will try to use self.valid_loader if it was set in previous self.train(..) call (default=None).

        :param train_dataloader: DataLoader, data loader for QA training, can be ignored when quantization_params["ptq_only"]=True (default=None).

        :param quantization_params: Mapping, with the following entries:defaults-

            ptq_only: False              # whether to launch QAT, or leave PTQ only
            selective_quantizer_params:
              calibrator_w: "max"        # calibrator type for weights, acceptable types are ["max", "histogram"]
              calibrator_i: "histogram"  # calibrator type for inputs acceptable types are ["max", "histogram"]
              per_channel: True          # per-channel quantization of weights, activations stay per-tensor by default
              learn_amax: False          # enable learnable amax in all TensorQuantizers using straight-through estimator
              skip_modules:              # optional list of module names (strings) to skip from quantization

            calib_params:
              histogram_calib_method: "percentile"  # calibration method for all "histogram" calibrators, acceptable types are ["percentile", "entropy", mse"],
               "max" calibrators always use "max"
              percentile: 99.99                     # percentile for all histogram calibrators with method "percentile", other calibrators are not affected

              num_calib_batches:                    # number of batches to use for calibration, if None, 512 / batch_size will be used
              verbose: False                        # if calibrator should be verbose


        :param training_params: Mapping, training hyper parameters for QAT, same as in super.train(...). When None, will try to use self.training_params
         which is set in previous self.train(..) call (default=None).

        :param additional_qat_configs_to_log: Dict, Optional dictionary containing configs that will be added to the QA training's
         sg_logger. Format should be {"Config_title_1": {...}, "Config_title_2":{..}}.

        :param valid_metrics_list:  (list(torchmetrics.Metric)) metrics list for evaluation of the calibrated model.
        When None, the validation metrics from training_params are used). (default=None).

        :return: Validation results of the QAT model in case quantization_params['ptq_only']=False and of the PTQ
        model otherwise.
        """

        if quantization_params is None:
            quantization_params = load_recipe("quantization_params/default_quantization_params").quantization_params
            logger.info(f"Using default quantization params: {quantization_params}")
        training_params = training_params or self.training_params.to_dict()
        valid_metrics_list = valid_metrics_list or get_param(training_params, "valid_metrics_list")
        train_dataloader = train_dataloader or self.train_loader
        val_dataloader = val_dataloader or self.valid_loader
        model = model or self.net

        res = self.calibrate_model(
            calib_dataloader=calib_dataloader,
            model=model,
            quantization_params=quantization_params,
            val_dataloader=val_dataloader,
            valid_metrics_list=valid_metrics_list,
        )
        # TRAIN
        if get_param(quantization_params, "ptq_only", False):
            logger.info("quantization_params.ptq_only=True. Performing PTQ only!")
            suffix = "ptq"
        else:
            model.train()
            torch.cuda.empty_cache()

            res = self.train(
                model=model,
                train_loader=train_dataloader,
                valid_loader=val_dataloader,
                training_params=training_params,
                additional_configs_to_log=additional_qat_configs_to_log,
            )
            suffix = "qat"
        # EXPORT QUANTIZED MODEL TO ONNX
        input_shape = next(iter(val_dataloader))[0].shape
        os.makedirs(self.checkpoints_dir_path, exist_ok=True)
        qdq_onnx_path = os.path.join(self.checkpoints_dir_path, f"{self.experiment_name}_{'x'.join((str(x) for x in input_shape))}_{suffix}.onnx")

        # TODO: modify SG's convert_to_onnx for quantized models and use it instead
        export_quantized_module_to_onnx(
            model=model.cpu(),
            onnx_filename=qdq_onnx_path,
            input_shape=input_shape,
            input_size=input_shape,
            train=False,
        )
        logger.info(f"Exported {suffix.upper()} ONNX to {qdq_onnx_path}")
        return res

    def calibrate_model(self, calib_dataloader, model, quantization_params, val_dataloader, valid_metrics_list):
        """
        Performs calibration.

        :param calib_dataloader: DataLoader, data loader for calibration.

        :param model: torch.nn.Module, Model to perform calibration on. When None, will try to use self.net which is
        set in previous self.train(..) call (default=None).

        :param val_dataloader: DataLoader, data loader for validation. Used both for validating the calibrated model.
            When None, will try to use self.valid_loader if it was set in previous self.train(..) call (default=None).

        :param quantization_params: Mapping, with the following entries:defaults-
            selective_quantizer_params:
              calibrator_w: "max"        # calibrator type for weights, acceptable types are ["max", "histogram"]
              calibrator_i: "histogram"  # calibrator type for inputs acceptable types are ["max", "histogram"]
              per_channel: True          # per-channel quantization of weights, activations stay per-tensor by default
              learn_amax: False          # enable learnable amax in all TensorQuantizers using straight-through estimator
              skip_modules:              # optional list of module names (strings) to skip from quantization

            calib_params: histogram_calib_method: "percentile"  # calibration method for all "histogram" calibrators,
            acceptable types are ["percentile", "entropy", mse"], "max" calibrators always use "max" percentile:
            99.99                     # percentile for all histogram calibrators with method "percentile",
            other calibrators are not affected num_calib_batches:                    # number of batches to use for
            calibration, if None, 512 / batch_size will be used verbose: False                        # if calibrator
            should be verbose



        :param valid_metrics_list:  (list(torchmetrics.Metric)) metrics list for evaluation of the calibrated model.

        :return: Validation results of the calibrated model.
        """
        selective_quantizer_params = get_param(quantization_params, "selective_quantizer_params")
        calib_params = get_param(quantization_params, "calib_params")
        model = model or self.net
        model.to(device_config.device)
        # QUANTIZE MODEL
        model.eval()
        fuse_repvgg_blocks_residual_branches(model)
        q_util = SelectiveQuantizer(
            default_quant_modules_calibrator_weights=get_param(selective_quantizer_params, "calibrator_w"),
            default_quant_modules_calibrator_inputs=get_param(selective_quantizer_params, "calibrator_i"),
            default_per_channel_quant_weights=get_param(selective_quantizer_params, "per_channel"),
            default_learn_amax=get_param(selective_quantizer_params, "learn_amax"),
            verbose=get_param(calib_params, "verbose"),
        )
        q_util.register_skip_quantization(layer_names=get_param(selective_quantizer_params, "skip_modules"))
        q_util.quantize_module(model)
        # CALIBRATE MODEL
        logger.info("Calibrating model...")
        calibrator = QuantizationCalibrator(
            verbose=get_param(calib_params, "verbose"),
            torch_hist=True,
        )
        calibrator.calibrate_model(
            model,
            method=get_param(calib_params, "histogram_calib_method"),
            calib_data_loader=calib_dataloader,
            num_calib_batches=get_param(calib_params, "num_calib_batches") or len(calib_dataloader),
            percentile=get_param(calib_params, "percentile", 99.99),
        )
        calibrator.reset_calibrators(model)  # release memory taken by calibrators
        # VALIDATE PTQ MODEL AND PRINT SUMMARY
        logger.info("Validating PTQ model...")
        valid_metrics_dict = self.test(model=model, test_loader=val_dataloader, test_metrics_list=valid_metrics_list)
        results = ["PTQ Model Validation Results"]
        results += [f"   - {metric:10}: {value}" for metric, value in valid_metrics_dict.items()]
        logger.info("\n".join(results))

        return valid_metrics_dict
