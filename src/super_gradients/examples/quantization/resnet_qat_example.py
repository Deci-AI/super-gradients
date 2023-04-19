import argparse

from torch import nn

import super_gradients
from super_gradients import Trainer
from super_gradients.common.object_names import Models
from super_gradients.modules.quantization.resnet_bottleneck import QuantBottleneck as sg_QuantizedBottleneck
from super_gradients.training import MultiGPUMode
from super_gradients.training import models as sg_models
from super_gradients.training.dataloaders import imagenet_train, imagenet_val
from super_gradients.training.metrics import Accuracy, Top5
from super_gradients.training.models.classification_models.resnet import Bottleneck
from super_gradients.training.models.classification_models.resnet import Bottleneck as sg_Bottleneck
from super_gradients.training.utils.quantization.calibrator import QuantizationCalibrator
from super_gradients.training.utils.quantization.core import QuantizedMetadata
from super_gradients.training.utils.quantization.export import export_quantized_module_to_onnx
from super_gradients.training.utils.quantization.selective_quantization_utils import SelectiveQuantizer


def naive_quantize(model: nn.Module):
    q_util = SelectiveQuantizer(
        default_quant_modules_calibrator_weights="max",
        default_quant_modules_calibrator_inputs="percentile",
        default_per_channel_quant_weights=True,
        default_learn_amax=False,
    )
    # SG already registers non-naive QuantBottleneck as in selective_quantize() down there, pop it for the sake of example
    if Bottleneck in q_util.mapping_instructions:
        q_util.mapping_instructions.pop(Bottleneck)
    q_util.quantize_module(model)

    return model


def selective_quantize(model: nn.Module):
    mappings = {
        sg_Bottleneck: QuantizedMetadata(
            float_source=sg_Bottleneck,
            quantized_target_class=sg_QuantizedBottleneck,
            action=QuantizedMetadata.ReplacementAction.RECURE_AND_REPLACE,
        ),
    }

    sq_util = SelectiveQuantizer(
        custom_mappings=mappings,
        default_quant_modules_calibrator_weights="max",
        default_quant_modules_calibrator_inputs="percentile",
        default_per_channel_quant_weights=True,
        default_learn_amax=False,
    )
    sq_util.quantize_module(model)

    return model


def sg_vanilla_resnet50():
    return sg_models.get(Models.RESNET50, pretrained_weights="imagenet", num_classes=1000)


def sg_naive_qdq_resnet50():
    return naive_quantize(sg_vanilla_resnet50())


def sg_selective_qdq_resnet50():
    return selective_quantize(sg_vanilla_resnet50())


models = {
    "sg_vanilla_resnet50": sg_vanilla_resnet50,
    "sg_naive_qdq_resnet50": sg_naive_qdq_resnet50,
    "sg_selective_qdq_resnet50": sg_selective_qdq_resnet50,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    super_gradients.init_trainer()

    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--calibrate", action="store_true")

    args, _ = parser.parse_known_args()

    train_params = {
        "max_epochs": args.max_epochs,
        "initial_lr": args.lr,
        "optimizer": "SGD",
        "optimizer_params": {"weight_decay": 0.0001, "momentum": 0.9, "nesterov": True},
        "loss": "cross_entropy",
        "train_metrics_list": [Accuracy(), Top5()],
        "valid_metrics_list": [Accuracy(), Top5()],
        "test_metrics_list": [Accuracy(), Top5()],
        "loss_logging_items_names": ["Loss"],
        "metric_to_watch": "Accuracy",
        "greater_metric_to_watch_is_better": True,
    }

    trainer = Trainer(experiment_name=args.model_name, multi_gpu=MultiGPUMode.OFF, device="cuda")

    train_dataloader = imagenet_train(dataloader_params={"batch_size": args.batch, "shuffle": True})
    val_dataloader = imagenet_val(dataloader_params={"batch_size": args.batch, "shuffle": True, "drop_last": True})

    model = models[args.model_name]().cuda()

    if args.calibrate:
        calibrator = QuantizationCalibrator(verbose=True)
        calibrator.calibrate_model(model, method="percentile", calib_data_loader=train_dataloader, num_calib_batches=1024 // args.batch or 1)

    trainer.train(model=model, training_params=train_params, train_loader=train_dataloader, valid_loader=val_dataloader)

    valid_metrics_dict = trainer.test(model=model, test_loader=val_dataloader, test_metrics_list=[Accuracy()], metrics_progress_verbose=True)

    export_quantized_module_to_onnx(model=model, onnx_filename=f"{args.model_name}.onnx", input_shape=(args.batch, 3, 224, 224))

    print(f"FINAL ACCURACY: {valid_metrics_dict['Accuracy'].cpu().item()}")
