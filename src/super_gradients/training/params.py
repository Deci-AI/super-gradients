from super_gradients.training.utils import HpmStruct
from copy import deepcopy

DEFAULT_TRAINING_PARAMS = {
    "lr_warmup_epochs": 0,
    "lr_warmup_steps": 0,
    "lr_cooldown_epochs": 0,
    "warmup_initial_lr": None,
    "cosine_final_lr_ratio": 0.01,
    "optimizer": "SGD",
    "optimizer_params": {},
    "criterion_params": {},
    "ema": False,
    "batch_accumulate": 1,  # number of batches to accumulate before every backward pass
    "ema_params": {},
    "zero_weight_decay_on_bias_and_bn": False,
    "load_opt_params": True,
    "run_validation_freq": 1,
    "save_model": True,
    "metric_to_watch": "Accuracy",
    "launch_tensorboard": False,
    "tb_files_user_prompt": False,  # Asks User for Tensorboard Deletion Prompt
    "silent_mode": False,  # Silents the Print outs
    "mixed_precision": False,
    "tensorboard_port": None,
    "save_ckpt_epoch_list": [],  # indices where the ckpt will save automatically
    "average_best_models": True,
    "dataset_statistics": False,  # add a dataset statistical analysis and sample images to tensorboard
    "save_tensorboard_to_s3": False,
    "lr_schedule_function": None,
    "train_metrics_list": [],
    "valid_metrics_list": [],
    "greater_metric_to_watch_is_better": True,
    "precise_bn": False,
    "precise_bn_batch_size": None,
    "seed": 42,
    "lr_mode": None,
    "phase_callbacks": None,
    "log_installed_packages": True,
    "sg_logger": "base_sg_logger",
    "sg_logger_params": {
        "tb_files_user_prompt": False,  # Asks User for Tensorboard Deletion Prompt
        "project_name": "",
        "launch_tensorboard": False,
        "tensorboard_port": None,
        "save_checkpoints_remote": False,  # upload checkpoint files to s3
        "save_tensorboard_remote": False,  # upload tensorboard files to s3
        "save_logs_remote": False,
    },  # upload log files to s3
    "warmup_mode": "linear_step",
    "step_lr_update_freq": None,
    "lr_updates": [],
    "clip_grad_norm": None,
    "pre_prediction_callback": None,
    "ckpt_best_name": "ckpt_best.pth",
    "enable_qat": False,
    "resume": False,
    "resume_path": None,
    "ckpt_name": "ckpt_latest.pth",
    "resume_strict_load": False,
    "sync_bn": False,
    "kill_ddp_pgroup_on_end": True,  # Whether to kill the DDP process group in the end of training.
    "max_train_batches": None,  # For debug- when not None- will break out of inner train loop
    # (i.e iterating over train_loader) when reaching this number of batches.
    "max_valid_batches": None,  # For debug- when not None- will break out of inner valid loop
    # (i.e iterating over valid_loader) when reaching this number of batches.
    "resume_from_remote_sg_logger": False  # When true, ckpt_name (checkpoint filename to resume, ckpt_latest.pth by
    # default) will be downloaded into the experiment checkpoints directory prior to loading weights, then resumed
    # from that checkpoint. The source is unique to every logger, and currently supported for WandB loggers only.
    # Note that for this to work, the experiment must be ran with sg_logger_params.save_checkpoints_remote=True. For
    # WandB loggers, one must also pass the run id through the wandb_id arg in sg_logger_params.
}

DEFAULT_OPTIMIZER_PARAMS_SGD = {"weight_decay": 1e-4, "momentum": 0.9}

DEFAULT_OPTIMIZER_PARAMS_ADAM = {"weight_decay": 1e-4}

DEFAULT_OPTIMIZER_PARAMS_RMSPROP = {"weight_decay": 1e-4, "momentum": 0.9}

DEFAULT_OPTIMIZER_PARAMS_RMSPROPTF = {"weight_decay": 1e-4, "momentum": 0.9}

TRAINING_PARAM_SCHEMA = {
    "type": "object",
    "properties": {
        "max_epochs": {"type": "number", "minimum": 1},
        # FIXME: CHECK THE IMPORTANCE OF THE COMMENTED SCHEMA- AS IT CAUSES HYDRA USE TO CRASH
        # "lr_updates": {"type": "array", "minItems": 1},
        "lr_decay_factor": {"type": "number", "minimum": 0, "maximum": 1},
        "lr_warmup_epochs": {"type": "number", "minimum": 0, "maximum": 10},
        "initial_lr": {"type": "number", "exclusiveMinimum": 0, "maximum": 10},
    },
    "if": {"properties": {"lr_mode": {"const": "step"}}},
    "then": {"required": ["lr_updates", "lr_decay_factor"]},
    "required": ["max_epochs", "lr_mode", "initial_lr", "loss"],
}


class TrainingParams(HpmStruct):
    def __init__(self, **entries):
        # WE initialize by the default training params, overridden by the provided params
        default_training_params = deepcopy(DEFAULT_TRAINING_PARAMS)
        super().__init__(**default_training_params)
        self.set_schema(TRAINING_PARAM_SCHEMA)
        if len(entries) > 0:
            self.override(**entries)

    def override(self, **entries):
        super().override(**entries)
        self.validate()
