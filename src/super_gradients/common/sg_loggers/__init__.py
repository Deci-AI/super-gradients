from super_gradients.common.sg_loggers.base_sg_logger import BaseSGLogger
from super_gradients.common.sg_loggers.deci_platform_sg_logger import DeciPlatformSGLogger
from super_gradients.common.sg_loggers.wandb_sg_logger import WandBSGLogger
from super_gradients.common.sg_loggers.clearml_sg_logger import ClearMLSGLogger

SG_LOGGERS = {
    "base_sg_logger": BaseSGLogger,
    "deci_platform_sg_logger": DeciPlatformSGLogger,
    "wandb_sg_logger": WandBSGLogger,
    "clearml_sg_logger": ClearMLSGLogger,
}
