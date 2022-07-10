from super_gradients.common.sg_loggers.base_sg_logger import BaseSGLogger
from super_gradients.common.sg_loggers.platform_sg_logger import PlatformSGLogger
from super_gradients.common.sg_loggers.wandb_sg_logger import WandBSGLogger

SG_LOGGERS = {'base_sg_logger': BaseSGLogger,
              'platform_sg_logger': PlatformSGLogger,
              'wandb_sg_logger': WandBSGLogger}
