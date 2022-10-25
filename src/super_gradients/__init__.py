from deci_lab_client.client import DeciPlatformClient

from super_gradients.training import ARCHITECTURES, losses, utils, datasets_utils, DataAugmentation, Trainer, KDTrainer
from super_gradients.common import init_trainer, is_distributed, object_names
from super_gradients.examples.train_from_recipe_example import train_from_recipe
from super_gradients.examples.train_from_kd_recipe_example import train_from_kd_recipe
from super_gradients.sanity_check import env_sanity_check
import atexit
__all__ = ['ARCHITECTURES', 'losses', 'utils', 'datasets_utils', 'DataAugmentation',
           'Trainer', 'KDTrainer', 'object_names',
           'init_trainer', 'is_distributed', 'train_from_recipe', 'train_from_kd_recipe',
           'env_sanity_check']

env_sanity_check()


import sys
import logging
import atexit



# logging.basicConfig(
#    level=logging.DEBUG,
#    format='%(asctime)s:%(levelname)s:%(name)s:%(message)s',
#    # filename="out.log",
#    # filemode='a'
# )
TIME_FORMAT = '%a %b %-d %Y %-I:%M:%S %p'
logformat = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
formatter = logging.Formatter(fmt=logformat, datefmt=TIME_FORMAT)


logger = logging.getLogger(__name__)
ERROR_FILE = "/home/louis.dupont/PycharmProjects/super-gradients/logfile.log"


file_handler = logging.FileHandler('/home/louis.dupont/PycharmProjects/super-gradients/logfile.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def excepthook_wrapper(excepthook):
    """Add log before the excepthook"""
    def wrapped_excepthook(exc_type, exc_value, exc_traceback):
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        excepthook(exc_type, exc_value, exc_traceback)
    return wrapped_excepthook

import os

if __name__ == "__main__":
    os.environ["DECI_PLATFORM_TOKEN"] = "eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJsb3Vpcy5kdXBvbnRAZGVjaS5haSIsImNvbXBhbnlfaWQiOiI2M2IzMGM5Yy1mYjlkLTQ0NTgtYTRmYS02ZDU2MjU3OTBlMTQiLCJ3b3Jrc3BhY2VfaWQiOiI2M2IzMGM5Yy1mYjlkLTQ0NTgtYTRmYS02ZDU2MjU3OTBlMTQiLCJjb21wYW55X25hbWUiOiJkZWNpLmFpLUxvdWlzLUR1cG9udCIsInVzZXJfaWQiOiI1ODc1ZjEwNi1hZGZkLTQ3MTEtOGY0Yy1jNDE1YmM3YTNmYTUiLCJzb3VyY2UiOiJQbGF0Zm9ybSIsImV4cCI6OTEzMTY1ODg4MywiaXNfcHJlbWl1bSI6ZmFsc2V9.gqsNzhi9PzQiAO7r8HyLDcv8ShaZaNnZPDcKYSmDpgHVznnzRgymFducULeGVumBi8770lkiv8PVH5xPTqIDSg"
    if os.getenv("DECI_PLATFORM_TOKEN"):
        print("DECI_PLATFORM_TOKEN was specified")
        sys.excepthook = excepthook_wrapper(sys.excepthook)

        import atexit
        platform_client = DeciPlatformClient()
        platform_client.login(token=os.getenv("DECI_PLATFORM_TOKEN"))
        platform_client.register_experiment(name="text_exp_louis")
        platform_client.save_experiment_file(file_path=ERROR_FILE)
        # def exit_handler(platform_client):
        #     # platform_client.register_experiment(name="text_exp_louis")
        #     # platform_client.save_experiment_file(file_path=ERROR_FILE)
        #     print("DONE")
        #
        # atexit.register(exit_handler, platform_client)
    #
    # hooks = ExitHooks()
    # hooks.hook()
    # atexit.register(exit_handler)

    # test
    # sys.exit(1)
    # print("OK")
    # 1/0
    # print("after")