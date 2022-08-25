from super_gradients.training import Trainer, MultiGPUMode
from super_gradients.training.dataloaders.dataloader_factory import imagenet_train, imagenet_val
import super_gradients

super_gradients.init_trainer()
sm = Trainer("sanity_checkdl", multi_gpu=MultiGPUMode.OFF)
dltrain = imagenet_train()
dlval = imagenet_val()
print(dlval)


for x, y in dltrain:
    pass
