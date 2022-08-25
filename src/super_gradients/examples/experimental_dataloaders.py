from super_gradients.training import Trainer, MultiGPUMode
from super_gradients.training.dataloaders.dataloader_factory import imagenet_train, imagenet_val, imagenet_resnet50_kd_train, imagenet_resnet50_kd_val
import super_gradients

super_gradients.init_trainer()
sm = Trainer("sanity_checkdl", multi_gpu=MultiGPUMode.DISTRIBUTED_DATA_PARALLEL)
dltrain = imagenet_resnet50_kd_train()#dataloader_params={"sampler": {"InfiniteSampler": {}}})
dlval = imagenet_resnet50_kd_val()
print(dlval)


for x, y in dltrain:
    break
for x, y in dlval:
    break
