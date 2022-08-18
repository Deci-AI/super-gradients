from super_gradients.training import SgModel, MultiGPUMode
from super_gradients.training.dataloaders.dataloader_factory import coco2017_train, coco2017_val
import super_gradients
from super_gradients.training.utils.detection_utils import CrowdDetectionCollateFN

super_gradients.init_trainer()
sm = SgModel("sanity_checkdl", multi_gpu=MultiGPUMode.DISTRIBUTED_DATA_PARALLEL)
dltrain = coco2017_train()
dlval = coco2017_val(dataset_params={"with_crowd": True}, dataloader_params={"collate_fn": CrowdDetectionCollateFN()})
print(dlval)


