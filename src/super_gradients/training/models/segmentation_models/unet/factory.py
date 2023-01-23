from super_gradients.common.factories.base_factory import BaseFactory
from super_gradients.training.models.segmentation_models.unet.unet_encoder import RepVGGStage, QARepVGGStage, STDCStage, RegnetXStage
from super_gradients.training.models.segmentation_models.unet.unet_decoder import UpCatBlock, UpFactorBlock


BACKBONE_STAGES = dict(
    RepVGGStage=RepVGGStage,
    QARepVGGStage=QARepVGGStage,
    STDCStage=STDCStage,
    RegnetXStage=RegnetXStage,
)

UP_FUSE_BLOCKS = dict(
    UpCatBlock=UpCatBlock,
    UpFactorBlock=UpFactorBlock,
)


class BackboneStageFactory(BaseFactory):
    def __init__(self):
        super().__init__(BACKBONE_STAGES)


class UpFuseBlockFactory(BaseFactory):
    def __init__(self):
        super().__init__(UP_FUSE_BLOCKS)
