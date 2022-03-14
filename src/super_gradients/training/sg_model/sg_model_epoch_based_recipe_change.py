from super_gradients.training.sg_model import SgModel

class TurnOffMosaicRecipeChangeSGModel(SgModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def change_train_recipe(self):
        # SHUTS DOWN THE "mosaic" DATA LOADING AUGMENTATION
        print("Closing Mosaic")
        if self.dataset_interface.trainset.sample_loading_method == 'mosaic':
            self.dataset_interface.trainset.sample_loading_method = 'default'

        # SHUTS DOWN MIXUP
        print("Closing Mixup")
        self.dataset_interface.trainset.mixup_prob = 0.
            # FIXME - TOO HARSH - COMMENTING OUT FOR NOW
            # self.dataset_interface.trainset.augment = False

        # FIXME - THIS YIELDED WORSE RESULTS IN ORIGINAL REPRODUCTION RECIPE
        # # SHUTS DOWN THE "mixup" DATA LOADING AUGMENTATION
        # self.dataset_interface.trainset.mixup_prob = 0.

        # TURN ON THE L1 LOSS IN YoloXDetectionLoss
        self.criterion.use_l1 = True

