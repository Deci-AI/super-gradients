from super_gradients.training.sg_model import SgModel

class TurnOffMosaicRecipeChangeSGModel(SgModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def change_train_recipe(self):
        # THIS CHANGE WILL SHUT DOWN THE "mosaic" DATA LOADING
        if self.dataset_interface.trainset.sample_loading_method == 'mosaic':
            self.dataset_interface.trainset.sample_loading_method = 'default'

        # TURN ON THE L1 LOSS IN YoloXDetectionLoss
        self.criterion.use_l1 = True

