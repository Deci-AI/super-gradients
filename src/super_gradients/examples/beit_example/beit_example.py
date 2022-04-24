from super_gradients.training.sg_model import SgModel
import super_gradients

super_gradients.init_trainer()

sg_model = SgModel("beitexample")
sg_model.build_model("beit_base_patch16_224", arch_params={"num_classes": 1000}, checkpoint_params={"pretrained_weights": "imagenet"})
