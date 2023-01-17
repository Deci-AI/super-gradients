import unittest
from super_gradients.training import training_hyperparams


class TrainingParamsTest(unittest.TestCase):
    def test_get_train_params(self):
        train_params = training_hyperparams.coco2017_yolox_train_params()
        self.assertTrue(train_params["loss"] == "yolox_loss")
        self.assertTrue(train_params["max_epochs"] == 300)

    def test_get_train_params_with_overrides(self):
        train_params = training_hyperparams.coco2017_yolox_train_params(overriding_params={"max_epochs": 5})
        self.assertTrue(train_params["loss"] == "yolox_loss")
        self.assertTrue(train_params["max_epochs"] == 5)


if __name__ == "__main__":
    unittest.main()
