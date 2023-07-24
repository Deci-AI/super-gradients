import os
import unittest
import tempfile

from super_gradients.common.object_names import Models
from super_gradients.training import models


class TestModelPredict(unittest.TestCase):
    def setUp(self) -> None:
        rootdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.images = [
            os.path.join(rootdir, "documentation", "source", "images", "examples", "countryside.jpg"),
            os.path.join(rootdir, "documentation", "source", "images", "examples", "street_busy.jpg"),
            "https://cdn-attachments.timesofmalta.com/cc1eceadde40d2940bc5dd20692901371622153217-1301777007-4d978a6f-620x348.jpg",
        ]

    def test_classification_models(self):
        model = models.get(Models.RESNET18, pretrained_weights="imagenet")

        with tempfile.TemporaryDirectory() as tmp_dirname:
            predictions = model.predict(self.images)
            predictions.show()
            predictions.save(output_folder=tmp_dirname)


if __name__ == "__main__":
    unittest.main()
