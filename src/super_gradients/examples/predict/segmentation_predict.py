from super_gradients.common.object_names import Models
from super_gradients.training import models


model = models.get(model_name=Models.PP_LITE_T_SEG75, pretrained_weights="cityscapes")

IMAGES = [
    "https://deci-pretrained-models.s3.amazonaws.com/sample_images/beatles-abbeyroad.jpg",
]

predictions = model.predict(IMAGES)
predictions.show()
predictions.save(output_folder="")  # Save in working directory
