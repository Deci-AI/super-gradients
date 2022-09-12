from io import BytesIO

import requests
from PIL import Image

from super_gradients.training import dataloaders, models


def det():
    # Load image

    # TODO: Get better URL (out of coco dataset maybe)
    url = "https://miro.medium.com/max/737/1*7E4DF-UxaEpjlFAxB9R20w.png"

    response = requests.get(url)
    image = Image.open(BytesIO(response.content))

    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.PILToTensor(),
    ])

    image = transform(image)

    print(image.shape, type(image))

    val_dataloader = dataloaders.get(name="coco2017_val")
    image, _ = next(iter(val_dataloader))
    print(image.shape, type(image))

    # Get pretrained model
    model = models.get("yolox_s", pretrained_weights="coco")

    # Run predict
    output = model(image)  # model.predict(image)


def seg():
    # Load image
    val_dataloader = dataloaders.get(name="cityscapes_val")
    image, label = next(iter(val_dataloader))
    print(image.shape, label.shape)

    # Get pretrained model
    model = models.get("pp_lite_t_seg", pretrained_weights="stdc1_seg50_cityscapes")

    # Run predict
    output = model(image)  # model.predict(image)
    print(output[0].shape)


def cls():
    # Load image
    val_dataloader = dataloaders.get(name="imagenet_val")
    image, label = next(iter(val_dataloader))
    print(image.shape, label.shape)

    # Get pretrained model
    model = models.get("resnet18", pretrained_weights="imagenet")

    # Run predict
    output = model(image)  # model.predict(image)
    print(output.shape)


if __name__ == '__main__':
    det()
    # cls()
#    seg()

