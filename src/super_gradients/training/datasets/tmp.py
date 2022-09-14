import os
from typing import List, Union

import PIL
import hydra
import numpy as np
import pkg_resources
from PIL import Image
import torch
from omegaconf import DictConfig

from super_gradients.common.factories.list_factory import ListFactory
from super_gradients.common.factories.transforms_factory import TransformsFactory




def image_preprocess(dataset_params: Union[dict, DictConfig, str], image: Union[np.ndarray, PIL.Image, torch.Tensor]) -> torch.Tensor:
    """
    preprocess the image by the validation set transforms of the dataset params provided

    :param dataset_params: a path to a dataset_params yaml file or a dictionary containing the content of such a file
    """
    if isinstance(dataset_params, str):
        with hydra.initialize_config_dir(config_dir=pkg_resources.resource_filename("super_gradients.recipes", "")):
            # config is relative to a module
            dataset_params = hydra.compose(config_name=os.path.join("dataset_params", dataset_params))

    if isinstance(image, torch.Tensor):
        image = image.cpu().detach().numpy()

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    transforms = ListFactory(TransformsFactory()).get(dataset_params['val_dataset_params']['transforms'])


    sample = {"image": image, "mask": image, "target": np.ndarray(), "crowd_target": None}
    for transform in transforms:
        sample = transform(sample)

    res = sample['image']
    return res


pil_image = PIL.Image.open('/home/ofri/img1.png')
np_image = np.asarray(pil_image)
torch_image = torch.tensor(np_image)


res0 = image_preprocess('imagenet_dataset_params', pil_image)
res1 = image_preprocess('imagenet_dataset_params', np_image)
res2 = image_preprocess('imagenet_dataset_params', torch_image)


res2 = image_preprocess('coco_detection_dataset_params', pil_image)
