from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_val


import torch
import numpy as np

from super_gradients.training.datasets.datasets_conf import COCO_DETECTION_CLASSES_LIST
from super_gradients.training.utils import sg_trainer_utils


import torch
import numpy as np


def crop_bounding_boxes(images, labels):
    """
    Crop bounding boxes from images based on the provided labels and return class IDs.

    Parameters:
    - images (torch.Tensor): A tensor of shape (batch_size, 3, 640, 640).
    - labels (torch.Tensor): A tensor of shape (total_num_boxes, 6) representing
                              (image_id, cls, x, y, w, h).

    Returns:
    - List[np.ndarray]: A list of numpy arrays representing the cropped bounding boxes.
    - List[int]: A list of class IDs for each cropped image.
    """
    cropped_images = []
    class_ids = []

    for label in labels:
        image_id, cls, x, y, w, h = label.int()
        # Convert coordinates to pixel values (assuming 640x640 images)
        x1, y1, x2, y2 = x, y, x + w, y + h

        # Crop the image directly using tensor slicing and convert to numpy array
        cropped_image = images[image_id, :, y1:y2, x1:x2].numpy()
        cropped_images.append(cropped_image.astype(int))
        class_ids.append(cls.item())

    return cropped_images, class_ids


import matplotlib.pyplot as plt
import numpy as np


import matplotlib.pyplot as plt
import numpy as np


def visualize_cropped_images_with_labels(cropped_images, labels):
    """
    Visualize a list of cropped images with class labels.

    Parameters:
    - cropped_images (List[np.ndarray]): A list of numpy arrays representing cropped images.
    - labels (List[int]): A list of integers representing class labels for each cropped image.
    """
    num_images = len(cropped_images)
    cols = min(5, num_images)  # Set number of columns in the plot
    rows = num_images // cols + int(num_images % cols > 0)  # Calculate required rows

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    # If there's only one row, axs may not be an array
    if rows == 1:
        axs = [axs]

    for i, img in enumerate(cropped_images):
        ax = axs[i // cols][i % cols] if rows > 1 else axs[i]
        img = np.transpose(img, (1, 2, 0))
        img = img[:, :, ::-1]
        ax.imshow(img)  # Convert from (C, H, W) to (H, W, C)
        ax.axis("off")
        class_id = labels[i]
        ax.set_title(COCO_DETECTION_CLASSES_LIST[class_id])

    # Hide any unused subplots
    for i in range(num_images, rows * cols):
        axs[i // cols][i % cols].axis("off")

    plt.tight_layout()
    plt.show()


import numpy as np
import torch


# Example usage
# cropped_images = [your list of cropped images in numpy format]
# preprocessed_images = preprocess_images_albumentations(cropped_images)


dl = coco_detection_yolo_format_val()
first_train_batch = next(iter(dl))
batch_images, batch_labels, _ = sg_trainer_utils.unpack_batch_items(first_train_batch)

cropped_images, labels = crop_bounding_boxes(batch_images, batch_labels)
# visualize_cropped_images_with_labels(cropped_images, labels)

from super_gradients.training import models
from super_gradients.common.object_names import Models

model = models.get(Models.VIT_BASE, pretrained_weights="imagenet")
model.eval()
cropped_images = [np.transpose(c, (1, 2, 0)).astype(np.uint8)[:, :, ::-1] for c in cropped_images]
p = model.predict(cropped_images)
p.show()
