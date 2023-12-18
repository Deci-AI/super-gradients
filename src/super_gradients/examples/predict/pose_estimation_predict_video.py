import torch
from super_gradients.training import models

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path_to_video", type=str)

if __name__ == "__main__":
    args = parser.parse_args()

    # Note that currently only YoloX, PPYoloE and YOLO-NAS are supported.
    model = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose")

    # We want to use cuda if available to speed up inference.
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    predictions = model.predict(args.path_to_video)
    predictions.save(f"{args.path_to_video.split('/')[-1]}_prediction.mp4")

    predictions = model.predict(args.path_to_video)
    predictions.save(f"{args.path_to_video.split('/')[-1]}_prediction.gif")  # Can also be saved as a gif.

    predictions = model.predict(args.path_to_video)
    predictions.show()
