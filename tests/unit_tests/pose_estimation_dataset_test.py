from super_gradients.training.datasets.pose_estimation_datasets.coco_keypoints import COCOKeypoints
from super_gradients.training.datasets.pose_estimation_datasets.target_generators import DEKRTargetsGenerator
from super_gradients.training.transforms.keypoint_transforms import KeypointsCompose, KeypointsRandomVerticalFlip


def test_dataset():
    target_generator = DEKRTargetsGenerator(
        output_stride=4,
        sigma=2,
        center_sigma=4,
        bg_weight=0.1,
        offset_radius=4,
    )

    dataset = COCOKeypoints(
        dataset_root="e:/coco2017",
        images_dir="images/train2017",
        json_file="annotations/person_keypoints_train2017.json",
        include_empty_samples=False,
        transforms=KeypointsCompose(
            [
                KeypointsRandomVerticalFlip(),
            ]
        ),
        target_generator=target_generator,
    )

    assert dataset is not None
    assert dataset[0] is not None
