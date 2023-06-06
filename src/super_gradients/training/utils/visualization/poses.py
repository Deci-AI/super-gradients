import cv2

from super_gradients.training.utils.visualization.detection import draw_bbox


def draw_skeleton(image, keypoints, score, joint_links, joint_colors=None, keypoint_colors=None, show_confidence=False):
    image = image.copy()
    if joint_colors is None:
        joint_colors = [(255, 0, 255)] * len(joint_links)

    if keypoint_colors is None:
        keypoint_colors = [(0, 255, 0)] * len(keypoints)

    keypoints = keypoints.astype(int)
    for joint, color in zip(joint_links, joint_colors):
        p1 = tuple(keypoints[joint[0]][:2])
        p2 = tuple(keypoints[joint[1]][:2])
        cv2.line(image, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)

    for keypoint, color in zip(keypoints, keypoint_colors):
        cv2.circle(image, tuple(keypoint[:2]), radius=3, color=color, thickness=-1, lineType=cv2.LINE_AA)

    if show_confidence:
        x, y, w, h = cv2.boundingRect(keypoints[..., 0:2])
        image = draw_bbox(image, title=f"{score:.2f}", box_thickness=2, color=(255, 0, 255), x1=x, y1=y, x2=x + w, y2=y + h)

    return image
