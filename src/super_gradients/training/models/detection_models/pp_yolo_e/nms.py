from typing import Tuple

from torch import Tensor


class MultiClassNMS(object):
    def __init__(
        self,
        score_threshold=0.05,
        nms_top_k=-1,
        keep_top_k=100,
        nms_threshold=0.5,
        normalized=True,
        nms_eta=1.0,
        return_index=False,
        return_rois_num=True,
        trt=False,
    ):
        super().__init__()
        self.score_threshold = score_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        self.nms_threshold = nms_threshold
        self.normalized = normalized
        self.nms_eta = nms_eta
        self.return_index = return_index
        self.return_rois_num = return_rois_num
        self.trt = trt

    def __call__(self, bboxes: Tensor, score: Tensor, background_label: int = -1) -> Tuple[Tensor, Tensor, Tensor]:
        """
        bboxes (Tensor|List[Tensor]): 1. (Tensor) Predicted bboxes with shape
                                         [N, M, 4], N is the batch size and M
                                         is the number of bboxes
                                      2. (List[Tensor]) bboxes and bbox_num,
                                         bboxes have shape of [M, C, 4], C
                                         is the class number and bbox_num means
                                         the number of bboxes of each batch with
                                         shape [N,]
        score (Tensor): Predicted scores with shape [N, C, M] or [M, C]
        background_label (int): Ignore the background label; For example, RCNN
                                is num_classes and YOLO is -1.
        """
        # TODO: Reimplement in torchvision
        return bboxes, score, None
        # kwargs = self.__dict__.copy()
        # if isinstance(bboxes, tuple):
        #     bboxes, bbox_num = bboxes
        #     kwargs.update({"rois_num": bbox_num})
        # if background_label > -1:
        #     kwargs.update({"background_label": background_label})
        # kwargs.pop("trt")
        # # TODO(wangxinxin08): paddle version should be develop or 2.3 and above to run nms on tensorrt
        # if self.trt and (
        #     int(paddle.version.major) == 0 or (int(paddle.version.major) >= 2 and int(paddle.version.minor) >= 3)
        # ):
        #     # TODO(wangxinxin08): tricky switch to run nms on tensorrt
        #     kwargs.update({"nms_eta": 1.1})
        #     bbox, bbox_num, _ = ops.multiclass_nms(bboxes, score, **kwargs)
        #     bbox = bbox.reshape([1, -1, 6])
        #     idx = torch.nonzero(bbox[..., 0] != -1)
        #     bbox = torch.gather_nd(bbox, idx)
        #     return bbox, bbox_num, None
        # else:
        #     return ops.multiclass_nms(bboxes, score, **kwargs)
