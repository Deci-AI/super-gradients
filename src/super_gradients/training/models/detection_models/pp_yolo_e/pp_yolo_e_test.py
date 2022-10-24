import collections
import unittest

import numpy as np
import paddle
import ppdet
import torch
from paddle import load as paddle_load
from ppdet.modeling.backbones import CSPResNet as PPCSPResNet
from ppdet.modeling.heads import PPYOLOEHead as Paddle_PPYOLOEHead
from pytorch_toolbelt.utils import count_parameters, describe_outputs
from super_gradients.training.losses.ppyolo_loss import PPYoloLoss
from super_gradients.training.models.detection_models.csp_resnet import CSPResNet
from super_gradients.training.models.detection_models.pp_yolo_e.nms import MultiClassNMS
from super_gradients.training.models.detection_models.pp_yolo_e.pan import CustomCSPPAN
from super_gradients.training.models.detection_models.pp_yolo_e.pp_yolo_head import PPYOLOEHead
from torch import nn


def convert_weights_from_padldle_to_torch(state_dict: collections.OrderedDict) -> collections.OrderedDict:
    torch_state_dict = []

    for key, value in state_dict.items():
        torch_value = torch.from_numpy(value.numpy())
        key = key.replace("bn._mean", "bn.running_mean").replace("bn._variance", "bn.running_var")
        torch_state_dict.append((key, torch_value))

    return collections.OrderedDict(torch_state_dict)


class PPYoloTestCast(unittest.TestCase):
    def test_csp_resnet(self):
        input_np = np.random.standard_normal((4, 3, 640, 512)).astype(np.float32)

        for config, pretrain in [
            (
                dict(depth_mult=0.33, width_mult=0.50, use_large_stem=True),
                "D:/Develop/GitHub/PaddlePaddle/PaddleDetection/ppdet/modeling/backbones/CSPResNetb_s_pretrained.pdparams",
            ),
            (
                dict(depth_mult=0.67, width_mult=0.75, use_large_stem=True),
                "D:/Develop/GitHub/PaddlePaddle/PaddleDetection/ppdet/modeling/backbones/CSPResNetb_m_pretrained.pdparams",
            ),
            (
                dict(depth_mult=1, width_mult=1, use_large_stem=True),
                "D:/Develop/GitHub/PaddlePaddle/PaddleDetection/ppdet/modeling/backbones/CSPResNetb_l_pretrained.pdparams",
            ),
            (
                dict(depth_mult=1.33, width_mult=1.25, use_large_stem=True),
                "D:/Develop/GitHub/PaddlePaddle/PaddleDetection/ppdet/modeling/backbones/CSPResNetb_x_pretrained.pdparams",
            ),
        ]:
            paddle_weights = paddle_load(pretrain)

            pp_encoder = PPCSPResNet(**config)
            pp_encoder.eval()
            pp_encoder.set_state_dict(paddle_weights)
            pp_tensor = paddle.to_tensor(input_np)
            pp_output = pp_encoder(dict(image=pp_tensor))

            print("Testing raw pytorch model")
            print("Out spec", pp_encoder.out_shape)

            pt_encoder = CSPResNet(**config).cuda().eval()
            pt_encoder.load_state_dict(convert_weights_from_padldle_to_torch(paddle_weights))
            input_pt = torch.from_numpy(input_np).cuda()
            pt_output = pt_encoder(input_pt)

            print(count_parameters(pt_encoder, human_friendly=True))
            print(describe_outputs(pt_output))

            np.testing.assert_allclose(pt_output[0].detach().cpu().numpy(), pp_output[0].numpy(), atol=1e-4)
            np.testing.assert_allclose(pt_output[1].detach().cpu().numpy(), pp_output[1].numpy(), atol=1e-4)
            np.testing.assert_allclose(pt_output[2].detach().cpu().numpy(), pp_output[2].numpy(), atol=1e-4)

            print("Testing prep_model_for_conversion")
            pt_encoder.prep_model_for_conversion(input_size=input_np.shape[2:])
            pt_output = pt_encoder(input_pt)
            print(count_parameters(pt_encoder, human_friendly=True))
            print(describe_outputs(pt_output))

            np.testing.assert_allclose(pt_output[0].detach().cpu().numpy(), pp_output[0].numpy(), atol=1e-4)
            np.testing.assert_allclose(pt_output[1].detach().cpu().numpy(), pp_output[1].numpy(), atol=1e-4)
            np.testing.assert_allclose(pt_output[2].detach().cpu().numpy(), pp_output[2].numpy(), atol=1e-4)

            print("Testing traced model")
            with torch.no_grad():
                pt_encoder_traced = torch.jit.trace(pt_encoder, example_inputs=input_pt)
            pt_output = pt_encoder_traced(input_pt)
            print(describe_outputs(pt_output))

            np.testing.assert_allclose(pt_output[0].detach().cpu().numpy(), pp_output[0].numpy(), atol=1e-4)
            np.testing.assert_allclose(pt_output[1].detach().cpu().numpy(), pp_output[1].numpy(), atol=1e-4)
            np.testing.assert_allclose(pt_output[2].detach().cpu().numpy(), pp_output[2].numpy(), atol=1e-4)

    def test_csp_neck(self):
        for config, pretrain in [
            (
                dict(depth_mult=0.33, width_mult=0.50, use_large_stem=True),
                "D:/Develop/GitHub/PaddlePaddle/PaddleDetection/ppdet/modeling/backbones/ppyoloe_plus_crn_s_80e_coco.pdparams",
            ),
            (
                dict(depth_mult=0.67, width_mult=0.75, use_large_stem=True),
                "D:/Develop/GitHub/PaddlePaddle/PaddleDetection/ppdet/modeling/backbones/ppyoloe_plus_crn_m_80e_coco.pdparams",
            ),
            (
                dict(depth_mult=1, width_mult=1, use_large_stem=True),
                "D:/Develop/GitHub/PaddlePaddle/PaddleDetection/ppdet/modeling/backbones/ppyoloe_plus_crn_l_80e_coco.pdparams",
            ),
            (
                dict(depth_mult=1.33, width_mult=1.25, use_large_stem=True),
                "D:/Develop/GitHub/PaddlePaddle/PaddleDetection/ppdet/modeling/backbones/ppyoloe_plus_crn_x_80e_coco.pdparams",
            ),
        ]:
            paddle_weights = paddle_load(pretrain)
            neck_weights = collections.OrderedDict(
                [(k.replace("neck.", ""), v) for (k, v) in paddle_weights.items() if k.startswith("neck")]
            )

            in_channels = (256, 512, 1024)
            in_channels = tuple([max(round(num_channels * config["width_mult"]), 1) for num_channels in in_channels])

            fm1 = np.random.standard_normal((4, in_channels[2], 24, 16)).astype(np.float32)
            fm2 = np.random.standard_normal((4, in_channels[1], 48, 32)).astype(np.float32)
            fm3 = np.random.standard_normal((4, in_channels[0], 96, 64)).astype(np.float32)

            pp_pan = ppdet.modeling.necks.CustomCSPPAN(
                in_channels=in_channels,
                out_channels=(768, 384, 192),
                stage_num=1,
                block_num=3,
                block_size=3,
                spp=True,
                act="swish",
                depth_mult=config["depth_mult"],
                width_mult=config["width_mult"],
            )
            pp_pan.eval()
            pp_pan.set_state_dict(neck_weights)
            pp_inputs = [paddle.to_tensor(fm3), paddle.to_tensor(fm2), paddle.to_tensor(fm1)]
            pp_outputs = pp_pan(pp_inputs)

            pt_pan = CustomCSPPAN(
                in_channels=in_channels,
                out_channels=(768, 384, 192),
                stage_num=1,
                block_num=3,
                block_size=3,
                spp=True,
                activation_type=nn.SiLU,
                depth_mult=config["depth_mult"],
                width_mult=config["width_mult"],
            )
            pt_pan.eval()
            pt_pan.load_state_dict(convert_weights_from_padldle_to_torch(neck_weights), strict=True)
            print("Params before conversion", count_parameters(pt_pan))
            pt_inputs = [torch.from_numpy(fm3), torch.from_numpy(fm2), torch.from_numpy(fm1)]

            pt_outputs = pt_pan(pt_inputs)
            np.testing.assert_allclose(pt_outputs[0].detach().cpu().numpy(), pp_outputs[0].numpy(), atol=1e-4)
            np.testing.assert_allclose(pt_outputs[1].detach().cpu().numpy(), pp_outputs[1].numpy(), atol=1e-4)
            np.testing.assert_allclose(pt_outputs[2].detach().cpu().numpy(), pp_outputs[2].numpy(), atol=1e-4)

            pt_pan.prep_model_for_conversion(input_size=(512, 512))
            print("Params after conversion", count_parameters(pt_pan))

            pt_outputs = pt_pan(pt_inputs)
            np.testing.assert_allclose(pt_outputs[0].detach().cpu().numpy(), pp_outputs[0].numpy(), atol=1e-4)
            np.testing.assert_allclose(pt_outputs[1].detach().cpu().numpy(), pp_outputs[1].numpy(), atol=1e-4)
            np.testing.assert_allclose(pt_outputs[2].detach().cpu().numpy(), pp_outputs[2].numpy(), atol=1e-4)

    def test_pp_yolo_head(self):
        fm1 = np.random.standard_normal((4, 384, 24, 16)).astype(np.float32)
        fm2 = np.random.standard_normal((4, 192, 48, 32)).astype(np.float32)
        fm3 = np.random.standard_normal((4, 96, 96, 64)).astype(np.float32)

        pp_head = Paddle_PPYOLOEHead(
            in_channels=(384, 192, 96),
            num_classes=80,
            fpn_strides=(32, 16, 8),
            grid_cell_scale=5.0,
            grid_cell_offset=0.5,
            static_assigner_epoch=30,
            use_varifocal_loss=True,
            loss_weight={"class": 1.0, "iou": 2.5, "dfl": 0.5},
            static_assigner=ppdet.modeling.assigners.ATSSAssigner(topk=9),
            assigner=ppdet.modeling.assigners.TaskAlignedAssigner(topk=13, alpha=1.0, beta=6.0),
            nms=ppdet.modeling.MultiClassNMS(nms_top_k=1000, keep_top_k=300, score_threshold=0.01, nms_threshold=0.7),
        )

        pt_head = PPYOLOEHead(
            in_channels=(384, 192, 96),
            num_classes=80,
            fpn_strides=(32, 16, 8),
            grid_cell_scale=5.0,
            grid_cell_offset=0.5,
            nms=MultiClassNMS(nms_top_k=1000, keep_top_k=300, score_threshold=0.01, nms_threshold=0.7),
        )
        pt_head.train().cuda()
        pt_loss = PPYoloLoss(num_classes=pt_head.num_classes, use_static_assigner=True).cuda()

        pt_inputs = [torch.from_numpy(fm1).cuda(), torch.from_numpy(fm2).cuda(), torch.from_numpy(fm3).cuda()]
        pp_inputs = [paddle.to_tensor(fm1), paddle.to_tensor(fm2), paddle.to_tensor(fm3)]

        pt_targets = {
            "gt_class": torch.tensor(
                [
                    [[1], [2], [3]],
                    [[4], [3], [0]],
                    [[1], [0], [0]],
                    [[0], [0], [0]],
                ]
            )
            .long()
            .cuda(),
            "gt_bbox": torch.tensor(
                [
                    [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]],
                    [[10, 20, 30, 40], [50, 60, 70, 80], [0, 0, 0, 0]],
                    [[10, 20, 30, 40], [0, 0, 0, 0], [0, 0, 0, 0]],
                    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                ]
            )
            .float()
            .cuda(),
            # Zero values in pad_gt_mask indicate padded
            "pad_gt_mask": torch.tensor(
                [
                    [[1], [1], [1]],
                    [[1], [1], [0]],
                    [[1], [0], [0]],
                    [[0], [0], [0]],
                ]
            ).cuda(),
            "epoch_id": 1,
        }

        pp_targets = {
            "gt_class": paddle.to_tensor(
                [
                    [[1], [2], [3]],
                    [[4], [3], [0]],
                    [[1], [0], [0]],
                    [[0], [0], [0]],
                ],
                dtype="int32",
            ),
            "gt_bbox": paddle.to_tensor(
                [
                    [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]],
                    [[10, 20, 30, 40], [50, 60, 70, 80], [0, 0, 0, 0]],
                    [[10, 20, 30, 40], [0, 0, 0, 0], [0, 0, 0, 0]],
                    [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                ],
                dtype="float32",
            ),
            # Zero values in pad_gt_mask indicate padded
            "pad_gt_mask": paddle.to_tensor(
                [
                    [[1], [1], [1]],
                    [[1], [1], [0]],
                    [[1], [0], [0]],
                    [[0], [0], [0]],
                ],
                dtype="bool",
            ),
            "epoch_id": 1,
        }

        print(count_parameters(pt_head))

        # With static assigner
        print("Testing whether PP-Yolo-E head produced same output for static assigner")
        pt_head_preds = pt_head(pt_inputs)
        pt_outputs = pt_loss(pt_head_preds, pt_targets)
        pp_outputs = pp_head(pp_inputs, pp_targets)

        print(describe_outputs(pt_outputs))

        np.testing.assert_allclose(pt_outputs["loss"].detach().cpu().numpy(), pp_outputs["loss"].numpy(), atol=1e-4)
        np.testing.assert_allclose(
            pt_outputs["loss_l1"].detach().cpu().numpy(), pp_outputs["loss_l1"].numpy(), atol=1e-4
        )
        np.testing.assert_allclose(
            pt_outputs["loss_cls"].detach().cpu().numpy(), pp_outputs["loss_cls"].numpy(), atol=1e-4
        )
        np.testing.assert_allclose(
            pt_outputs["loss_iou"].detach().cpu().numpy(), pp_outputs["loss_iou"].numpy(), atol=1e-4
        )
        np.testing.assert_allclose(
            pt_outputs["loss_dfl"].detach().cpu().numpy(), pp_outputs["loss_dfl"].numpy(), atol=1e-4
        )

        # With ATSS assigner
        print("Testing whether PP-Yolo-E head produced same output for ATSS assigner")
        pp_targets["epoch_id"] = 999
        pp_outputs = pp_head(pp_inputs, pp_targets)

        pt_loss.use_static_assigner = False
        pt_head_preds = pt_head(pt_inputs)
        pt_outputs = pt_loss(pt_head_preds, pt_targets)

        print(describe_outputs(pt_outputs))

        np.testing.assert_allclose(pt_outputs["loss"].detach().cpu().numpy(), pp_outputs["loss"].numpy(), atol=1e-4)
        np.testing.assert_allclose(
            pt_outputs["loss_l1"].detach().cpu().numpy(), pp_outputs["loss_l1"].numpy(), atol=1e-4
        )
        np.testing.assert_allclose(
            pt_outputs["loss_cls"].detach().cpu().numpy(), pp_outputs["loss_cls"].numpy(), atol=1e-4
        )
        np.testing.assert_allclose(
            pt_outputs["loss_iou"].detach().cpu().numpy(), pp_outputs["loss_iou"].numpy(), atol=1e-4
        )
        np.testing.assert_allclose(
            pt_outputs["loss_dfl"].detach().cpu().numpy(), pp_outputs["loss_dfl"].numpy(), atol=1e-4
        )

        # In eval mode
        print("Testing whether PP-Yolo-E head produced same output in eval mode")
        pt_head.eval()
        pp_head.eval()

        for exclude_nms in [False, True]:
            for exclude_post_process in [False, True]:
                print(f"Testing with exclude_post_process={exclude_post_process}, exclude_nms={exclude_nms}")
                pt_head.exclude_nms = exclude_nms
                pt_head.exclude_post_process = exclude_post_process

                pp_head.exclude_nms = exclude_nms
                pp_head.exclude_post_process = exclude_post_process

                pt_outputs_eval = pt_head(pt_inputs)
                pp_outputs_eval = pp_head(pp_inputs, pp_targets)

                print(describe_outputs(pt_outputs_eval))
                np.testing.assert_allclose(
                    pt_outputs_eval[0].detach().cpu().numpy(), pp_outputs_eval[0].numpy(), atol=1e-4
                )
                np.testing.assert_allclose(
                    pt_outputs_eval[1].detach().cpu().numpy(), pp_outputs_eval[1].numpy(), atol=1e-4
                )
                np.testing.assert_allclose(
                    pt_outputs_eval[2].detach().cpu().numpy(), pp_outputs_eval[2].numpy(), atol=1e-4
                )
                np.testing.assert_allclose(
                    pt_outputs_eval[3].detach().cpu().numpy(), pp_outputs_eval[3].numpy(), atol=1e-4
                )


if __name__ == "__main__":
    unittest.main()
