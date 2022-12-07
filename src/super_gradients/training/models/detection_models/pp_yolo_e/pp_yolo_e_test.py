import collections
import os.path
import unittest

import numpy as np
import paddle
import ppdet
import requests
import torch
from paddle import load as paddle_load
from ppdet.modeling.backbones import CSPResNet as PPCSPResNet
from ppdet.modeling.heads import PPYOLOEHead as Paddle_PPYOLOEHead
from pytorch_toolbelt.utils import count_parameters, describe_outputs
from super_gradients.training.losses.ppyolo_loss import PPYoloELoss
from super_gradients.training.models.detection_models.csp_resnet import CSPResNet
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloE
from super_gradients.training.models.detection_models.pp_yolo_e.pan import CustomCSPPAN
from super_gradients.training.models.detection_models.pp_yolo_e.pp_yolo_head import PPYOLOEHead
from super_gradients.training.utils import HpmStruct
from torch import nn


def convert_weights_from_padldle_to_torch(state_dict: collections.OrderedDict) -> collections.OrderedDict:
    torch_state_dict = []
    import re

    for key, value in state_dict.items():
        torch_value = torch.from_numpy(value.numpy())
        key = key.replace("bn._mean", "bn.running_mean").replace("bn._variance", "bn.running_var").replace("attn.fc", "attn.project")

        key = key.replace("yolo_head", "head")

        key = re.sub(r"stages\.(\d+)\.blocks.(\d+).conv2.conv2.conv.weight", r"stages.\1.blocks.\2.conv2.branch_1x1.conv.weight", key)
        key = re.sub(r"stages\.(\d+)\.blocks.(\d+).conv2.conv2.bn", r"stages.\1.blocks.\2.conv2.branch_1x1.bn", key)
        key = re.sub(r"stages\.(\d+)\.blocks.(\d+).conv2.conv1.conv.weight", r"stages.\1.blocks.\2.conv2.branch_3x3.conv.weight", key)
        key = re.sub(r"stages\.(\d+)\.blocks.(\d+).conv2.conv1.bn", r"stages.\1.blocks.\2.conv2.branch_3x3.bn", key)

        key = key.replace("conv_down.bn", "conv_down.seq.bn")
        key = key.replace("conv_down.conv", "conv_down.seq.conv")

        key = key.replace("conv1.bn", "conv1.seq.bn")
        key = key.replace("conv1.conv", "conv1.seq.conv")

        key = key.replace("conv2.bn", "conv2.seq.bn")
        key = key.replace("conv2.conv", "conv2.seq.conv")

        key = key.replace("conv3.bn", "conv3.seq.bn")
        key = key.replace("conv3.conv", "conv3.seq.conv")

        torch_state_dict.append((key, torch_value))

    return collections.OrderedDict(torch_state_dict)


class PPYoloTestCast(unittest.TestCase):
    def download_file_if_needed(self, url):
        local_filename = url.split("/")[-1]
        if os.path.exists(local_filename):
            return
        # NOTE the stream=True parameter below
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    # If you have chunk encoded response uncomment if
                    # and set chunk_size parameter to None.
                    # if chunk:
                    f.write(chunk)
        return local_filename

    def test_csp_resnet(self):
        self.download_file_if_needed("https://paddledet.bj.bcebos.com/models/pretrained/CSPResNetb_s_pretrained.pdparams")
        self.download_file_if_needed("https://paddledet.bj.bcebos.com/models/pretrained/CSPResNetb_x_pretrained.pdparams")
        self.download_file_if_needed("https://paddledet.bj.bcebos.com/models/pretrained/CSPResNetb_m_pretrained.pdparams")
        self.download_file_if_needed("https://paddledet.bj.bcebos.com/models/pretrained/CSPResNetb_l_pretrained.pdparams")

        input_np = np.random.standard_normal((4, 3, 640, 512)).astype(np.float32)

        common_params = dict(
            layers=(3, 6, 6, 3), channels=(64, 128, 256, 512, 1024), activation="silu", return_idx=(1, 2, 3), use_alpha=False, use_large_stem=True
        )
        for config, pretrain in [
            (
                dict(depth_mult=0.33, width_mult=0.50, **common_params),
                "CSPResNetb_s_pretrained.pdparams",
            ),
            (
                dict(depth_mult=0.67, width_mult=0.75, **common_params),
                "CSPResNetb_m_pretrained.pdparams",
            ),
            (
                dict(depth_mult=1, width_mult=1, **common_params),
                "CSPResNetb_l_pretrained.pdparams",
            ),
            (
                dict(depth_mult=1.33, width_mult=1.25, **common_params),
                "CSPResNetb_x_pretrained.pdparams",
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

            torch_weights = convert_weights_from_padldle_to_torch(paddle_weights)
            torch_path = os.path.splitext(pretrain)[0] + ".pth"
            torch.save(torch_weights, torch_path)
            print("Saved weights to torch_path")

            pt_encoder = CSPResNet(**config, pretrained_weights=torch_path).eval()

            input_pt = torch.from_numpy(input_np)
            pt_output = pt_encoder(input_pt)

            print(count_parameters(pt_encoder, human_friendly=True))
            print(describe_outputs(pt_output))

            np.testing.assert_allclose(pt_output[0].detach().cpu().numpy(), pp_output[0].numpy(), atol=1e-4, rtol=1e-3)
            np.testing.assert_allclose(pt_output[1].detach().cpu().numpy(), pp_output[1].numpy(), atol=1e-4, rtol=1e-3)
            np.testing.assert_allclose(pt_output[2].detach().cpu().numpy(), pp_output[2].numpy(), atol=1e-4, rtol=1e-3)

            print("Testing prep_model_for_conversion")
            pt_encoder.prep_model_for_conversion(input_size=input_np.shape[2:])
            pt_output = pt_encoder(input_pt)
            print(count_parameters(pt_encoder, human_friendly=True))
            print(describe_outputs(pt_output))

            np.testing.assert_allclose(pt_output[0].detach().cpu().numpy(), pp_output[0].numpy(), atol=1e-4, rtol=1e-3)
            np.testing.assert_allclose(pt_output[1].detach().cpu().numpy(), pp_output[1].numpy(), atol=1e-4, rtol=1e-3)
            np.testing.assert_allclose(pt_output[2].detach().cpu().numpy(), pp_output[2].numpy(), atol=1e-4, rtol=1e-3)

            print("Testing traced model")
            with torch.no_grad():
                pt_encoder_traced = torch.jit.trace(pt_encoder, example_inputs=input_pt)
            pt_output = pt_encoder_traced(input_pt)
            print(describe_outputs(pt_output))

            np.testing.assert_allclose(pt_output[0].detach().cpu().numpy(), pp_output[0].numpy(), atol=1e-4, rtol=1e-3)
            np.testing.assert_allclose(pt_output[1].detach().cpu().numpy(), pp_output[1].numpy(), atol=1e-4, rtol=1e-3)
            np.testing.assert_allclose(pt_output[2].detach().cpu().numpy(), pp_output[2].numpy(), atol=1e-4, rtol=1e-3)

    def test_ppyolo_conversion(self):
        self.download_file_if_needed("https://paddledet.bj.bcebos.com/models/ppyoloe_crn_m_300e_coco.pdparams")
        self.download_file_if_needed("https://paddledet.bj.bcebos.com/models/ppyoloe_crn_s_400e_coco.pdparams")
        self.download_file_if_needed("https://paddledet.bj.bcebos.com/models/ppyoloe_crn_l_300e_coco.pdparams")
        self.download_file_if_needed("https://paddledet.bj.bcebos.com/models/ppyoloe_crn_x_300e_coco.pdparams")

        common_params = dict(
            num_classes=80,
            activation="silu",
            normalization=None,
            backbone=dict(
                activation="silu",
                layers=(3, 6, 6, 3),
                channels=(64, 128, 256, 512, 1024),
                return_idx=(1, 2, 3),
                use_alpha=False,
                use_large_stem=True,
            ),
            neck=dict(
                in_channels=[256, 512, 1024],
                out_channels=[1024, 512, 256],
                activation="silu",
                block_num=3,
                stage_num=1,
                spp=False,
            ),
            head=dict(
                num_classes=80,
                in_channels=[1024, 512, 256],
                fpn_strides=[32, 16, 8],
                grid_cell_scale=5.0,
                grid_cell_offset=0.5,
                activation="silu",
                reg_max=16,  # Number of bins for size prediction
                eval_size=None,
                exclude_nms=False,
                exclude_post_process=False,
            ),
        )
        for config, pretrain in [
            (
                dict(depth_mult=0.33, width_mult=0.50, **common_params),
                "ppyoloe_crn_s_400e_coco.pdparams",
            ),
            (
                dict(depth_mult=0.67, width_mult=0.75, **common_params),
                "ppyoloe_crn_m_300e_coco.pdparams",
            ),
            (
                dict(depth_mult=1, width_mult=1, **common_params),
                "ppyoloe_crn_l_300e_coco.pdparams",
            ),
            (
                dict(depth_mult=1.33, width_mult=1.25, **common_params),
                "ppyoloe_crn_x_300e_coco.pdparams",
            ),
        ]:
            paddle_weights = paddle_load(pretrain)

            torch_weights = convert_weights_from_padldle_to_torch(paddle_weights)
            torch_path = os.path.splitext(pretrain)[0] + ".pth"
            torch.save(torch_weights, torch_path)
            print("Saved weights to torch_path")

            config["backbone"]["width_mult"] = config["width_mult"]
            config["backbone"]["depth_mult"] = config["depth_mult"]

            config["neck"]["width_mult"] = config["width_mult"]
            config["neck"]["depth_mult"] = config["depth_mult"]

            config["head"]["width_mult"] = config["width_mult"]

            pt_encoder = PPYoloE(HpmStruct(**config)).eval()
            pt_encoder.load_state_dict(torch_weights)

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
            neck_weights = collections.OrderedDict([(k.replace("neck.", ""), v) for (k, v) in paddle_weights.items() if k.startswith("neck")])

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
            nms=None,
        )

        pt_head = PPYOLOEHead(
            in_channels=(384, 192, 96),
            num_classes=80,
            fpn_strides=(32, 16, 8),
            grid_cell_scale=5.0,
            grid_cell_offset=0.5,
        )
        pt_head.train().cuda()
        pt_loss = PPYoloELoss(num_classes=pt_head.num_classes, use_static_assigner=True).cuda()

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
        np.testing.assert_allclose(pt_outputs["loss_l1"].detach().cpu().numpy(), pp_outputs["loss_l1"].numpy(), atol=1e-4)
        np.testing.assert_allclose(pt_outputs["loss_cls"].detach().cpu().numpy(), pp_outputs["loss_cls"].numpy(), atol=1e-4)
        np.testing.assert_allclose(pt_outputs["loss_iou"].detach().cpu().numpy(), pp_outputs["loss_iou"].numpy(), atol=1e-4)
        np.testing.assert_allclose(pt_outputs["loss_dfl"].detach().cpu().numpy(), pp_outputs["loss_dfl"].numpy(), atol=1e-4)

        # With ATSS assigner
        print("Testing whether PP-Yolo-E head produced same output for ATSS assigner")
        pp_targets["epoch_id"] = 999
        pp_outputs = pp_head(pp_inputs, pp_targets)

        pt_loss.use_static_assigner = False
        pt_head_preds = pt_head(pt_inputs)
        pt_outputs = pt_loss(pt_head_preds, pt_targets)

        print(describe_outputs(pt_outputs))

        np.testing.assert_allclose(pt_outputs["loss"].detach().cpu().numpy(), pp_outputs["loss"].numpy(), atol=1e-4)
        np.testing.assert_allclose(pt_outputs["loss_l1"].detach().cpu().numpy(), pp_outputs["loss_l1"].numpy(), atol=1e-4)
        np.testing.assert_allclose(pt_outputs["loss_cls"].detach().cpu().numpy(), pp_outputs["loss_cls"].numpy(), atol=1e-4)
        np.testing.assert_allclose(pt_outputs["loss_iou"].detach().cpu().numpy(), pp_outputs["loss_iou"].numpy(), atol=1e-4)
        np.testing.assert_allclose(pt_outputs["loss_dfl"].detach().cpu().numpy(), pp_outputs["loss_dfl"].numpy(), atol=1e-4)

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
                np.testing.assert_allclose(pt_outputs_eval[0].detach().cpu().numpy(), pp_outputs_eval[0].numpy(), atol=1e-4)
                np.testing.assert_allclose(pt_outputs_eval[1].detach().cpu().numpy(), pp_outputs_eval[1].numpy(), atol=1e-4)
                np.testing.assert_allclose(pt_outputs_eval[2].detach().cpu().numpy(), pp_outputs_eval[2].numpy(), atol=1e-4)
                np.testing.assert_allclose(pt_outputs_eval[3].detach().cpu().numpy(), pp_outputs_eval[3].numpy(), atol=1e-4)


if __name__ == "__main__":
    unittest.main()
