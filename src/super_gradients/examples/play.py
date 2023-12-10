from random import seed

import numpy as np
import super_gradients
import torch

torch.cuda.manual_seed(1)
torch.manual_seed(1)
np.random.seed(1)
seed(1)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


def test_yolo_nas_backbone_with_fe_batch_consistency():
    device = torch.device("cuda")
    yolo_nas = super_gradients.training.models.get("yolo_nas_l", pretrained_weights="coco").to(device)
    yolo_nas.prep_model_for_conversion(full_fusion=True)
    yolo_nas.eval()
    random_im = np.random.randint(low=0, high=255, size=(1, 3, 640 * 8, 640 * 8), dtype=np.uint8)
    batch_input_M = torch.tensor(np.concatenate([random_im for _ in range(1)], axis=0)).to(dtype=torch.float32).to(device)
    batch_input_N = torch.tensor(np.concatenate([random_im for _ in range(2)], axis=0)).to(dtype=torch.float32).to(device)
    batch_output_per_layer_M = yolo_nas.backbone(batch_input_M)
    batch_output_per_layer_N = yolo_nas.backbone(batch_input_N)
    for batch_output_M, batch_output_N in zip(batch_output_per_layer_M, batch_output_per_layer_N):
        print(f"MEAN ERROR  {(torch.abs(batch_output_M[0] - batch_output_N[0])).mean()}")
        print(f"MAX ERROR  {(torch.abs(batch_output_M[0] - batch_output_N[0])).max()}")
        # print(f"ERROR COUNT{(batch_output_M[0] != batch_output_N[0]).sum()}")
        # print(batch_output_M[0] != batch_output_N[0])


if __name__ == "__main__":
    test_yolo_nas_backbone_with_fe_batch_consistency()
