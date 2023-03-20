#   File "/home/eugene.khvedchenia/super-gradients/src/super_gradients/training/losses/ppyolo_loss.py", line 863, in _bbox_loss
#    pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).reshape([-1, 4])
#RuntimeError: numel: integer multiplication overflow
deciyolo_s_rf100_underwater_objects:
	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=underwater-objects-5v7p8 multi_gpu=Off num_gpus=1


deciyolo_s_rf100_weed-crop-aerial:
	CUDA_VISIBLE_DEVICES=1 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=weed-crop-aerial multi_gpu=Off num_gpus=1


aerial-cows

phages


aquarium-qlnqy


document-parts


peixos-fish


wall-damage

construction-safety-gsnvb


road-signs-6ih4y


wine-labels:
	CUDA_VISIBLE_DEVICES=3 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=wine-labels multi_gpu=Off num_gpus=1
