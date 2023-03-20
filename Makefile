deciyolo_s_rf100_underwater_objects:
	CUDA_VISIBLE_DEVICES=1 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=underwater-objects-5v7p8 multi_gpu=Off num_gpus=1
