cells-uyemf:
	CUDA_VISIBLE_DEVICES=0 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=coins-1apki
	CUDA_VISIBLE_DEVICES=1 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=coins-1apki
	CUDA_VISIBLE_DEVICES=2 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=coins-1apki
	CUDA_VISIBLE_DEVICES=3 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=coins-1apki
	CUDA_VISIBLE_DEVICES=4 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=coins-1apki
	CUDA_VISIBLE_DEVICES=5 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=coins-1apki
	CUDA_VISIBLE_DEVICES=6 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=coins-1apki
	CUDA_VISIBLE_DEVICES=7 python -m super_gradients.train_from_recipe --config-name=roboflow_deciyolo_s dataset_name=coins-1apki
