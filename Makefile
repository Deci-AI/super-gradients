coco2017_ppyoloe_s:
	python src/super_gradients/examples/train_from_recipe_example/train_from_recipe.py --config-name=coco2017_ppyoloe_s_compile_enabled multi_gpu=Off num_gpus=1
	python src/super_gradients/examples/train_from_recipe_example/train_from_recipe.py --config-name=coco2017_ppyoloe_s_compile_disabled multi_gpu=Off num_gpus=1

coco2017_ssd_lite_mobilenet_v2:
	python src/super_gradients/examples/train_from_recipe_example/train_from_recipe.py --config-name=coco2017_ssd_lite_mobilenet_v2_compile_enabled multi_gpu=Off num_gpus=1
	python src/super_gradients/examples/train_from_recipe_example/train_from_recipe.py --config-name=coco2017_ssd_lite_mobilenet_v2_compile_disabled multi_gpu=Off num_gpus=1

coco2017_yolox:
	python src/super_gradients/examples/train_from_recipe_example/train_from_recipe.py --config-name=coco2017_yolox_compile_enabled multi_gpu=Off num_gpus=1
	python src/super_gradients/examples/train_from_recipe_example/train_from_recipe.py --config-name=coco2017_yolox_compile_disabled multi_gpu=Off num_gpus=1

imagenet_resnet50:
	python src/super_gradients/examples/train_from_recipe_example/train_from_recipe.py --config-name=imagenet_resnet50_compile_enabled multi_gpu=Off num_gpus=1
	python src/super_gradients/examples/train_from_recipe_example/train_from_recipe.py --config-name=imagenet_resnet50_compile_disabled multi_gpu=Off num_gpus=1


cityscapes_ddrnet:
	python src/super_gradients/examples/train_from_recipe_example/train_from_recipe.py --config-name=cityscapes_ddrnet_compile_enabled multi_gpu=Off num_gpus=1
	python src/super_gradients/examples/train_from_recipe_example/train_from_recipe.py --config-name=cityscapes_ddrnet_compile_disabled multi_gpu=Off num_gpus=1
