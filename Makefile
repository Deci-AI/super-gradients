unit_tests:
	python -m unittest tests/deci_core_unit_test_suite_runner.py

integration_tests:
	python -m unittest tests/deci_core_integration_test_suite_runner.py

yolo_nas_integration_tests:
	python -m unittest tests/integration_tests/yolo_nas_integration_test.py

recipe_accuracy_tests:
	python src/super_gradients/train_from_recipe.py --config-name=coco2017_pose_dekr_w32_no_dc experiment_name=shortened_coco2017_pose_dekr_w32_ap_test epochs=1 batch_size=4 val_batch_size=8 training_hyperparams.lr_warmup_steps=0 training_hyperparams.average_best_models=False training_hyperparams.max_train_batches=1000 training_hyperparams.max_valid_batches=100 multi_gpu=DDP num_gpus=4
	python src/super_gradients/train_from_recipe.py --config-name=cifar10_resnet               experiment_name=shortened_cifar10_resnet_accuracy_test   epochs=100 training_hyperparams.average_best_models=False multi_gpu=DDP num_gpus=4
	python src/super_gradients/train_from_recipe.py --config-name=coco2017_yolox               experiment_name=shortened_coco2017_yolox_n_map_test      epochs=10  architecture=yolox_n training_hyperparams.loss=yolox_fast_loss training_hyperparams.average_best_models=False multi_gpu=DDP num_gpus=4
	python src/super_gradients/train_from_recipe.py --config-name=cityscapes_regseg48          experiment_name=shortened_cityscapes_regseg48_iou_test   epochs=10 training_hyperparams.average_best_models=False multi_gpu=DDP num_gpus=4
	python src/super_gradients/examples/convert_recipe_example/convert_recipe_example.py --config-name=cifar10_conversion_params experiment_name=shortened_cifar10_resnet_accuracy_test
	coverage run --source=super_gradients -m unittest tests/deci_core_recipe_test_suite_runner.py


sweeper_test:
	python -m super_gradients.train_from_recipe -m --config-name=cifar10_resnet \
	  ckpt_root_dir=$$PWD \
	  experiment_name=sweep_cifar10 \
	  training_hyperparams.max_epochs=1 \
	  training_hyperparams.initial_lr=0.001,0.01

	# Make sure that experiment_dir includes $$expected_num_dir subdirectories. If not, fail
	subdir_count=$$(find "$$PWD/sweep_cifar10" -mindepth 1 -maxdepth 1 -type d | wc -l); \
	if [ "$$subdir_count" -ne 2 ]; then \
	  echo "Error: $$PWD/sweep_cifar10 should include 2 subdirectories but includes $$subdir_count."; \
	  exit 1; \
	fi


examples_to_docs:
	jupyter nbconvert --to markdown --output-dir="documentation/source/" --execute src/super_gradients/examples/model_export/models_export.ipynb

coco2017_yolo_nas_pose_s:
	python src/super_gradients/train_from_recipe.py --config-name=coco2017_yolo_nas_pose_s_weights_and_biases training_hyperparams.sync_bn=False training_hyperparams.initial_lr=1e-3

coco2017_yolo_nas_pose_s_mixup:
	python src/super_gradients/train_from_recipe.py --config-name=coco2017_yolo_nas_pose_s_mixup_weights_and_biases training_hyperparams.sync_bn=False training_hyperparams.initial_lr=1e-3

coco2017_yolo_nas_pose_m:
	python src/super_gradients/train_from_recipe.py --config-name=coco2017_yolo_nas_pose_m_weights_and_biases dataset_params.val_dataset_params.data_dir=/data/coco
	torchrun --standalone --nnodes=1 --nproc_per_node=8 src/super_gradients/train_from_recipe.py --config-name=coco2017_yolo_nas_pose_m_weights_and_biases dataset_params.val_dataset_params.data_dir=/data/coco
	CUDA_VISIBLE_DEVICES=6 python src/super_gradients/train_from_recipe.py --config-name=coco2017_yolo_nas_pose_m_weights_and_biases dataset_params.val_dataset_params.data_dir=/data/coco multi_gpu=Off num_gpus=1

coco2017_yolo_nas_pose_l:
	python src/super_gradients/train_from_recipe.py --config-name=coco2017_yolo_nas_pose_l_weights_and_biases batch_size=24 training_hyperparams.sync_bn=False

coco2017_yolo_nas_pose_l_bce:
	python src/super_gradients/train_from_recipe.py --config-name=coco2017_yolo_nas_pose_l_bce_weights_and_biases batch_size=24 training_hyperparams.sync_bn=False

# --config-name=animalpose_yolo_nas_pose_s
  #dataset_params.train_dataset_params.data_dir=g:/animalpose
  #dataset_params.val_dataset_params.data_dir=g:/animalpose
  #dataset_params.train_dataloader_params.num_workers=8
  #dataset_params.val_dataloader_params.num_workers=8
  #+dataset_params.train_dataloader_params.persistent_workers=True
  #+dataset_params.val_dataloader_params.persistent_workers=True
  #multi_gpu=Off
  #num_gpus=1

animalpose_gridsearch:
	CUDA_VISIBLE_DEVICES=0 python src/super_gradients/train_from_recipe.py --config-name=animalpose_yolo_nas_pose_s_grid_search training_hyperparams.criterion_params.classification_loss_type=bce    training_hyperparams.criterion_params.use_cocoeval_formula=True  training_hyperparams.criterion_params.rescale_keypoint_loss_by_assigned_weight=True  &
	CUDA_VISIBLE_DEVICES=1 python src/super_gradients/train_from_recipe.py --config-name=animalpose_yolo_nas_pose_s_grid_search training_hyperparams.criterion_params.classification_loss_type=bce    training_hyperparams.criterion_params.use_cocoeval_formula=True  training_hyperparams.criterion_params.rescale_keypoint_loss_by_assigned_weight=False &
	CUDA_VISIBLE_DEVICES=2 python src/super_gradients/train_from_recipe.py --config-name=animalpose_yolo_nas_pose_s_grid_search training_hyperparams.criterion_params.classification_loss_type=bce    training_hyperparams.criterion_params.use_cocoeval_formula=False training_hyperparams.criterion_params.rescale_keypoint_loss_by_assigned_weight=True  &
	CUDA_VISIBLE_DEVICES=3 python src/super_gradients/train_from_recipe.py --config-name=animalpose_yolo_nas_pose_s_grid_search training_hyperparams.criterion_params.classification_loss_type=bce    training_hyperparams.criterion_params.use_cocoeval_formula=False training_hyperparams.criterion_params.rescale_keypoint_loss_by_assigned_weight=False &
	CUDA_VISIBLE_DEVICES=4 python src/super_gradients/train_from_recipe.py --config-name=animalpose_yolo_nas_pose_s_grid_search training_hyperparams.criterion_params.classification_loss_type=focal  training_hyperparams.criterion_params.use_cocoeval_formula=True  training_hyperparams.criterion_params.rescale_keypoint_loss_by_assigned_weight=True  &
	CUDA_VISIBLE_DEVICES=5 python src/super_gradients/train_from_recipe.py --config-name=animalpose_yolo_nas_pose_s_grid_search training_hyperparams.criterion_params.classification_loss_type=focal  training_hyperparams.criterion_params.use_cocoeval_formula=True  training_hyperparams.criterion_params.rescale_keypoint_loss_by_assigned_weight=False &
	CUDA_VISIBLE_DEVICES=6 python src/super_gradients/train_from_recipe.py --config-name=animalpose_yolo_nas_pose_s_grid_search training_hyperparams.criterion_params.classification_loss_type=focal  training_hyperparams.criterion_params.use_cocoeval_formula=False training_hyperparams.criterion_params.rescale_keypoint_loss_by_assigned_weight=True  &
	CUDA_VISIBLE_DEVICES=7 python src/super_gradients/train_from_recipe.py --config-name=animalpose_yolo_nas_pose_s_grid_search training_hyperparams.criterion_params.classification_loss_type=focal  training_hyperparams.criterion_params.use_cocoeval_formula=False training_hyperparams.criterion_params.rescale_keypoint_loss_by_assigned_weight=False &

animalpose_gridsearch_ciou:
	CUDA_VISIBLE_DEVICES=0 python src/super_gradients/train_from_recipe.py --config-name=animalpose_yolo_nas_pose_s_grid_search training_hyperparams.criterion_params.classification_loss_type=bce    training_hyperparams.criterion_params.use_cocoeval_formula=True  training_hyperparams.criterion_params.rescale_keypoint_loss_by_assigned_weight=True  training_hyperparams.criterion_params.regression_iou_loss_type=ciou &
	CUDA_VISIBLE_DEVICES=1 python src/super_gradients/train_from_recipe.py --config-name=animalpose_yolo_nas_pose_s_grid_search training_hyperparams.criterion_params.classification_loss_type=bce    training_hyperparams.criterion_params.use_cocoeval_formula=True  training_hyperparams.criterion_params.rescale_keypoint_loss_by_assigned_weight=False training_hyperparams.criterion_params.regression_iou_loss_type=ciou &
	CUDA_VISIBLE_DEVICES=2 python src/super_gradients/train_from_recipe.py --config-name=animalpose_yolo_nas_pose_s_grid_search training_hyperparams.criterion_params.classification_loss_type=bce    training_hyperparams.criterion_params.use_cocoeval_formula=False training_hyperparams.criterion_params.rescale_keypoint_loss_by_assigned_weight=True  training_hyperparams.criterion_params.regression_iou_loss_type=ciou &
	CUDA_VISIBLE_DEVICES=3 python src/super_gradients/train_from_recipe.py --config-name=animalpose_yolo_nas_pose_s_grid_search training_hyperparams.criterion_params.classification_loss_type=bce    training_hyperparams.criterion_params.use_cocoeval_formula=False training_hyperparams.criterion_params.rescale_keypoint_loss_by_assigned_weight=False training_hyperparams.criterion_params.regression_iou_loss_type=ciou &
	CUDA_VISIBLE_DEVICES=4 python src/super_gradients/train_from_recipe.py --config-name=animalpose_yolo_nas_pose_s_grid_search training_hyperparams.criterion_params.classification_loss_type=focal  training_hyperparams.criterion_params.use_cocoeval_formula=True  training_hyperparams.criterion_params.rescale_keypoint_loss_by_assigned_weight=True  training_hyperparams.criterion_params.regression_iou_loss_type=ciou &
	CUDA_VISIBLE_DEVICES=5 python src/super_gradients/train_from_recipe.py --config-name=animalpose_yolo_nas_pose_s_grid_search training_hyperparams.criterion_params.classification_loss_type=focal  training_hyperparams.criterion_params.use_cocoeval_formula=True  training_hyperparams.criterion_params.rescale_keypoint_loss_by_assigned_weight=False training_hyperparams.criterion_params.regression_iou_loss_type=ciou &
	CUDA_VISIBLE_DEVICES=6 python src/super_gradients/train_from_recipe.py --config-name=animalpose_yolo_nas_pose_s_grid_search training_hyperparams.criterion_params.classification_loss_type=focal  training_hyperparams.criterion_params.use_cocoeval_formula=False training_hyperparams.criterion_params.rescale_keypoint_loss_by_assigned_weight=True  training_hyperparams.criterion_params.regression_iou_loss_type=ciou &
	CUDA_VISIBLE_DEVICES=7 python src/super_gradients/train_from_recipe.py --config-name=animalpose_yolo_nas_pose_s_grid_search training_hyperparams.criterion_params.classification_loss_type=focal  training_hyperparams.criterion_params.use_cocoeval_formula=False training_hyperparams.criterion_params.rescale_keypoint_loss_by_assigned_weight=False training_hyperparams.criterion_params.regression_iou_loss_type=ciou &
