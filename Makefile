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

WANDB_PARAMS = training_hyperparams.sg_logger=wandb_sg_logger +training_hyperparams.sg_logger_params.api_server=https://wandb.research.deci.ai +training_hyperparams.sg_logger_params.entity=super-gradients training_hyperparams.sg_logger_params.launch_tensorboard=false training_hyperparams.sg_logger_params.monitor_system=true +training_hyperparams.sg_logger_params.project_name=PoseEstimation training_hyperparams.sg_logger_params.save_checkpoints_remote=true training_hyperparams.sg_logger_params.save_logs_remote=true training_hyperparams.sg_logger_params.save_tensorboard_remote=false training_hyperparams.sg_logger_params.tb_files_user_prompt=false

examples_to_docs:
	jupyter nbconvert --to markdown --output-dir="documentation/source/" --execute src/super_gradients/examples/model_export/models_export.ipynb

coco2017_yolo_nas_pose_s:
	python src/super_gradients/train_from_recipe.py --config-name=coco2017_yolo_nas_pose_s $(WANDB_PARAMS)

coco2017_yolo_nas_pose_m:
	python src/super_gradients/train_from_recipe.py --config-name=coco2017_yolo_nas_pose_m $(WANDB_PARAMS)

coco2017_yolo_nas_pose_m_resume:
	python src/super_gradients/train_from_recipe.py --config-name=coco2017_yolo_nas_pose_m $(WANDB_PARAMS) resume=True

coco2017_yolo_nas_pose_l:
	python src/super_gradients/train_from_recipe.py --config-name=coco2017_yolo_nas_pose_l $(WANDB_PARAMS)

coco2017_yolo_nas_pose_l_no_ema:
	python src/super_gradients/train_from_recipe.py --config-name=coco2017_yolo_nas_pose_l $(WANDB_PARAMS) training_hyperparams.ema=False

coco2017_yolo_nas_pose_n:
	python src/super_gradients/train_from_recipe.py --config-name=coco2017_yolo_nas_pose_n $(WANDB_PARAMS)

coco2017_yolo_nas_pose_n_resume:
	python src/super_gradients/train_from_recipe.py --config-name=coco2017_yolo_nas_pose_n $(WANDB_PARAMS) resume=True

coco2017_yolo_nas_pose_l_resume:
	python src/super_gradients/train_from_recipe.py --config-name=coco2017_yolo_nas_pose_l $(WANDB_PARAMS) resume=True


LOCAL_TRAINING_PARAMS = dataset_params.train_dataset_params.data_dir=/home/bloodaxe/data/crowdpose dataset_params.val_dataset_params.data_dir=/home/bloodaxe/data/crowdpose num_gpus=1 multi_gpu=Off

crowdpose_yolo_nas_pose_s:
	#	CUDA_VISIBLE_DEVICES=0 python src/super_gradients/train_from_recipe.py --config-name=crowdpose_yolo_nas_pose_s $(LOCAL_TRAINING_PARAMS) &
	#	CUDA_VISIBLE_DEVICES=1 python src/super_gradients/train_from_recipe.py --config-name=crowdpose_yolo_nas_pose_s $(LOCAL_TRAINING_PARAMS) training_hyperparams.criterion_params.classification_loss_type=bce &
	#	CUDA_VISIBLE_DEVICES=2 python src/super_gradients/train_from_recipe.py --config-name=crowdpose_yolo_nas_pose_s $(LOCAL_TRAINING_PARAMS) training_hyperparams.criterion_params.dfl_loss_weight=1.0 &
	#	CUDA_VISIBLE_DEVICES=3 python src/super_gradients/train_from_recipe.py --config-name=crowdpose_yolo_nas_pose_s $(LOCAL_TRAINING_PARAMS) training_hyperparams.criterion_params.pose_cls_loss_weight=0.1 training_hyperparams.criterion_params.pose_reg_loss_weight=5.0 &

	CUDA_VISIBLE_DEVICES=0 python src/super_gradients/train_from_recipe.py --config-name=crowdpose_yolo_nas_pose_s $(LOCAL_TRAINING_PARAMS) training_hyperparams.criterion_params.pose_cls_loss_weight=0.10 training_hyperparams.criterion_params.pose_reg_loss_weight=5.0 training_hyperparams.criterion_params.dfl_loss_weight=1.0 &
	CUDA_VISIBLE_DEVICES=1 python src/super_gradients/train_from_recipe.py --config-name=crowdpose_yolo_nas_pose_s $(LOCAL_TRAINING_PARAMS) training_hyperparams.criterion_params.pose_cls_loss_weight=0.01 training_hyperparams.criterion_params.pose_reg_loss_weight=5.0 training_hyperparams.criterion_params.dfl_loss_weight=1.0 &
	CUDA_VISIBLE_DEVICES=2 python src/super_gradients/train_from_recipe.py --config-name=crowdpose_yolo_nas_pose_s $(LOCAL_TRAINING_PARAMS) training_hyperparams.criterion_params.pose_cls_loss_weight=0.01 training_hyperparams.criterion_params.pose_reg_loss_weight=1.0 training_hyperparams.criterion_params.dfl_loss_weight=2.5 &
	CUDA_VISIBLE_DEVICES=3 python src/super_gradients/train_from_recipe.py --config-name=crowdpose_yolo_nas_pose_s $(LOCAL_TRAINING_PARAMS) training_hyperparams.criterion_params.pose_cls_loss_weight=0.10 training_hyperparams.criterion_params.pose_reg_loss_weight=5.0 training_hyperparams.criterion_params.dfl_loss_weight=1.0 training_hyperparams.criterion_params.classification_loss_type=bce &
