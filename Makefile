unit_tests:
	python -m unittest tests/deci_core_unit_test_suite_runner.py

integration_tests:
	python -m unittest tests/deci_core_integration_test_suite_runner.py

yolo_nas_integration_tests:
	python -m unittest tests/integration_tests/yolo_nas_integration_test.py

recipe_accuracy_tests:
	python src/super_gradients/train_from_recipe.py --config-name=coco2017_pose_dekr_w32_no_dc experiment_name=shortened_coco2017_pose_dekr_w32_ap_test epochs=1 batch_size=4 val_batch_size=8 training_hyperparams.lr_warmup_steps=0 training_hyperparams.average_best_models=False training_hyperparams.max_train_batches=1000 training_hyperparams.max_valid_batches=100 multi_gpu=DDP num_gpus=4
	python src/super_gradients/train_from_recipe.py --config-name=cifar10_resnet               experiment_name=shortened_cifar10_resnet_accuracy_test   epochs=100 training_hyperparams.average_best_models=False multi_gpu=DDP num_gpus=4
	python src/super_gradients/train_from_recipe.py --config-name=coco2017_yolox               experiment_name=shortened_coco2017_yolox_n_map_test      epochs=10  architecture=yolox_n training_hyperparams.loss=YoloXFastDetectionLoss training_hyperparams.average_best_models=False multi_gpu=DDP num_gpus=4
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

# Here you define a list of notebooks we want to execute and convert to markdown files
NOTEBOOKS_TO_RUN := src/super_gradients/examples/model_export/models_export.ipynb
NOTEBOOKS_TO_RUN += src/super_gradients/examples/model_export/models_export_pose.ipynb
NOTEBOOKS_TO_RUN += notebooks/what_are_recipes_and_how_to_use.ipynb
NOTEBOOKS_TO_RUN += notebooks/transfer_learning_classification.ipynb
NOTEBOOKS_TO_RUN += notebooks/how_to_use_knowledge_distillation_for_classification.ipynb
NOTEBOOKS_TO_RUN += notebooks/PTQ_and_QAT_for_classification.ipynb

# If there are additional notebooks that must not be executed, but still should be checked for version match, add them here
NOTEBOOKS_TO_CHECK := $(NOTEBOOKS_TO_RUN)
NOTEBOOKS_TO_CHECK += notebooks/yolo_nas_pose_eval_with_pycocotools.ipynb

# This Makefile target runs notebooks listed below and converts them to markdown files in documentation/source/
run_and_convert_notebooks_to_docs: $(NOTEBOOKS_TO_RUN)
	jupyter nbconvert --to markdown --output-dir="documentation/source/" --execute $^

# This Makefile target runs notebooks listed below and converts them to markdown files in documentation/source/
check_notebooks_version_match: $(NOTEBOOKS_TO_CHECK)
	python tests/verify_notebook_version.py $^

WANDB_PARAMS = training_hyperparams.sg_logger=wandb_sg_logger +training_hyperparams.sg_logger_params.api_server=https://wandb.research.deci.ai +training_hyperparams.sg_logger_params.entity=super-gradients training_hyperparams.sg_logger_params.launch_tensorboard=false training_hyperparams.sg_logger_params.monitor_system=true +training_hyperparams.sg_logger_params.project_name=PoseEstimation training_hyperparams.sg_logger_params.save_checkpoints_remote=true training_hyperparams.sg_logger_params.save_logs_remote=true training_hyperparams.sg_logger_params.save_tensorboard_remote=false training_hyperparams.sg_logger_params.tb_files_user_prompt=false

coco2017_yolo_nas_pose_n_multiscale:
	python src/super_gradients/train_from_recipe.py --config-name=coco2017_yolo_nas_pose_n_multiscale $(WANDB_PARAMS)

coco2017_yolo_nas_pose_s_multiscale:
	python src/super_gradients/train_from_recipe.py --config-name=coco2017_yolo_nas_pose_s_multiscale $(WANDB_PARAMS)

coco2017_yolo_nas_pose_s_multiscale_light:
	python src/super_gradients/train_from_recipe.py --config-name=coco2017_yolo_nas_pose_s_multiscale_light $(WANDB_PARAMS)

coco2017_yolo_nas_pose_s_multiscale_resume:
	python src/super_gradients/train_from_recipe.py --config-name=coco2017_yolo_nas_pose_s_multiscale $(WANDB_PARAMS) resume=True

coco2017_yolo_nas_pose_m_multiscale:
	python src/super_gradients/train_from_recipe.py --config-name=coco2017_yolo_nas_pose_m_multiscale $(WANDB_PARAMS)

coco2017_yolo_nas_pose_l_multiscale:
	python src/super_gradients/train_from_recipe.py --config-name=coco2017_yolo_nas_pose_l_multiscale $(WANDB_PARAMS)

coco2017_yolo_nas_pose_s_multiscale_light_local:
	python src/super_gradients/train_from_recipe.py --config-name=coco2017_yolo_nas_pose_s_multiscale_light \
    num_gpus=4 \
    dataset_params.train_dataloader_params.num_workers=16 \
    +dataset_params.train_dataloader_params.prefetch_factor=8 \
    +dataset_params.val_dataloader_params.prefetch_factor=8 \
    dataset_params.train_dataset_params.data_dir=/home/bloodaxe/data/coco2017 dataset_params.val_dataset_params.data_dir=/home/bloodaxe/data/coco2017
