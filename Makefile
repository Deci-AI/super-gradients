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
NOTEBOOKS_TO_CHECK := src/super_gradients/examples/model_export/models_export.ipynb
NOTEBOOKS_TO_CHECK += src/super_gradients/examples/model_export/models_export_pose.ipynb
NOTEBOOKS_TO_CHECK += notebooks/what_are_recipes_and_how_to_use.ipynb
NOTEBOOKS_TO_CHECK += notebooks/transfer_learning_classification.ipynb
NOTEBOOKS_TO_CHECK += notebooks/how_to_use_knowledge_distillation_for_classification.ipynb
NOTEBOOKS_TO_CHECK += notebooks/detection_how_to_connect_custom_dataset.ipynb
NOTEBOOKS_TO_CHECK += notebooks/PTQ_and_QAT_for_classification.ipynb
NOTEBOOKS_TO_CHECK += notebooks/quickstart_segmentation.ipynb
NOTEBOOKS_TO_CHECK += notebooks/segmentation_connect_custom_dataset.ipynb
NOTEBOOKS_TO_CHECK += notebooks/transfer_learning_semantic_segmentation.ipynb
NOTEBOOKS_TO_CHECK += notebooks/detection_transfer_learning.ipynb
NOTEBOOKS_TO_CHECK += notebooks/how_to_run_model_predict.ipynb
NOTEBOOKS_TO_CHECK += notebooks/yolo_nas_custom_dataset_fine_tuning_with_qat.ipynb
NOTEBOOKS_TO_CHECK += notebooks/DEKR_PoseEstimationFineTuning.ipynb
NOTEBOOKS_TO_CHECK += notebooks/albumentations_tutorial.ipynb
NOTEBOOKS_TO_CHECK += notebooks/yolo_nas_pose_eval_with_pycocotools.ipynb
NOTEBOOKS_TO_CHECK += notebooks/dataloader_adapter.ipynb


# This Makefile target runs notebooks listed below and converts them to markdown files in documentation/source/
check_notebooks_version_match: $(NOTEBOOKS_TO_CHECK)
	python tests/verify_notebook_version.py $^
