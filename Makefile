deci_yolo_s:
	python src/super_gradients/train_from_recipe.py --config-name=coco2017_deciyolo_s


test_deci_yolo:
	python -m unittest tests/integration_tests/deci_yolo_test.py
