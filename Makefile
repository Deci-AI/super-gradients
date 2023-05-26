unit_tests:
	python -m unittest tests/deci_core_unit_test_suite_runner.py

integration_tests:
	python -m unittest tests/deci_core_integration_test_suite_runner.py

yolo_nas_integration_tests:
	python -m unittest tests/integration_tests/yolo_nas_integration_test.py
