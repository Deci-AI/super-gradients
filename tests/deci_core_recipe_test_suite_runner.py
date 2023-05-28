import sys
import unittest

from tests.recipe_training_tests.automatic_batch_selection_single_gpu_test import TestAutoBatchSelectionSingleGPU
from tests.recipe_training_tests.coded_qat_launch_test import CodedQATLuanchTest
from tests.recipe_training_tests.shortened_recipes_accuracy_test import ShortenedRecipesAccuracyTests


class CoreUnitTestSuiteRunner:
    def __init__(self):
        self.test_loader = unittest.TestLoader()
        self.recipe_tests_suite = unittest.TestSuite()
        self._add_modules_to_unit_tests_suite()
        self.test_runner = unittest.TextTestRunner(verbosity=3, stream=sys.stdout)

    def _add_modules_to_unit_tests_suite(self):
        """
        _add_modules_to_unit_tests_suite - Adds unit tests to the Unit Tests Test Suite
            :return:
        """
        self.recipe_tests_suite.addTest(self.test_loader.loadTestsFromModule(CodedQATLuanchTest))
        self.recipe_tests_suite.addTest(self.test_loader.loadTestsFromModule(ShortenedRecipesAccuracyTests))
        self.recipe_tests_suite.addTest(self.test_loader.loadTestsFromModule(TestAutoBatchSelectionSingleGPU))


if __name__ == "__main__":
    unittest.main()
