import sys
import unittest

from tests.integration_tests import EMAIntegrationTest, LRTest, PoseEstimationModelsIntegrationTest, YoloNASIntegrationTest


class CoreIntegrationTestSuiteRunner:
    def __init__(self):
        self.test_loader = unittest.TestLoader()
        self.integration_tests_suite = unittest.TestSuite()
        self._add_modules_to_integration_tests_suite()
        self.test_runner = unittest.TextTestRunner(verbosity=3, stream=sys.stdout)

    def _add_modules_to_integration_tests_suite(self):
        """
        _add_modules_to_integration_tests_suite - Adds unit tests to the Unit Tests Test Suite
            :return:
        """
        self.integration_tests_suite.addTest(self.test_loader.loadTestsFromModule(EMAIntegrationTest))
        self.integration_tests_suite.addTest(self.test_loader.loadTestsFromModule(LRTest))
        self.integration_tests_suite.addTest(self.test_loader.loadTestsFromModule(PoseEstimationModelsIntegrationTest))
        self.integration_tests_suite.addTest(self.test_loader.loadTestsFromModule(YoloNASIntegrationTest))


if __name__ == "__main__":
    unittest.main()
