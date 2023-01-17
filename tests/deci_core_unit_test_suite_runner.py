import sys
import unittest


class CoreUnitTestSuiteRunner:
    def __init__(self):
        self.loader = unittest.TestLoader()

        self.unit_tests_suite = self.loader.discover("./unit_tests")

        self.test_runner = unittest.TextTestRunner(verbosity=3, stream=sys.stdout)


if __name__ == "__main__":
    unittest.main()
