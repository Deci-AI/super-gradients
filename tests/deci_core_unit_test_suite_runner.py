import sys
import unittest


if __name__ == "__main__":
    unit_tests_suite = unittest.TestLoader().discover("./unit_tests/")
    test_runner = unittest.TextTestRunner(verbosity=3, stream=sys.stdout)
    test_runner.run(unit_tests_suite)
