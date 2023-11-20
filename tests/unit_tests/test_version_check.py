import unittest

from tests.verify_notebook_version import try_extract_super_gradients_version_from_pip_install_command


class TestVersionCheck(unittest.TestCase):
    def test_pip_install_no_version(self):
        self.assertIsNone(try_extract_super_gradients_version_from_pip_install_command("!pip install super-gradients"))

    def test_pip_install_major_only(self):
        self.assertEquals(try_extract_super_gradients_version_from_pip_install_command("!pip install super-gradients==3"), "3")

    def test_pip_install_major_minor(self):
        self.assertEquals(try_extract_super_gradients_version_from_pip_install_command("!pip install super-gradients==3.0"), "3.0")

    def test_pip_install_major_patch(self):
        self.assertEquals(try_extract_super_gradients_version_from_pip_install_command("!pip install super-gradients==3.3.1"), "3.3.1")

    def test_pip_install_with_underscore(self):
        self.assertEquals(try_extract_super_gradients_version_from_pip_install_command("!pip install super_gradients==3.3.1"), "3.3.1")

    def test_pip_install_with_extra_args(self):
        self.assertEquals(try_extract_super_gradients_version_from_pip_install_command("!pip install -q super-gradients==3.3.1"), "3.3.1")
        self.assertEquals(try_extract_super_gradients_version_from_pip_install_command("!pip install super-gradients==3.3.1 --extra-index-url=foobar"), "3.3.1")

    def test_pip_install_with_space(self):
        self.assertEquals(try_extract_super_gradients_version_from_pip_install_command("! pip install -q super-gradients==3.3.1"), "3.3.1")

    def test_pip_install_with_stdout_redirect(self):
        self.assertEquals(try_extract_super_gradients_version_from_pip_install_command("! pip install -q super-gradients==3.3.1 &> /dev/null"), "3.3.1")

    def test_pip_install_with_extra_packages(self):
        self.assertEquals(try_extract_super_gradients_version_from_pip_install_command("! pip install super-gradients==3.3.1 torch==2.0 numpy>2"), "3.3.1")


if __name__ == "__main__":
    unittest.main()
