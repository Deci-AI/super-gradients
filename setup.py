# coding: utf-8

"""
    Deci Training Toolkit
"""

from setuptools import setup
from setuptools import find_packages

README_LOCATION = "README.md"
REQ_LOCATION = "requirements.txt"
REQ_PRO_LOCATION = "requirements.pro.txt"
VERSION_FILE = "version.txt"


def readme():
    """print long description"""
    with open(README_LOCATION, encoding="utf-8") as f:
        return f.read()


def get_requirements():
    with open(REQ_LOCATION, encoding="utf-8") as f:
        requirements = f.read().splitlines()
        return [r for r in requirements if not r.startswith("--") and not r.startswith("#")]


def get_pro_requirements():
    with open(REQ_PRO_LOCATION, encoding="utf-8") as f:
        return f.read().splitlines()


def get_version():
    with open(VERSION_FILE, encoding="utf-8") as f:
        return f.readline()


setup(
    name="super-gradients",
    version=get_version(),
    description="SuperGradients",
    author="Deci AI",
    author_email="rnd@deci.ai",
    url="https://deci-ai.github.io/super-gradients/welcome.html",
    keywords=["Deci", "AI", "Training", "Deep Learning", "Computer Vision", "PyTorch", "SOTA", "Recipes", "Pre Trained", "Models"],
    install_requires=get_requirements(),
    packages=find_packages(where="./src"),
    package_dir={"": "src"},
    package_data={
        "super_gradients.recipes": ["*.yaml", "**/*.yaml"],
        "super_gradients.common": ["auto_logging/auto_logging_conf.json"],
        "super_gradients.examples": ["*.ipynb", "**/*.ipynb"],
        "super_gradients": ["requirements.txt", "requirements.pro.txt"],
    },
    long_description=readme(),
    long_description_content_type="text/markdown",
    extras_require={"pro": get_pro_requirements()},
)
