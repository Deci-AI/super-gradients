# coding: utf-8

"""
    Deci Training Toolkit
"""

from setuptools import setup
from setuptools import find_packages

README_LOCATION = 'README.md'
REQ_LOCATION = 'requirements.txt'
VERSION_FILE = "version.txt"


def readme():
    """print long description"""
    with open(README_LOCATION) as f:
        return f.read()


def get_requirements():
    with open(REQ_LOCATION) as f:
        return f.read().splitlines()


def get_version():
    with open(VERSION_FILE) as f:
        return f.readline()


setup(
    name='super-gradients',
    version=get_version(),
    description="SuperGradients",
    author="Deci AI",
    author_email="rnd@deci.ai",
    url="https://deci.ai",
    keywords=["Deci", "AI", "Training"],
    install_requires=get_requirements(),
    include_package_data=True,
    packages=find_packages(),
    data_files=[('config', ['super_gradients/common/auto_logging/auto_logging_conf.json'])],
    long_description=readme()
)
