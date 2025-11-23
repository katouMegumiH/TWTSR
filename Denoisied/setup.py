# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open("README.rst", encoding='UTF-8') as f:
    readme = f.read()

with open("LICENSE", encoding='UTF-8') as f:
    license = f.read()

with open("requirements.txt", "r", encoding='UTF-8') as f:
    required = f.read().splitlines()

setup(
    name="program",
    version="0.1.2",
    description=("implemenent deep interpolation to denoise data by "
                 "removing independent noise"),
    long_description=readme,
    author="Jerome Lecoq",
    author_email="jeromel@alleninstitute.org",
    url="https://github.com/AllenInstitute/deepinterpolation",
    license=license,
    packages=find_packages(exclude=("tests", "docs")),
    install_requires=required,
)
