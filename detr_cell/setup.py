from typing import List

from setuptools import find_namespace_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements: List[str] = []
test_requirements = ["pytest>=3"]

setup(
    author="Pieter Blomme",
    author_email="pieter.blomme@robovision.ai",
    python_requires=">=3.6",
    classifiers=[],
    description="DetrCell Cell",
    entry_points={
        "rvai.cells": [
            "detr_cell=rvai.cells.detr_cell:__cell__",
        ],
    },
    install_requires=requirements,
    long_description=readme,
    include_package_data=True,
    name="rvai.cells.detr_cell",
    namespace_packages=[
        "rvai",
        "rvai.cells"
    ],
    packages=find_namespace_packages(
        include=[
            "rvai.cells.detr_cell",
            "rvai.cells.detr_cell.*",
        ]
    ),
    test_suite="tests",
    tests_require=test_requirements,
    extras_require={
        'test': test_requirements
    },
    version="0.1.0",
    zip_safe=False,
)
