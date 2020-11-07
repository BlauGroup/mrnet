#!/usr/bin/env python

from pathlib import Path

from setuptools import find_packages, setup

module_dir = Path(__file__).resolve().parent

with open(module_dir / "README.md") as f:
    long_desc = f.read()

setup(
    name="mrnet",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    description="",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    url="https://github.com/materialsproject/mrnet",
    author="The Materials Project",
    author_email="feedback@materialsproject.org",
    license="modified BSD",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"mrnet": ["py.typed"]},
    zip_safe=False,
    include_package_data=True,
    install_requires=[
        "setuptools",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: System Administrators",
        "Intended Audience :: Information Technology",
        "Operating System :: OS Independent",
        "Topic :: Other/Nonlisted Topic",
        "Topic :: Database :: Front-Ends",
        "Topic :: Scientific/Engineering",
    ],
    tests_require=["pytest"],
    python_requires=">=3.7",
)
