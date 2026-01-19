#!/usr/bin/env python

from os import path, walk
from setuptools import setup, find_packages

package_name = "commutils"
package_version = "1.0.0"

with open(path.join(path.dirname(path.realpath(__file__)), "readme.md"), "r") as fh:
    long_description = fh.read()

# Install nodes and demos.
scripts_list = []
for (root, _, files) in walk(path.join("tests")):
    for test_file in files:
        scripts_list.append(path.join(root, test_file))

# Setup the package
setup(
    name=package_name,
    version=package_version,
    package_dir={'': 'python',},
    packages=find_packages(where='python'),
    scripts=scripts_list,
    # required dependencies.
    install_requires=["setuptools"],
    # optional (hack) dependencies.
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="Jun Li",
    maintainer_email="junlileeds@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ihcr/commutils",
    description="Wrapper around some common useful tools.",
    license="BSD-3-clause",
    entry_points={
        "console_scripts": [],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-3-clause",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)