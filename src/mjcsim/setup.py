#!/usr/bin/env python

from os import path, walk
from setuptools import find_packages, setup

package_name = 'mjcsim'
package_version = "1.0.0"

with open(path.join(path.dirname(path.realpath(__file__)), "readme.md"), "r") as fh:
    long_description = fh.read()


# Setup the package
setup(
    name=package_name,
    version=package_version,
    package_dir={'': 'python',
                 'commutils': '../commutils/python/commutils',  
                 },
    packages=find_packages(where='python'),
    install_requires=[
        "setuptools", "mujoco", "numpy",
    ],
    zip_safe=True,
    maintainer='Xinyuan Liu',
    maintainer_email='1214247879@qq.com',
    description="MuJoCo wrapper for robots.",
    license="BSD-3-clause",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ihcr/mjcsim",
    entry_points={
        "console_scripts": [],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
