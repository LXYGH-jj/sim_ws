"""
@file yaml_parser.py
@package commutils
@author Jun LI (junlileeds@gmail.com)
@license License BSD-3-Clause
@Copyright (c) 2021, University of Leeds and Harbin Institute of Technology.
@date 2021-08-16
"""

import yaml

def load_yaml(yaml_filename):
    with open(yaml_filename) as fd:
        params = yaml.load(fd, Loader=yaml.FullLoader)
    return params