"""
@file hqp_data.py
@package legarm_wbc
@author Xinyuan Liu (liuxinyuan872@gmail.com)
@license License BSD-3-Clause
@Copyright (c) 2026, Harbin Institute of Technology.
@date 2026-01
"""

class HQPData:
    def __init__(self):
        self.num_vars = 0
        self.constraints = {}
        self.hierarchical_tasks = {}

    def set_num_variables(self, dim):
        self.num_vars = dim

    def add_constraint(self, name, matrix, vector):
        assert matrix.shape[0] == vector.shape[0]
        self.constraints[name] = {"C":matrix, "d":vector}

    def add_task(self, name, priority, weight, matrix, vector):
        assert matrix.shape[0] == vector.shape[0]
        self.hierarchical_tasks[name] = {
            "priority":priority, "weight":weight, "A":matrix, "b":vector}
