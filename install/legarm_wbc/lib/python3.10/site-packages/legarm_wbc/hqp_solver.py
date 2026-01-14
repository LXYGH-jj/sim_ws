"""
@file hqp_solver.py
@package legarm_wbc
@author Xinyuan Liu (liuxinyuan872@gmail.com)
@license License BSD-3-Clause
@Copyright (c) 2026, Harbin Institute of Technology.
@date 2026-01
"""

import numpy as np
from scipy.linalg import pinv
from qpsolvers import solve_qp

class HQPSolver:
    def __init__(self, slack_variable=1e-10):
        self.slack_variable = slack_variable

    def nullspace_projector(self, M):
        projector = np.eye(M.shape[1]) - pinv(M).dot(M)
        return projector

    def solve(self, hqp_data):
        # extract uniq and sorted priority list
        priority_list = []
        for task_name, task_description in hqp_data.hierarchical_tasks.items():
            priority_list.append(task_description["priority"])
        priority_list = list(set(priority_list))  # remove repeated priority
        priority_list.sort()

        # compress tasks with same priority
        task_hierarchy = {}
        for priority in priority_list:
            A, b = [], []
            for task_name, task_description in hqp_data.hierarchical_tasks.items():
                if task_description["priority"] == priority:
                    A.append(task_description["weight"]*task_description["A"])
                    b.append(task_description["weight"]*task_description["b"])
            A = np.vstack(A)
            b = np.hstack(b)
            task_hierarchy[priority] = {"A":A, "b":b}

        # stack inequality constraints
        D0, f0 = np.zeros((0,hqp_data.num_vars)), np.zeros(0) 
        C, d = [], []
        for cons_name, cons_description in hqp_data.constraints.items():
            C.append(cons_description["C"])
            d.append(cons_description["d"])
        D0 = np.vstack(C)
        f0 = np.hstack(d)

        # solve hierarchical QP based on priority
        C_bar, d_bar = np.eye(hqp_data.num_vars), np.zeros(hqp_data.num_vars)
        for priority, task in task_hierarchy.items():
            Ai_bar = task["A"].dot(C_bar)
            bi_bar = task["b"] - task["A"].dot(d_bar)
            P = Ai_bar.T.dot(Ai_bar)
            P += np.eye(P.shape[0]) * float(self.slack_variable)
            di = solve_qp(P=P,
                          q=-Ai_bar.T.dot(bi_bar),
                          G=D0.dot(C_bar),
                          h=f0 - D0.dot(d_bar),
                          A=None,
                          b=None,
                          solver='quadprog')
            d_bar = d_bar + C_bar.dot(di)
            C_bar = C_bar.dot(self.nullspace_projector(Ai_bar))
        return d_bar
