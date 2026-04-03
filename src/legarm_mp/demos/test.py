#!/usr/bin/env python3
"""正确获取 URDF 质量分布 - 通过 joint 索引"""

import pinocchio as pin
import numpy as np

model = pin.buildModelFromUrdf(
    "/home/liu/sim_ws/src/legarm_mp/models/b2z1noarm.urdf",
    pin.JointModelFreeFlyer()
)

print("=== 所有关节惯性质量 ===")
total_mass = 0
for i, name in enumerate(model.names):
    if i < len(model.inertias):
        mass = model.inertias[i].mass
        if mass > 1e-10:
            print(f"{name} (joint {i}): {mass:.4f} kg")
            total_mass += mass

print(f"\n总质量：{total_mass:.4f} kg")

print("\n=== 基座相关质量 ===")
# inertias[0] 是 universe, inertias[1] 是 root_joint (FreeFlyer)
print(f"root_joint (FreeFlyer): {model.inertias[1].mass:.4f} kg")

# 通过 joint 获取基座相关质量
base_joint_id = model.getJointId("root_joint")
print(f"base_link 通过 root_joint: {model.inertias[base_joint_id].mass:.4f} kg")

# 计算固定附件质量 (通过 joint 获取)
fixed_links = [
    ('joint_lidar', 'lidar_link'),
    ('joint_f_dc', 'f_dc_link'),
    ('joint_r_dc', 'r_dc_link'),
    ('joint_imu', 'imu_link'),
    ('joint_head', 'head_Link'),
    ('joint_tail', 'tail_link'),
]

fixed_mass = 0
for joint_name, link_name in fixed_links:
    joint_id = model.getJointId(joint_name)
    if joint_id < len(model.inertias) and joint_id > 0:
        # 固定关节的惯性在 parent joint 中
        mass = model.inertias[joint_id].mass
        if mass > 1e-10:
            print(f"{link_name} (via {joint_name}): {mass:.4f} kg")
            fixed_mass += mass

print(f"\n基座 + 固定附件总质量：{model.inertias[1].mass:.4f} kg")

print("\n=== 腿部质量 (通过 joint 获取) ===")
leg_joints = {
    'FL': ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint'],
    'FR': ['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint'],
    'RL': ['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint'],
    'RR': ['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint'],
}

for leg, joints in leg_joints.items():
    leg_mass = 0
    for joint_name in joints:
        joint_id = model.getJointId(joint_name)
        if joint_id < len(model.inertias):
            mass = model.inertias[joint_id].mass
            leg_mass += mass
    print(f"{leg} 腿：{leg_mass:.4f} kg")

print("\n=== 髋关节位置 ===")
hip_names = ["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"]
for hip_joint in hip_names:
    joint_id = model.getJointId(hip_joint)
    if joint_id < len(model.jointPlacements):
        pos = model.jointPlacements[joint_id].translation
        print(f"{hip_joint}: {pos}")

print("\n=== SRBM 规划器推荐质量 ===")
print(f"方案 A (root_joint): {model.inertias[1].mass:.4f} kg  ← ✅ 推荐")
print(f"方案 B (总质量): {total_mass:.4f} kg")
print(f"当前配置：78.9990 kg")

print("\n=== 基座惯性 (root_joint) ===")
base_inertia = model.inertias[1]
print(f"Ixx: {base_inertia.inertia[0, 0]:.4f}")
print(f"Iyy: {base_inertia.inertia[1, 1]:.4f}")
print(f"Izz: {base_inertia.inertia[2, 2]:.4f}")

print("\n=== URDF 中的基座惯性 ===")
print("从 URDF base_link:")
print("  Ixx: 0.2747, Iyy: 1.0618, Izz: 1.1825")
print("从 Pinocchio root_joint:")
print(f"  Ixx: {base_inertia.inertia[0, 0]:.4f}, Iyy: {base_inertia.inertia[1, 1]:.4f}, Izz: {base_inertia.inertia[2, 2]:.4f}")