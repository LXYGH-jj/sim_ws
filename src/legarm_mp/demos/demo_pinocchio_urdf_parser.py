from pathlib import Path
import numpy as np
from sys import argv

import pinocchio as pin

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
mesh_dir = os.path.join(script_dir, "/home/liu/sim_ws/src/legarm_mp/models")


# 设置模型目录
pinocchio_model_dir = Path(__file__).parent.parent / "models"

# 设置 URDF 文件路径
urdf_filename = (
    "/home/liu/sim_ws/src/legarm_mp/models/b2z1noarm.urdf"
    if len(argv) < 2
    else argv[1]
)

# # 设置网格文件目录
# mesh_dir = (
#     "/home/zxy/RobCode/b2z1_sim/src/legarm_mp/models/meshes"    
#     if len(argv) < 3
#     else argv[2]
# )


# 将 Path 对象转换为字符串
urdf_filename_str = str(urdf_filename)
mesh_dir_str = str(mesh_dir)

# 加载 URDF 模型
model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_filename_str, mesh_dir_str, pin.JointModelFreeFlyer())


# Check dimensions of the original model
print("standard model: dim=" + str(len(model.joints)))
for jn in model.joints:
    print(jn)
print("-" * 30)

print("model name: " + model.name)
data = model.createData()
q = pin.randomConfiguration(model)
print(f"q: {q.T}")

# q = np.array([0., 0., 0.36, 0., 0., 0., 1.,
#               0.15,  0.72, -1.42, 
#               -0.15,  0.72, -1.42, 
#               0.15,  0.72, -1.42,
#               -0.15,  0.72, -1.42,])

q = np.array([0., 0., 0.4, 0., 0., 0., 1.,
                0., 0.75, -1.5, 
                0., 0.75, -1.5, 
                0., 0.75, -1.5, 
                0., 0.75, -1.5,])
                # 0.,0.,0.,0.,0.,0.])

v = np.array([0., 0., 0. , 0., 0., 0.,
              0., 0., 0., 
              0., 0., 0., 
              0., 0., 0.,
              0., 0., 0.,])
            #   0.,0.,0.,0.,0.,0.])

pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)
pin.computeAllTerms(model, data, q, v)
pin.updateFramePlacements(model, data)


total_mass = pin.computeTotalMass(model)
print("total mass:", total_mass)
# center_mass = pin.centerOfMass(model, data, q)
# print("center_mass:", center_mass)


# print("Mass matrix: \n" , data.M)
# print("Non-linear effects: \n" , data.nle.transpose())
# print("Gravity vector: \n", data.g.transpose())
# # print("Joint torques: ", data.tau.transpose())
# gravity = model.gravity
# print(gravity)

# for i, inertia in enumerate(model.inertias):
#     print(f"Link {i}:")
#     print(f"  Mass: {inertia.mass}")
#     print(f"  Center of mass: {inertia.lever}")
#     print(f"  Inertia (wr. com):\n{inertia.inertia}")
#     print("-" * 30)





# 获取基座位置
# base_frame_id = model.getFrameId("base_link")            # 针对B2
base_frame_id = model.getFrameId("base_link")     # 针对Go2
oMb = data.oMf[base_frame_id]
base_translation = oMb.translation

# 获取并打印四个足端相对于基座的位置
foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot",]
for foot_name in foot_names:
    foot_frame_id = model.getFrameId(foot_name)
    oMf = data.oMf[foot_frame_id]
    foot_translation = oMf.translation
    relative_translation = foot_translation - base_translation
    print(f"{foot_name} position relative to base: {relative_translation}")

# 获取并打印四个髋关节相对于基座的位置
hip_names = ["FL_hip", "FR_hip", "RL_hip", "RR_hip"]


for hip_name in hip_names:
    hip_frame_id = model.getFrameId(hip_name)
    oMf = data.oMf[hip_frame_id]
    hip_translation = oMf.translation
    relative_translation = hip_translation - base_translation
    print(f"{hip_name} position relative to base: {relative_translation}")