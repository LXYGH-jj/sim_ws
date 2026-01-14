# 创建 test2.py 文件
import sys
sys.path.append('/home/liu/sim_ws/src/legarm_wbc/python')

from legarm_wbc.sim_robot import SimRobot
import numpy as np

# 创建机器人实例
robot = SimRobot(
    xml_path="/home/liu/sim_ws/src/legarm_wbc/models/b2z1/b2z1_grasp.xml",
    urdf_path="/home/liu/sim_ws/src/legarm_wbc/models/b2z1.urdf",
    base_name="base_link"
)

# 测试获取基座位置
try:
    base_pos = robot.get_body_pos("base_link")  # 尝试获取基座位置
    print(f"Base position: {base_pos}")
except Exception as e:
    print(f"Error getting base position: {e}")

# 测试获取框架姿态
try:
    frame_pose = robot.get_frame_pose("base_link")
    print(f"Frame pose: {frame_pose}")
except Exception as e:
    print(f"Error getting frame pose: {e}")