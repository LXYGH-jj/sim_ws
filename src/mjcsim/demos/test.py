# test_viewer.py
import time
from mjcsim.env import MujocoEnv

env = MujocoEnv(gui=True, dt=0.002)
print('viewer:', env._viewer)
# 保持进程一段时间以观察窗口
time.sleep(15)