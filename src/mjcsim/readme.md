# mjcsim - MuJoCo Robot Simulator
A MuJoCo-based robot simulation package.
-------
1. Install Mujoco:
  pip3 install mujoco

### Package Usage
Replace `<work_folder>` with a specific workspace name, such as rob_ws.
```
mkdir -p <work_folder>/src
cd <work_folder>/src
git clone https://github.com/LXYGH-jj/sim_ws.git/src/commutils
git clone https://github.com/LXYGH-jj/sim_ws.git/src/mjcsim
cd ..
colcon build
```
Once the code has been compiled, you can source .bash file in `install/setup.bash`
```
. install/setup.bash
```

**Loading env in PyBullet**

```
import mujoco as mjc
from mjcsim.env import MujocoEnvWithGround

env = MujocoEnvWithGround()
```

### Running demos
```
cd <work_folder>/

python3 ./src/mjcsim/demos/demo_robot_standing.py /configs/b2z1_standing.yaml
```

### License and Copyrights

Copyright (c) 2026, Harbin Institute of Technology.
BSD 3-Clause License
