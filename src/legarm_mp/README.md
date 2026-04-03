colcon build
. install/setup.bash

python3 ./src/legarm_mp/demos/demo_b2z1_head_stabilizing.py /configs/b2z1_head_stabilizing.yaml

python3 ./src/legarm_mp/demos/demo_b2z1_head_switch.py /configs/b2z1_head_stabilizing.yaml

python3 ./src/legarm_mp/demos/demo_b2z1_nohead_switch.py /configs/b2z1_nohead_stabilizing.yaml

python3 ./src/legarm_mp/demos/demo_b2z1_head_switch1.py /configs/b2z1_head_stabilizing1.yaml

