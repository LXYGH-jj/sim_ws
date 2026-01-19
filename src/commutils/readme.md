commutils
---------

The code provides some useful tools.

### Package Usage
Replace `<work_folder>` with a specific workspace name, such as rob_ws.
```
mkdir -p <work_folder>/src
cd <work_folder>/src
git clone https://github.com/ihcr/commutils.git
cd ..
colcon build
```
Once the code has been compiled, you can source .bash file in `install/setup.bash`
```
. install/setup.bash
```

### License and Copyrights

Copyright (c) 2021, University of Leeds and Harbin Institute of Technology.
BSD 3-Clause License