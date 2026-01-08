commutils
---------

The code provides some useful tools.


### Dependency Installation

[yaml-cpp](https://github.com/jbeder/yaml-cpp) and [googletest](https://github.com/google/googletest) are needed.

1. Install yaml-cpp:
```
sudo apt update
sudo apt install libyaml-cpp-dev
```

2. Install googletest
```
sudo apt install libgtest-dev
cd /usr/src/gtest
sudo cmake CMakeLists.txt
sudo make
sudo cp ./lib/*.a /usr/lib
```


### Package Usage
Replace `<work_folder>` with a specific workspace name, such as rob_ws.
```
mkdir -p <work_folder>/src
cd <work_folder>/src
git clone git@github.com:agileloma/commutils.git
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
