# Elastic-Tracker

## 0. Overview
**Elastic-Tracker** is a flexible trajectory planning framework that can deal with challenging tracking tasks with guaranteed safety and visibility.

**Authors**: Jialin Ji, Neng Pan and [Fei Gao](https://ustfei.com/) from the [ZJU Fast Lab](http://zju-fast.com/). 

**Paper**: [Elastic Tracker: A Spatio-temporal Trajectory Planner Flexible Aerial Tracking](https://arxiv.org/abs/2109.07111), Jialin Ji, Neng Pan, Chao Xu, Fei Gao, Accepted in IEEE International Conference on Robotics and Automation (__ICRA 2022__).

**Video Links**: [youtube](https://www.youtube.com/watch?v=G5taHOpAZj8) or [bilibili](https://www.bilibili.com/video/BV1o44y1b7wC)
<a href="https://www.youtube.com/watch?v=G5taHOpAZj8" target="blank">
  <p align="center">
    <img src="figs/cover.png" width="500"/>
  </p>
</a>

## 1. Run Simulations

[NOTE] remember to change the CUDA option of **src/uav_simulator/local_sensing/CMakeLists.txt**

>Preparation and visualization:
```
git clone https://github.com/ZJU-FAST-Lab/Elastic-Tracker.git
cd Elastic-Tracker
catkin_make
source devel/setup.zsh
chmod +x sh_utils/pub_triger.sh
roslaunch mapping rviz_sim.launch
```

>A small drone with the global map as the chasing target:
```
roslaunch planning fake_target.launch
```

>Start the elastic tracker:
```
roslaunch planning simulation1.launch
```

>Triger the drone to track the target:
```
./sh_utils/pub_triger.sh
```
<p align="center">
    <img src="figs/sim1.gif" width="500"/>
</p>

Comparision of the planners <font color=blue>with</font> and <font color=orange>without</font> **visibility guarantee**:
```
roslaunch planning simulation2.launch
```
<p align="center">
    <img src="figs/sim2.gif" width="500"/>
</p>
