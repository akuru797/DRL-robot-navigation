export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11311
export ROS_PORT_SIM=11311
export GAZEBO_RESOURCE_PATH=~/repos/DRL-robot-navigation/catkin_ws/src/multi_robot_scenario/launch
source ~/.bashrc
cd ~/repos/DRL-robot-navigation/catkin_ws
source devel_isolated/setup.bash
cd ~/repos/DRL-robot-navigation/DDPG