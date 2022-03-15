cmake_minimum_required(VERSION 3.0.2)
project(quadrotor_msgs)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  nav_msgs
  geometry_msgs
  message_generation
)

add_message_files(
  DIRECTORY msg
  FILES
  OccMap3d.msg
  PolyTraj.msg
  ReplanState.msg
  PositionCommand.msg
  SO3Command.msg
  AuxCommand.msg
  Px4ctrlDebug.msg
  TakeoffLand.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  nav_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime geometry_msgs nav_msgs
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)
