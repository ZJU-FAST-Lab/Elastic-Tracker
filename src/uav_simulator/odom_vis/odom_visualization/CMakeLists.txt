
cmake_minimum_required(VERSION 2.8.3)
project(odom_visualization)

find_package(Eigen3 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  nav_msgs
  visualization_msgs
  quadrotor_msgs
  tf
  pose_utils
  Eigen3 REQUIRED
)


catkin_package(
 INCLUDE_DIRS 
 LIBRARIES 
 DEPENDS OpenCV Eigen Boost

#  DEPENDS system_lib
)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})


include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS} 
)

add_executable(odom_visualization src/odom_visualization.cpp)

target_link_libraries(odom_visualization
   ${catkin_LIBRARIES}
   ${ARMADILLO_LIBRARIES}
   pose_utils
)

add_dependencies(odom_visualization
  ${catkin_EXPORTED_TARGETS}
)

add_executable(odom_visualization_car src/odom_visualization_car.cpp)

target_link_libraries(odom_visualization_car
   ${catkin_LIBRARIES}
   ${ARMADILLO_LIBRARIES}
   pose_utils
)

add_dependencies(odom_visualization_car
  ${catkin_EXPORTED_TARGETS}
)
