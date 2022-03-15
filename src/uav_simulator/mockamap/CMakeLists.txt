cmake_minimum_required(VERSION 3.0.2)
project(mockamap)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  pcl_ros
  pcl_conversions
)

catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES mockamap
  CATKIN_DEPENDS roscpp pcl_ros pcl_conversions
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

file(GLOB ${PROJECT_NAME}_SRCS src/*.cpp)

add_executable(${PROJECT_NAME}_node
  ${${PROJECT_NAME}_SRCS})

target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
)
