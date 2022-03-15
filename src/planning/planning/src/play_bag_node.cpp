#include <mapping/mapping.h>
#include <quadrotor_msgs/OccMap3d.h>
#include <quadrotor_msgs/ReplanState.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <wr_msg/wr_msg.hpp>

void replan_state_callback(const quadrotor_msgs::ReplanStateConstPtr& msgPtr) {
  // ROS_WARN("[log] REPLAN STATE RECEIVED!");
  if (msgPtr->state == 2) {
    wr_msg::writeMsg(*msgPtr, ros::package::getPath("planning") + "/../../../debug/replan_state.bin");
    assert(false);
  }
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "play_bag_node");
  ros::NodeHandle nh("~");
  ros::Subscriber replanState_sub =
      nh.subscribe<quadrotor_msgs::ReplanState>("replanState", 1, replan_state_callback, ros::TransportHints().tcpNoDelay());

  ros::spin();
  return 0;
}
