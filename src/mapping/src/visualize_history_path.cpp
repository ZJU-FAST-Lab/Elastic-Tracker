#include <Eigen/Geometry>

#include "geometry_msgs/PoseStamped.h"
#include "nav_msgs/Odometry.h"
#include "nav_msgs/Path.h"
#include "quadrotor_msgs/PositionCommand.h"
#include "ros/ros.h"
#include "visualization_msgs/Marker.h"

using namespace std;

nav_msgs::Path drone_path, target_path;
ros::Publisher drone_path_pub, target_path_pub, lines_pub, spring_pub;
visualization_msgs::Marker line_list;

void target_odom_callback(const nav_msgs::Odometry::ConstPtr& msg) {
  target_path.header.frame_id = "world";
  geometry_msgs::PoseStamped pose;
  pose.pose.position.x = msg->pose.pose.position.x;
  pose.pose.position.y = msg->pose.pose.position.y;
  pose.pose.position.z = msg->pose.pose.position.z;
  target_path.poses.push_back(pose);
  target_path_pub.publish(target_path);
  ROS_INFO("pulish path once");

  // NOTE spring
  if (drone_path.poses.empty()) {
    return;
  }

  {
    Eigen::Vector3d p0, p1;
    p0.x() = msg->pose.pose.position.x;
    p0.y() = msg->pose.pose.position.y;
    p0.z() = msg->pose.pose.position.z;

    p1.x() = drone_path.poses.back().pose.position.x;
    p1.y() = drone_path.poses.back().pose.position.y;
    p1.z() = drone_path.poses.back().pose.position.z;

    Eigen::Vector3d dp = p1 - p0;
    Eigen::Vector3d dx = dp.normalized();
    Eigen::Vector3d dy = Eigen::Vector3d(0, 0, 1).cross(dx);
    Eigen::Vector3d dz = dx.cross(dy);

    nav_msgs::Path spring_msg;
    spring_msg.header.frame_id = "world";
    for (double t = 0; t < 10 * 2 * M_PI; t += 0.1) {
      double y = 0.2 * cos(t);
      double z = 0.2 * sin(t);
      Eigen::Vector3d p = p0 + dp * t / (10 * 2 * M_PI) + z * dz + y * dy;
      geometry_msgs::PoseStamped pose;
      pose.pose.position.x = p.x();
      pose.pose.position.y = p.y();
      pose.pose.position.z = p.z();
      spring_msg.poses.push_back(pose);
    }
    spring_pub.publish(spring_msg);
  }

  static ros::Time t_last = ros::Time::now();
  ros::Time t_now = ros::Time::now();
  if ((t_now - t_last).toSec() < 0.3) {
    return;
  }
  t_last = t_now;
  line_list.header.stamp = ros::Time::now();
  line_list.header.frame_id = "world";
  line_list.action = visualization_msgs::Marker::ADD;
  line_list.ns = "lines";
  line_list.type = visualization_msgs::Marker::LINE_LIST;
  line_list.scale.x = 0.03;
  line_list.color.r = 0.0;
  line_list.color.g = 1.0;
  line_list.color.b = 0.1;
  line_list.color.a = 0.5;
  ROS_INFO("check1");

  geometry_msgs::Point p;
  p.x = msg->pose.pose.position.x;
  p.y = msg->pose.pose.position.y;
  p.z = msg->pose.pose.position.z;
  line_list.points.push_back(p);
  ROS_INFO("check2");

  p.x = drone_path.poses.back().pose.position.x;
  p.y = drone_path.poses.back().pose.position.y;
  p.z = drone_path.poses.back().pose.position.z;
  line_list.points.push_back(p);
  ROS_INFO("check3");

  lines_pub.publish(line_list);
}

void cmd_callback(const quadrotor_msgs::PositionCommand cmd) {
  drone_path.header.frame_id = "world";
  geometry_msgs::PoseStamped pose;
  pose.pose.position.x = cmd.position.x;
  pose.pose.position.y = cmd.position.y;
  pose.pose.position.z = cmd.position.z;
  drone_path.poses.push_back(pose);
  drone_path_pub.publish(drone_path);
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "visualize_history_path");
  ROS_INFO("visualize_history_path node init ok");

  ros::NodeHandle n("~");
  ros::Subscriber cmd_sub = n.subscribe("/position_cmd", 100, cmd_callback);
  ros::Subscriber target_odom_sub = n.subscribe("/object_odom_dtc2brig", 100, target_odom_callback);

  drone_path_pub = n.advertise<nav_msgs::Path>("history_drone_pose", 100, true);
  target_path_pub = n.advertise<nav_msgs::Path>("history_target_pose", 100, true);
  lines_pub = n.advertise<visualization_msgs::Marker>("drone_target_link_line", 100, true);

  spring_pub = n.advertise<nav_msgs::Path>("spring", 100, true);

  ros::spin();

  return 0;
}
