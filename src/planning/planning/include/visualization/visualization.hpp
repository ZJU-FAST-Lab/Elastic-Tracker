#pragma once
#include <nav_msgs/Path.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Geometry>
#include <iostream>
#include <unordered_map>

namespace visualization {
using PublisherMap = std::unordered_map<std::string, ros::Publisher>;
enum Color { white,
             red,
             green,
             blue,
             yellow,
             greenblue };

class Visualization {
 private:
  ros::NodeHandle nh_;
  PublisherMap publisher_map_;

  void setMarkerColor(visualization_msgs::Marker& marker,
                      Color color = blue,
                      double a = 1) {
    marker.color.a = a;
    switch (color) {
      case white:
        marker.color.r = 1;
        marker.color.g = 1;
        marker.color.b = 1;
        break;
      case red:
        marker.color.r = 1;
        marker.color.g = 0;
        marker.color.b = 0;
        break;
      case green:
        marker.color.r = 0;
        marker.color.g = 1;
        marker.color.b = 0;
        break;
      case blue:
        marker.color.r = 0;
        marker.color.g = 0;
        marker.color.b = 1;
        break;
      case yellow:
        marker.color.r = 1;
        marker.color.g = 1;
        marker.color.b = 0;
        break;
      case greenblue:
        marker.color.r = 0;
        marker.color.g = 1;
        marker.color.b = 1;
        break;
    }
  }

  void setMarkerColor(visualization_msgs::Marker& marker,
                      double a,
                      double r,
                      double g,
                      double b) {
    marker.color.a = a;
    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;
  }

  void setMarkerScale(visualization_msgs::Marker& marker,
                      const double& x,
                      const double& y,
                      const double& z) {
    marker.scale.x = x;
    marker.scale.y = y;
    marker.scale.z = z;
  }

  void setMarkerPose(visualization_msgs::Marker& marker,
                     const double& x,
                     const double& y,
                     const double& z) {
    marker.pose.position.x = x;
    marker.pose.position.y = y;
    marker.pose.position.z = z;
    marker.pose.orientation.w = 1;
    marker.pose.orientation.x = 0;
    marker.pose.orientation.y = 0;
    marker.pose.orientation.z = 0;
  }
  template <class ROTATION>
  void setMarkerPose(visualization_msgs::Marker& marker,
                     const double& x,
                     const double& y,
                     const double& z,
                     const ROTATION& R) {
    marker.pose.position.x = x;
    marker.pose.position.y = y;
    marker.pose.position.z = z;
    Eigen::Quaterniond r(R);
    marker.pose.orientation.w = r.w();
    marker.pose.orientation.x = r.x();
    marker.pose.orientation.y = r.y();
    marker.pose.orientation.z = r.z();
  }

 public:
  Visualization(ros::NodeHandle& nh) : nh_(nh) {}

  template <class CENTER, class TOPIC>
  void visualize_a_ball(const CENTER& c,
                        const double& r,
                        const TOPIC& topic,
                        const Color color = blue,
                        const double a = 1) {
    auto got = publisher_map_.find(topic);
    if (got == publisher_map_.end()) {
      ros::Publisher pub = nh_.advertise<visualization_msgs::Marker>(topic, 10);
      publisher_map_[topic] = pub;
    }
    visualization_msgs::Marker marker;
    marker.header.frame_id = "world";
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    setMarkerColor(marker, color, a);
    setMarkerScale(marker, 2 * r, 2 * r, 2 * r);
    setMarkerPose(marker, c[0], c[1], c[2]);
    marker.header.stamp = ros::Time::now();
    publisher_map_[topic].publish(marker);
  }

  template <class PC, class TOPIC>
  void visualize_pointcloud(const PC& pc, const TOPIC& topic) {
    auto got = publisher_map_.find(topic);
    if (got == publisher_map_.end()) {
      ros::Publisher pub = nh_.advertise<sensor_msgs::PointCloud2>(topic, 10);
      publisher_map_[topic] = pub;
    }
    pcl::PointCloud<pcl::PointXYZ> point_cloud;
    sensor_msgs::PointCloud2 point_cloud_msg;
    point_cloud.reserve(pc.size());
    for (const auto& pt : pc) {
      point_cloud.points.emplace_back(pt[0], pt[1], pt[2]);
    }
    pcl::toROSMsg(point_cloud, point_cloud_msg);
    point_cloud_msg.header.frame_id = "world";
    point_cloud_msg.header.stamp = ros::Time::now();
    publisher_map_[topic].publish(point_cloud_msg);
  }

  template <class PATH, class TOPIC>
  void visualize_path(const PATH& path, const TOPIC& topic) {
    auto got = publisher_map_.find(topic);
    if (got == publisher_map_.end()) {
      ros::Publisher pub = nh_.advertise<nav_msgs::Path>(topic, 10);
      publisher_map_[topic] = pub;
    }
    nav_msgs::Path path_msg;
    geometry_msgs::PoseStamped tmpPose;
    tmpPose.header.frame_id = "world";
    for (const auto& pt : path) {
      tmpPose.pose.position.x = pt[0];
      tmpPose.pose.position.y = pt[1];
      tmpPose.pose.position.z = pt[2];
      path_msg.poses.push_back(tmpPose);
    }
    path_msg.header.frame_id = "world";
    path_msg.header.stamp = ros::Time::now();
    publisher_map_[topic].publish(path_msg);
  }

  template <class BALLS, class TOPIC>
  void visualize_balls(const BALLS& balls,
                       const TOPIC& topic,
                       const Color color = blue,
                       const double a = 0.2) {
    auto got = publisher_map_.find(topic);
    if (got == publisher_map_.end()) {
      ros::Publisher pub =
          nh_.advertise<visualization_msgs::MarkerArray>(topic, 10);
      publisher_map_[topic] = pub;
    }
    visualization_msgs::Marker marker;
    marker.header.frame_id = "world";
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.id = 0;
    setMarkerColor(marker, color, a);
    visualization_msgs::MarkerArray marker_array;
    marker_array.markers.reserve(balls.size() + 1);
    marker.action = visualization_msgs::Marker::DELETEALL;
    marker_array.markers.push_back(marker);
    marker.action = visualization_msgs::Marker::ADD;
    for (const auto& ball : balls) {
      setMarkerPose(marker, ball[0], ball[1], ball[2]);
      auto d = 2 * ball.r;
      setMarkerScale(marker, d, d, d);
      marker_array.markers.push_back(marker);
      marker.id++;
    }
    publisher_map_[topic].publish(marker_array);
  }

  template <class ELLIPSOID, class TOPIC>
  void visualize_ellipsoids(const ELLIPSOID& ellipsoids,
                            const TOPIC& topic,
                            const Color color = blue,
                            const double a = 0.2) {
    auto got = publisher_map_.find(topic);
    if (got == publisher_map_.end()) {
      ros::Publisher pub =
          nh_.advertise<visualization_msgs::MarkerArray>(topic, 10);
      publisher_map_[topic] = pub;
    }
    visualization_msgs::Marker marker;
    marker.header.frame_id = "world";
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.id = 0;
    setMarkerColor(marker, color, a);
    visualization_msgs::MarkerArray marker_array;
    marker_array.markers.reserve(ellipsoids.size() + 1);
    marker.action = visualization_msgs::Marker::DELETEALL;
    marker_array.markers.push_back(marker);
    marker.action = visualization_msgs::Marker::ADD;
    for (const auto& e : ellipsoids) {
      setMarkerPose(marker, e.c[0], e.c[1], e.c[2], e.R);
      setMarkerScale(marker, 2 * e.rx, 2 * e.ry, 2 * e.rz);
      marker_array.markers.push_back(marker);
      marker.id++;
    }
    publisher_map_[topic].publish(marker_array);
  }

  template <class PAIRLINE, class TOPIC>
  // eg for PAIRLINE: std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>
  void visualize_pairline(const PAIRLINE& pairline, const TOPIC& topic) {
    auto got = publisher_map_.find(topic);
    if (got == publisher_map_.end()) {
      ros::Publisher pub = nh_.advertise<visualization_msgs::Marker>(topic, 10);
      publisher_map_[topic] = pub;
    }
    visualization_msgs::Marker marker;
    marker.header.frame_id = "world";
    marker.type = visualization_msgs::Marker::LINE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    setMarkerPose(marker, 0, 0, 0);
    setMarkerColor(marker, greenblue, 1);
    setMarkerScale(marker, 0.02, 0.02, 0.02);
    marker.points.resize(2 * pairline.size());
    for (size_t i = 0; i < pairline.size(); ++i) {
      marker.points[2 * i + 0].x = pairline[i].first[0];
      marker.points[2 * i + 0].y = pairline[i].first[1];
      marker.points[2 * i + 0].z = pairline[i].first[2];
      marker.points[2 * i + 1].x = pairline[i].second[0];
      marker.points[2 * i + 1].y = pairline[i].second[1];
      marker.points[2 * i + 1].z = pairline[i].second[2];
    }
    publisher_map_[topic].publish(marker);
  }

  template <class TOPIC>
  void visualize_arrow(const Eigen::Vector3d p0, const Eigen::Vector3d& p1, const TOPIC& topic, const Color& color = blue) {
    auto got = publisher_map_.find(topic);
    if (got == publisher_map_.end()) {
      ros::Publisher pub = nh_.advertise<visualization_msgs::Marker>(topic, 10);
      publisher_map_[topic] = pub;
    }
    visualization_msgs::Marker arrow_msg;
    arrow_msg.type = visualization_msgs::Marker::ARROW;
    arrow_msg.action = visualization_msgs::Marker::ADD;
    arrow_msg.header.frame_id = "world";
    arrow_msg.id = 0;
    arrow_msg.points.resize(2);
    setMarkerPose(arrow_msg, 0, 0, 0);
    setMarkerScale(arrow_msg, 0.05, 0.1, 0);
    setMarkerColor(arrow_msg, color);
    arrow_msg.points[0].x = p0[0];
    arrow_msg.points[0].y = p0[1];
    arrow_msg.points[0].z = p0[2];
    arrow_msg.points[1].x = p1[0];
    arrow_msg.points[1].y = p1[1];
    arrow_msg.points[1].z = p1[2];
    publisher_map_[topic].publish(arrow_msg);
  }

  // v0 -> v1 theta
  template <class TOPIC>
  void visualize_fan_shape_meshes(const std::vector<Eigen::Vector3d>& v0,
                                  const std::vector<Eigen::Vector3d>& v1,
                                  const std::vector<double>& thetas,
                                  const TOPIC& topic) {
    auto got = publisher_map_.find(topic);
    if (got == publisher_map_.end()) {
      ros::Publisher pub = nh_.advertise<visualization_msgs::Marker>(topic, 10);
      publisher_map_[topic] = pub;
    }
    visualization_msgs::Marker marker;
    marker.type = visualization_msgs::Marker::TRIANGLE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    marker.header.frame_id = "world";
    marker.id = 0;
    setMarkerPose(marker, 0, 0, 0);
    setMarkerScale(marker, 1, 1, 1);
    setMarkerColor(marker, green, 0.1);
    int M = v0.size();
    // int M = 1;
    for (int i = 0; i < M; ++i) {
      Eigen::Vector3d dp = v1[i] - v0[i];
      double theta0 = atan2(dp.y(), dp.x());
      double r = dp.norm();
      geometry_msgs::Point center;
      center.x = v0[i].x();
      center.y = v0[i].y();
      center.z = v0[i].z();
      geometry_msgs::Point p = center;
      p.x += r * cos(theta0 - thetas[i]);
      p.y += r * sin(theta0 - thetas[i]);
      for (double theta = theta0 - thetas[i] + 0.1; theta < theta0 + thetas[i]; theta += 0.1) {
        marker.points.push_back(center);
        marker.points.push_back(p);
        p = center;
        p.x += r * cos(theta);
        p.y += r * sin(theta);
        marker.points.push_back(p);
      }
    }
    publisher_map_[topic].publish(marker);
  }

  template <class ARROWS, class TOPIC>
  // ARROWS: pair<Vector3d, Vector3d>
  void visualize_arrows(const ARROWS& arrows, const TOPIC& topic) {
    auto got = publisher_map_.find(topic);
    if (got == publisher_map_.end()) {
      ros::Publisher pub =
          nh_.advertise<visualization_msgs::MarkerArray>(topic, 10);
      publisher_map_[topic] = pub;
    }
    visualization_msgs::Marker clear_previous_msg;
    clear_previous_msg.action = visualization_msgs::Marker::DELETEALL;
    visualization_msgs::Marker arrow_msg;
    arrow_msg.type = visualization_msgs::Marker::ARROW;
    arrow_msg.action = visualization_msgs::Marker::ADD;
    arrow_msg.header.frame_id = "world";
    arrow_msg.id = 0;
    arrow_msg.points.resize(2);
    setMarkerPose(arrow_msg, 0, 0, 0);
    setMarkerScale(arrow_msg, 0.02, 0.05, 0);
    setMarkerColor(arrow_msg, yellow);
    visualization_msgs::MarkerArray arrow_list_msg;
    arrow_list_msg.markers.reserve(1 + arrows.size());
    arrow_list_msg.markers.push_back(clear_previous_msg);
    for (const auto& arrow : arrows) {
      arrow_msg.points[0].x = arrow.first[0];
      arrow_msg.points[0].y = arrow.first[1];
      arrow_msg.points[0].z = arrow.first[2];
      arrow_msg.points[1].x = arrow.second[0];
      arrow_msg.points[1].y = arrow.second[1];
      arrow_msg.points[1].z = arrow.second[2];
      arrow_list_msg.markers.push_back(arrow_msg);
      arrow_msg.id += 1;
    }
    publisher_map_[topic].publish(arrow_list_msg);
  }

  template <class TRAJ, class TOPIC>
  // TRAJ:
  void visualize_traj(const TRAJ& traj, const TOPIC& topic) {
    std::vector<Eigen::Vector3d> path;
    auto duration = traj.getTotalDuration();
    for (double t = 0; t < duration; t += 0.01) {
      path.push_back(traj.getPos(t));
    }
    visualize_path(path, topic);
    std::vector<Eigen::Vector3d> wayPts;
    for (const auto& piece : traj) {
      wayPts.push_back(piece.getPos(0));
    }
    visualize_pointcloud(wayPts, std::string(topic) + "_wayPts");
  }

  template <class TRAJLIST, class TOPIC>
  // TRAJLIST: std::vector<TRAJ>
  void visualize_traj_list(const TRAJLIST& traj_list, const TOPIC& topic) {
    auto got = publisher_map_.find(topic);
    if (got == publisher_map_.end()) {
      ros::Publisher pub =
          nh_.advertise<visualization_msgs::MarkerArray>(topic, 10);
      publisher_map_[topic] = pub;
    }
    visualization_msgs::Marker clear_previous_msg;
    clear_previous_msg.action = visualization_msgs::Marker::DELETEALL;
    visualization_msgs::Marker path_msg;
    path_msg.type = visualization_msgs::Marker::LINE_STRIP;
    path_msg.action = visualization_msgs::Marker::ADD;
    path_msg.header.frame_id = "world";
    path_msg.id = 0;
    setMarkerPose(path_msg, 0, 0, 0);
    setMarkerScale(path_msg, 0.02, 0.05, 0);
    visualization_msgs::MarkerArray path_list_msg;
    path_list_msg.markers.reserve(1 + traj_list.size());
    path_list_msg.markers.push_back(clear_previous_msg);
    double a_step = 0.8 / traj_list.size();
    double a = 0.1;
    geometry_msgs::Point p_msg;
    for (const auto& traj : traj_list) {
      setMarkerColor(path_msg, white, a);
      a = a + a_step;
      path_msg.points.clear();
      for (double t = 0; t < traj.getTotalDuration(); t += 0.01) {
        auto p = traj.getPos(t);
        p_msg.x = p.x();
        p_msg.y = p.y();
        p_msg.z = p.z();
        path_msg.points.push_back(p_msg);
      }
      path_list_msg.markers.push_back(path_msg);
      path_msg.id += 1;
    }
    publisher_map_[topic].publish(path_list_msg);
  }

  template <class PATHLIST, class TOPIC>
  // PATHLIST: std::vector<PATH>
  void visualize_path_list(const PATHLIST& path_list, const TOPIC& topic) {
    auto got = publisher_map_.find(topic);
    if (got == publisher_map_.end()) {
      ros::Publisher pub =
          nh_.advertise<visualization_msgs::MarkerArray>(topic, 10);
      publisher_map_[topic] = pub;
    }
    visualization_msgs::Marker clear_previous_msg;
    clear_previous_msg.action = visualization_msgs::Marker::DELETEALL;
    visualization_msgs::Marker path_msg;
    path_msg.type = visualization_msgs::Marker::LINE_STRIP;
    path_msg.action = visualization_msgs::Marker::ADD;
    path_msg.header.frame_id = "world";
    path_msg.id = 0;
    setMarkerPose(path_msg, 0, 0, 0);
    setMarkerScale(path_msg, 0.02, 0.05, 0);
    visualization_msgs::MarkerArray path_list_msg;
    path_list_msg.markers.reserve(1 + path_list.size());
    path_list_msg.markers.push_back(clear_previous_msg);
    setMarkerColor(path_msg, greenblue);
    for (const auto& path : path_list) {
      path_msg.points.resize(path.size());
      for (size_t i = 0; i < path.size(); ++i) {
        path_msg.points[i].x = path[i].x();
        path_msg.points[i].y = path[i].y();
        path_msg.points[i].z = path[i].z();
      }
      path_list_msg.markers.push_back(path_msg);
      path_msg.id += 1;
    }
    publisher_map_[topic].publish(path_list_msg);
  }
};

}  // namespace visualization
