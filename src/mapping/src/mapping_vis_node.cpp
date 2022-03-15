#include <mapping/mapping.h>
#include <quadrotor_msgs/OccMap3d.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

ros::Publisher gridmap_vs_pub, gridmap_inflate_vs_pub;
bool remove_floor_ceil_ = false;

void gridmap_callback(const quadrotor_msgs::OccMap3dConstPtr& msgPtr) {
  mapping::OccGridMap gridmap;
  gridmap.from_msg(*msgPtr);
  sensor_msgs::PointCloud2 pc;
  gridmap.occ2pc(pc);
  pc.header.frame_id = "world";
  gridmap_vs_pub.publish(pc);
}
void gridmap_inflate_callback(const quadrotor_msgs::OccMap3dConstPtr& msgPtr) {
  mapping::OccGridMap gridmap;
  gridmap.from_msg(*msgPtr);
  sensor_msgs::PointCloud2 pc;
  if (remove_floor_ceil_) {
    gridmap.occ2pc(pc, 0.5, 2.5);
  } else {
    gridmap.occ2pc(pc);
  }
  pc.header.frame_id = "world";
  gridmap_inflate_vs_pub.publish(pc);
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "mapping_vis");
  ros::NodeHandle nh("~");
  nh.getParam("remove_floor_ceil", remove_floor_ceil_);
  gridmap_vs_pub = nh.advertise<sensor_msgs::PointCloud2>("vs_gridmap", 1);
  gridmap_inflate_vs_pub = nh.advertise<sensor_msgs::PointCloud2>("vs_gridmap_inflate", 1);
  ros::Subscriber gridmap_sub = nh.subscribe<quadrotor_msgs::OccMap3d>("gridmap", 1, gridmap_callback);
  ros::Subscriber gridmap_inflate_sub = nh.subscribe<quadrotor_msgs::OccMap3d>("gridmap_inflate", 1, gridmap_inflate_callback);

  ros::spin();
  return 0;
}
