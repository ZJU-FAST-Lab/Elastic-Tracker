#include <quadrotor_msgs/OccMap3d.h>

#include <prediction/prediction.hpp>
#include <visualization/visualization.hpp>

// int main(int argc, char const *argv[]) {
//   pre_bspline::Bspline bspline;
//   bspline.setup(20, 1.1);
//   Eigen::VectorXd N = bspline.calCoeff(4.0);
//   std::cout << "N: " << N.transpose() << std::endl;

//   pre_bspline::PreBspline predict;
//   return 0;
// }

int data_i = 0;
std::shared_ptr<prediction::Predict> prePtr_;
std::shared_ptr<visualization::Visualization> visPtr_;
mapping::OccGridMap map_;
bool map_received_ = false;
bool triger_received_ = false;

Eigen::Vector3d target_p_, target_v_;

void gridmap_callback(const quadrotor_msgs::OccMap3dConstPtr &msgPtr) {
  if (map_received_) {
    return;
  }
  map_.from_msg(*msgPtr);
  prePtr_->setMap(map_);
  ROS_WARN("[TEST NODE] GLOBAL MAP RECEIVED");
  map_received_ = true;
}

void testCallback(const ros::TimerEvent &e) {
}

void triger_callback(const geometry_msgs::PoseStampedConstPtr &msgPtr) {
  target_p_ << msgPtr->pose.position.x, msgPtr->pose.position.y, 1.0;
  Eigen::Quaterniond q;
  q.w() = msgPtr->pose.orientation.w;
  q.x() = msgPtr->pose.orientation.x;
  q.y() = msgPtr->pose.orientation.y;
  q.z() = msgPtr->pose.orientation.z;
  // target_v_ = q.toRotationMatrix() * Eigen::Vector3d(1, 0, 0);
  target_v_ = Eigen::Vector3d(1, 0, 0);

  std::vector<Eigen::Vector3d> predict_path;
  std::cout << "target_p: " << target_p_.transpose() << std::endl;
  std::cout << "target_v: " << target_v_.transpose() << std::endl;
  prePtr_->predict(target_p_, target_v_, predict_path);
  visPtr_->visualize_pointcloud(predict_path, "future_pts");
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "test_node");
  ros::NodeHandle nh("~");

  ros::Timer test_timer = nh.createTimer(ros::Duration(0.3), testCallback);
  ros::Subscriber gridmap_sub_ = nh.subscribe<quadrotor_msgs::OccMap3d>(
      "gridmap_inflate", 1, &gridmap_callback, ros::TransportHints().tcpNoDelay());
  ros::Subscriber triger_sub_ = nh.subscribe<geometry_msgs::PoseStamped>(
      "triger", 10, &triger_callback, ros::TransportHints().tcpNoDelay());

  ROS_WARN("[TEST NODE]: ready.");

  prePtr_ = std::make_shared<prediction::Predict>(nh);
  visPtr_ = std::make_shared<visualization::Visualization>(nh);

  ros::spin();

  return 0;
}