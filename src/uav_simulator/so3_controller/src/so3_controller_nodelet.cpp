#include "so3_controller/so3_controller.hpp"
#include <quadrotor_msgs/PositionCommand.h>
#include <quadrotor_msgs/SO3Command.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <tf/transform_datatypes.h>
#include <nodelet/nodelet.h>
#include <ros/ros.h>

namespace so3_controller {
class Nodelet : public nodelet::Nodelet{
 private:
  std::shared_ptr<SO3Controller> so3ControlPtr_;
  ros::Subscriber position_cmd_sub_, odom_sub_, imu_sub_;
  ros::Publisher  so3cmd_pub_;
  ros::Timer state_timer_;
  quadrotor_msgs::SO3Command so3cmd_;

  bool position_cmd_received_flag_ = false;
  Eigen::Vector3d des_pos_;
  double des_yaw_;

  void timer_callback(const ros::TimerEvent& event) {
    if (position_cmd_received_flag_) {
      position_cmd_received_flag_ = false;
    } else {
      Eigen::Vector3d des_vel(0, 0, 0);
      Eigen::Vector3d des_acc(0, 0, 0);
      Eigen::Vector3d kx(5.7, 5.7, 6.2);
      Eigen::Vector3d kv(3.4, 3.4, 4.0);
      so3ControlPtr_->calculateControl(des_pos_, des_vel, des_acc, 
                                       des_yaw_, 0, kx, kv);
      const Eigen::Vector3d& f = so3ControlPtr_->getF();
      const Eigen::Quaterniond& q = so3ControlPtr_->getQ();
      so3cmd_.force.x = f(0);
      so3cmd_.force.y = f(1);
      so3cmd_.force.z = f(2);
      so3cmd_.orientation.x = q.x();
      so3cmd_.orientation.y = q.y();
      so3cmd_.orientation.z = q.z();
      so3cmd_.orientation.w = q.w();
      so3cmd_pub_.publish(so3cmd_);
    }
  }

  void position_cmd_callback(const quadrotor_msgs::PositionCommand::ConstPtr& msg) {
    position_cmd_received_flag_ = true;
    Eigen::Vector3d des_pos(msg->position.x, msg->position.y, msg->position.z);
    Eigen::Vector3d des_vel(msg->velocity.x, msg->velocity.y, msg->velocity.z);
    Eigen::Vector3d des_acc(msg->acceleration.x, msg->acceleration.y, msg->acceleration.z);
    Eigen::Vector3d kx(msg->kx[0], msg->kx[1], msg->kx[2]);
    Eigen::Vector3d kv(msg->kv[0], msg->kv[1], msg->kv[2]);
    if (msg->kx[0] == 0) {
      kx(0) = 5.7; kx(1) = 5.7; kx(2) = 6.2;
      kv(0) = 3.4; kv(1) = 3.4; kv(2) = 4.0;
    }
    double des_yaw = msg->yaw;
    double des_yaw_dot = msg->yaw_dot;
    so3ControlPtr_->calculateControl(des_pos, des_vel, des_acc, 
                                     des_yaw, des_yaw_dot, kx, kv);
    const Eigen::Vector3d& f = so3ControlPtr_->getF();
    const Eigen::Quaterniond& q = so3ControlPtr_->getQ();
    so3cmd_.force.x = f(0);
    so3cmd_.force.y = f(1);
    so3cmd_.force.z = f(2);
    so3cmd_.orientation.x = q.x();
    so3cmd_.orientation.y = q.y();
    so3cmd_.orientation.z = q.z();
    so3cmd_.orientation.w = q.w();
    so3cmd_pub_.publish(so3cmd_);
    // store last des_pos and des_yaw
    des_pos_ = des_pos;
    des_yaw_ = des_yaw;
  }
  void odom_callback(const nav_msgs::Odometry::ConstPtr& msg) {
    Eigen::Vector3d pos(msg->pose.pose.position.x, 
                        msg->pose.pose.position.y,
                        msg->pose.pose.position.z);
    Eigen::Vector3d vel(msg->twist.twist.linear.x,
                        msg->twist.twist.linear.y,
                        msg->twist.twist.linear.z);
    so3ControlPtr_->setPos(pos);
    so3ControlPtr_->setVel(vel);
    so3cmd_.aux.current_yaw = tf::getYaw(msg->pose.pose.orientation);
    des_pos_ = pos;
    des_yaw_ = tf::getYaw(msg->pose.pose.orientation);
  }
  void imu_callback(const sensor_msgs::Imu::ConstPtr& msg) {
    Eigen::Vector3d acc(msg->linear_acceleration.x,
                        msg->linear_acceleration.y,
                        msg->linear_acceleration.z);
    so3ControlPtr_->setAcc(acc);
  }
 public:
  void onInit(void) {
    ros::NodeHandle nh(getMTPrivateNodeHandle());
    // parameters
    double mass, g;
    nh.getParam("mass", mass);
    nh.getParam("g", g);
    so3ControlPtr_ = std::make_shared<SO3Controller>(mass, g);
    so3cmd_.header.frame_id = "world";
    nh.getParam("gains/rot/x", so3cmd_.kR[0]);
    nh.getParam("gains/rot/y", so3cmd_.kR[1]);
    nh.getParam("gains/rot/z", so3cmd_.kR[2]);
    nh.getParam("gains/ang/x", so3cmd_.kOm[0]);
    nh.getParam("gains/ang/y", so3cmd_.kOm[1]);
    nh.getParam("gains/ang/z", so3cmd_.kOm[2]);
    nh.getParam("corrections/z", so3cmd_.aux.kf_correction);
    nh.getParam("corrections/r", so3cmd_.aux.angle_corrections[0]);
    nh.getParam("corrections/p", so3cmd_.aux.angle_corrections[1]);
    so3cmd_.aux.enable_motors = true;
    so3cmd_.aux.use_external_yaw = false;
    position_cmd_sub_ = nh.subscribe("position_cmd", 10, &Nodelet::position_cmd_callback, 
                                     this, ros::TransportHints().tcpNoDelay());
    odom_sub_ = nh.subscribe("odom", 10, &Nodelet::odom_callback, this, ros::TransportHints().tcpNoDelay());
    imu_sub_ = nh.subscribe("imu", 10, &Nodelet::imu_callback, this, ros::TransportHints().tcpNoDelay());
    so3cmd_pub_ = nh.advertise<quadrotor_msgs::SO3Command>("so3cmd", 10);
    state_timer_ = nh.createTimer(ros::Duration(0.1), &Nodelet::timer_callback, this);
  }
};

} // so3_controller

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(so3_controller::Nodelet, nodelet::Nodelet);
