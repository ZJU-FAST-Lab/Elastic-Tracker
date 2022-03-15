#include "so3_quadrotor/quadrotor_dynamics.hpp"
#include <quadrotor_msgs/SO3Command.h>
#include <nodelet/nodelet.h>
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf/transform_broadcaster.h>

namespace so3_quadrotor {
class Nodelet : public nodelet::Nodelet {
 private:
  std::shared_ptr<Quadrotor> quadrotorPtr_;
  Control control_;
  Cmd cmd_;
  ros::Publisher odom_pub_, imu_pub_, vis_pub_;
  ros::Subscriber cmd_sub_;
  ros::Timer simulation_timer;
  tf::TransformBroadcaster tf_br_;

  // parameters
  int simulation_rate_ = 1e3;
  int odom_rate_ = 400;
  // msgs
  nav_msgs::Odometry odom_msg_;
  sensor_msgs::Imu imu_msg_;
  visualization_msgs::MarkerArray vis_msg_;
  double motor_yaw_[4] = {0,0,0,0};


  void cmd_callback(const quadrotor_msgs::SO3Command::ConstPtr& cmd_msg) {
    cmd_.force[0]         = cmd_msg->force.x;
    cmd_.force[1]         = cmd_msg->force.y;
    cmd_.force[2]         = cmd_msg->force.z;
    cmd_.qx               = cmd_msg->orientation.x;
    cmd_.qy               = cmd_msg->orientation.y;
    cmd_.qz               = cmd_msg->orientation.z;
    cmd_.qw               = cmd_msg->orientation.w;
    cmd_.kR[0]            = cmd_msg->kR[0];
    cmd_.kR[1]            = cmd_msg->kR[1];
    cmd_.kR[2]            = cmd_msg->kR[2];
    cmd_.kOm[0]           = cmd_msg->kOm[0];
    cmd_.kOm[1]           = cmd_msg->kOm[1];
    cmd_.kOm[2]           = cmd_msg->kOm[2];
    cmd_.corrections[0]   = cmd_msg->aux.kf_correction;
    cmd_.corrections[1]   = cmd_msg->aux.angle_corrections[0];
    cmd_.corrections[2]   = cmd_msg->aux.angle_corrections[1];
    cmd_.current_yaw      = cmd_msg->aux.current_yaw;
    cmd_.use_external_yaw = cmd_msg->aux.use_external_yaw;
  }
  void timer_callback(const ros::TimerEvent& event) {
    auto last_control = control_;
    control_ = quadrotorPtr_->getControl(cmd_);
    for (size_t i = 0; i < 4; ++i) {
      if (std::isnan(control_.rpm[i]))
        control_.rpm[i] = last_control.rpm[i];
    }
    quadrotorPtr_->setInput(control_.rpm[0], control_.rpm[1], control_.rpm[2], control_.rpm[3]);
    quadrotorPtr_->step(1.0/simulation_rate_);
    static ros::Time next_odom_pub_time = ros::Time::now();
    ros::Time tnow = ros::Time::now();
    if (tnow >= next_odom_pub_time) {
      next_odom_pub_time += ros::Duration(1.0/odom_rate_);
      const Eigen::Vector3d&     pos = quadrotorPtr_->getPos();
      const Eigen::Vector3d&     vel = quadrotorPtr_->getVel();
      const Eigen::Vector3d&     acc = quadrotorPtr_->getAcc();
      const Eigen::Quaterniond& quat = quadrotorPtr_->getQuat();
      const Eigen::Vector3d&   omega = quadrotorPtr_->getOmega();
      // tf
      tf::Transform transform;
      transform.setOrigin( tf::Vector3(pos.x(), pos.y(), pos.z()) );
      transform.setRotation( tf::Quaternion(quat.x(), quat.y(), quat.z(), quat.w()) );
      tf_br_.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "body"));
      // odom
      odom_msg_.header.stamp = tnow;
      odom_msg_.pose.pose.position.x = pos(0);
      odom_msg_.pose.pose.position.y = pos(1);
      odom_msg_.pose.pose.position.z = pos(2);
      odom_msg_.pose.pose.orientation.x = quat.x();
      odom_msg_.pose.pose.orientation.y = quat.y();
      odom_msg_.pose.pose.orientation.z = quat.z();
      odom_msg_.pose.pose.orientation.w = quat.w();

      odom_msg_.twist.twist.linear.x = vel(0);
      odom_msg_.twist.twist.linear.y = vel(1);
      odom_msg_.twist.twist.linear.z = vel(2);

      odom_msg_.twist.twist.angular.x = omega(0);
      odom_msg_.twist.twist.angular.y = omega(1);
      odom_msg_.twist.twist.angular.z = omega(2);
      odom_pub_.publish(odom_msg_);
      // imu
      imu_msg_.header.stamp = tnow;
      imu_msg_.orientation.x = quat.x();
      imu_msg_.orientation.y = quat.y();
      imu_msg_.orientation.z = quat.z();
      imu_msg_.orientation.w = quat.w();

      imu_msg_.angular_velocity.x = omega(0);
      imu_msg_.angular_velocity.y = omega(1);
      imu_msg_.angular_velocity.z = omega(2);

      imu_msg_.linear_acceleration.x = acc[0];
      imu_msg_.linear_acceleration.y = acc[1];
      imu_msg_.linear_acceleration.z = acc[2];
      imu_pub_.publish(imu_msg_);
      // drone visualization
      std::vector<Eigen::Vector3d> propellers;
      propellers.emplace_back(0, +quadrotorPtr_->config.arm_length/2, 0.02);
      propellers.emplace_back(0, -quadrotorPtr_->config.arm_length/2, 0.02);
      propellers.emplace_back(-quadrotorPtr_->config.arm_length/2, 0, 0.02);
      propellers.emplace_back(+quadrotorPtr_->config.arm_length/2, 0, 0.02);
      for (size_t i=0; i<4; ++i) {
        double rpm = quadrotorPtr_->state.motor_rpm.coeff(i);
        if (i/2) {
          motor_yaw_[i] += rpm*60 / odom_rate_;
        } else {
          motor_yaw_[i] -= rpm*60 / odom_rate_;
        }
        motor_yaw_[i] = std::fmod(motor_yaw_[i], 2*M_PI);
        Eigen::Quaterniond quat_propeller = quat * uav_utils::ypr_to_quaternion(
          Eigen::Vector3d(motor_yaw_[i], 0, 0));
        Eigen::Vector3d pos_propeller = pos + quat.toRotationMatrix() * propellers[i];

        vis_msg_.markers[i].pose.position.x = pos_propeller(0);
        vis_msg_.markers[i].pose.position.y = pos_propeller(1);
        vis_msg_.markers[i].pose.position.z = pos_propeller(2);
        vis_msg_.markers[i].pose.orientation.x = quat_propeller.x();
        vis_msg_.markers[i].pose.orientation.y = quat_propeller.y();
        vis_msg_.markers[i].pose.orientation.z = quat_propeller.z();
        vis_msg_.markers[i].pose.orientation.w = quat_propeller.w();
      }
      for (size_t i=4; i<vis_msg_.markers.size(); ++i) {
        vis_msg_.markers[i].pose = odom_msg_.pose.pose;
        vis_msg_.markers[i].id = i;
      }
      vis_pub_.publish(vis_msg_);
    }
  }
 public:
  void onInit(void) {
    ros::NodeHandle nh(getMTPrivateNodeHandle());
    // parameters
    double init_x, init_y, init_z, init_yaw;
    nh.getParam("init_x", init_x);
    nh.getParam("init_y", init_y);
    nh.getParam("init_z", init_z);
    nh.getParam("init_yaw", init_yaw);
    // config of quadrotor
    Config config;
    nh.getParam("g",       config.g);
    nh.getParam("mass", config.mass);
    double Ixx, Iyy, Izz;
    nh.getParam("Ixx", Ixx);
    nh.getParam("Iyy", Iyy);
    nh.getParam("Izz", Izz);
    config.J = Eigen::Vector3d(Ixx, Iyy, Izz).asDiagonal();
    nh.getParam("kf", config.kf);
    double prop_radius;
    nh.getParam("prop_radius", prop_radius);
    config.km = 0.07 * (3 * prop_radius) * config.kf;
    nh.getParam("arm_length", config.arm_length);
    nh.getParam("motor_time_constant", config.motor_time_constant);
    nh.getParam("max_rpm", config.max_rpm);
    nh.getParam("min_rpm", config.min_rpm);
    nh.getParam("simulation_rate", simulation_rate_);
    nh.getParam("odom_rate", odom_rate_);

    quadrotorPtr_ = std::make_shared<Quadrotor>(config);
    quadrotorPtr_->setPos(Eigen::Vector3d(init_x, init_y, init_z));
    quadrotorPtr_->setYpr(Eigen::Vector3d(init_yaw, 0, 0));
    double rpm = sqrt(config.mass * config.g / 4 / config.kf);
    quadrotorPtr_->setRpm(Eigen::Vector4d(rpm, rpm, rpm, rpm));
    
    Eigen::Quaterniond quat = uav_utils::ypr_to_quaternion(Eigen::Vector3d(init_yaw, 0, 0));
    cmd_.force[2] =  config.mass * config.g;
    cmd_.qw = quat.w();
    cmd_.qx = quat.x();
    cmd_.qy = quat.y();
    cmd_.qz = quat.z();

    odom_pub_ = nh.advertise<nav_msgs::Odometry>("odom", 100);
    imu_pub_  = nh.advertise<sensor_msgs::Imu>("imu", 10);
    vis_pub_= nh.advertise<visualization_msgs::MarkerArray>("vis", 10);
    cmd_sub_  = nh.subscribe<quadrotor_msgs::SO3Command>("so3cmd", 10, &Nodelet::cmd_callback, this, ros::TransportHints().tcpNoDelay());
    simulation_timer = nh.createTimer(ros::Duration(1.0/simulation_rate_), &Nodelet::timer_callback, this);

    odom_msg_.header.frame_id = "world";
    imu_msg_ .header.frame_id = "world";
    vis_msg_.markers.resize(4);
    visualization_msgs::Marker propeller;
    propeller.header.frame_id = "world";
    propeller.type = visualization_msgs::Marker::CYLINDER;
    propeller.action = visualization_msgs::Marker::ADD;
    propeller.scale.x = 2 * prop_radius;
    propeller.scale.y = 0.1 * prop_radius;
    propeller.scale.z = 0.05 * prop_radius;
    propeller.color.a = 1;
    propeller.color.r = 1;
    propeller.color.g = 0.5;
    propeller.color.b = 0;
    for (size_t i=0; i<4; ++i) {
      vis_msg_.markers[i] = propeller;
      vis_msg_.markers[i].id = i;
    }
    visualization_msgs::Marker drone_body = propeller;
    drone_body.type = visualization_msgs::Marker::SPHERE;
    drone_body.color.g = 0.7;
    drone_body.scale.x = config.arm_length;
    drone_body.scale.y = 0.1 * config.arm_length;
    drone_body.scale.z = 0.05 * config.arm_length;
    vis_msg_.markers.push_back(drone_body);
    drone_body.scale.y = config.arm_length;
    drone_body.scale.x = 0.1 * config.arm_length;
    vis_msg_.markers.push_back(drone_body);
  };
};

} // so3_quadrotor

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(so3_quadrotor::Nodelet, nodelet::Nodelet);
