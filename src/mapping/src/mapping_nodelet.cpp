#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <mapping/mapping.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>
#include <nav_msgs/Odometry.h>
#include <nodelet/nodelet.h>
#include <pcl_conversions/pcl_conversions.h>
#include <quadrotor_msgs/OccMap3d.h>
#include <ros/ros.h>
#include <std_msgs/Int16MultiArray.h>

#include <atomic>
#include <thread>

namespace mapping {

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry>
    ImageOdomSyncPolicy;
typedef message_filters::Synchronizer<ImageOdomSyncPolicy>
    ImageOdomSynchronizer;

struct CamConfig {
  // camera paramters
  double rate;
  double range;
  int width;
  int height;
  double fx;
  double fy;
  double cx;
  double cy;
  double depth_scaling_factor;
};

class Nodelet : public nodelet::Nodelet {
 private:
  std::thread initThread_;
  CamConfig camConfig_;  // just store the parameters of camera

  int down_sample_factor_;

  Eigen::Matrix3d cam2body_R_;
  Eigen::Vector3d cam2body_p_;

  // just for depth filter
  Eigen::Vector3d last_cam_p_;
  Eigen::Quaterniond last_cam_q_;
  bool get_first_frame_ = false;
  cv::Mat last_depth_;
  double depth_filter_tolerance_;
  double depth_filter_mindist_;
  int depth_filter_margin_;

  std::atomic_flag callback_lock_ = ATOMIC_FLAG_INIT;
  message_filters::Subscriber<sensor_msgs::Image> depth_sub_;
  message_filters::Subscriber<nav_msgs::Odometry> odom_sub_;
  std::shared_ptr<ImageOdomSynchronizer> depth_odom_sync_Ptr_;
  ros::Publisher gridmap_inflate_pub_, local_pc_pub_, pcl_pub_;

  // NOTE just for global map in simulation
  ros::Timer global_map_timer_;
  ros::Subscriber map_pc_sub_;
  bool map_recieved_ = false;
  bool use_global_map_ = false;

  // NOTE for mask target
  bool use_mask_ = false;
  ros::Subscriber target_odom_sub_;
  std::atomic_flag target_lock_ = ATOMIC_FLAG_INIT;
  Eigen::Vector3d target_odom_;

  OccGridMap gridmap_;
  int inflate_size_;

  void depth_odom_callback(const sensor_msgs::ImageConstPtr& depth_msg,
                           const nav_msgs::OdometryConstPtr& odom_msg) {
    if (callback_lock_.test_and_set()) {
      return;
    }
    ros::Time t1, t2;
    // t1 = ros::Time::now();
    Eigen::Vector3d body_p(odom_msg->pose.pose.position.x,
                           odom_msg->pose.pose.position.y,
                           odom_msg->pose.pose.position.z);
    Eigen::Quaterniond body_q(
        odom_msg->pose.pose.orientation.w, odom_msg->pose.pose.orientation.x,
        odom_msg->pose.pose.orientation.y, odom_msg->pose.pose.orientation.z);
    Eigen::Vector3d cam_p = body_q.toRotationMatrix() * cam2body_p_ + body_p;
    Eigen::Quaterniond cam_q = body_q * Eigen::Quaterniond(cam2body_R_);
    cv_bridge::CvImagePtr depth_ptr = cv_bridge::toCvCopy(depth_msg);
    if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
      (depth_ptr->image).convertTo(depth_ptr->image, CV_16UC1, camConfig_.depth_scaling_factor);
    }
    cv::Mat depth_img = depth_ptr->image;

    // pub target

    int nr = depth_img.rows;
    int nc = depth_img.cols;
    std::vector<Eigen::Vector3d> obs_pts;
    // put the points of the depth into the list of obs_points

    // TODO depth filter

    // t1 = ros::Time::now();
    for (int i = depth_filter_margin_; i < nr - depth_filter_margin_; i += down_sample_factor_) {
      for (int j = depth_filter_margin_; j < nc - depth_filter_margin_; j += down_sample_factor_) {
        // (x,y,z) in camera frame
        double z = (depth_img.at<uint16_t>(i, j)) / camConfig_.depth_scaling_factor;
        if (depth_img.at<uint16_t>(i, j) == 0) {
          z = camConfig_.range + 0.5;
        }
        if (std::isnan(z) || std::isinf(z))
          continue;
        if (z < depth_filter_mindist_) {
          continue;
        }
        double y = (i - camConfig_.cy) * z / camConfig_.fy;
        double x = (j - camConfig_.cx) * z / camConfig_.fx;
        Eigen::Vector3d p(x, y, z);
        p = cam_q * p + cam_p;
        bool good_point = true;
        if (get_first_frame_) {
          // NOTE depth filter:
          Eigen::Vector3d p_rev_proj =
              last_cam_q_.inverse().toRotationMatrix() * (p - last_cam_p_);
          double vv = p_rev_proj.y() * camConfig_.fy / p_rev_proj.z() + camConfig_.cy;
          double uu = p_rev_proj.x() * camConfig_.fx / p_rev_proj.z() + camConfig_.cx;
          if (vv >= 0 && vv < nr && uu >= 0 && uu < nc) {
            double drift_dis = fabs(last_depth_.at<uint16_t>((int)vv, (int)uu) / camConfig_.depth_scaling_factor - p_rev_proj.z());
            if (drift_dis > depth_filter_tolerance_) {
              good_point = false;
            }
          }
        }
        if (good_point) {
          obs_pts.push_back(p);
        }
      }
    }
    last_depth_ = depth_img;
    last_cam_p_ = cam_p;
    last_cam_q_ = cam_q;
    get_first_frame_ = true;
    gridmap_.updateMap(cam_p, obs_pts);

    // NOTE use mask
    if (use_mask_) {  // mask target
      while (target_lock_.test_and_set())
        ;
      Eigen::Vector3d ld = target_odom_;
      Eigen::Vector3d ru = target_odom_;
      ld.x() -= 0.5;
      ld.y() -= 0.5;
      ld.z() -= 1.0;
      ru.x() += 0.5;
      ru.y() += 0.5;
      ru.z() += 1.0;
      gridmap_.setFree(ld, ru);
      target_lock_.clear();
    }

    // TODO pub local map
    // sensor_msgs::PointCloud2 pc_msg;
    // pcl::PointCloud<pcl::PointXYZ> pcd;
    // pcl::PointXYZ pt;
    // for (const auto p : mask_pts_) {
    //   pt.x = p.x();
    //   pt.y = p.y();
    //   pt.z = p.z();
    //   pcd.push_back(pt);
    // }
    // pcd.width = pcd.points.size();
    // pcd.height = 1;
    // pcd.is_dense = true;
    // pcl::toROSMsg(pcd, pc_msg);
    // pc_msg.header.frame_id = "world";
    // local_pc_pub_.publish(pc_msg);

    quadrotor_msgs::OccMap3d gridmap_msg;
    gridmap_msg.header.frame_id = "world";
    gridmap_msg.header.stamp = ros::Time::now();
    gridmap_.to_msg(gridmap_msg);
    gridmap_inflate_pub_.publish(gridmap_msg);

    callback_lock_.clear();
  }

  // NOTE
  void target_odom_callback(const nav_msgs::OdometryConstPtr& msgPtr) {
    while (target_lock_.test_and_set())
      ;
    target_odom_.x() = msgPtr->pose.pose.position.x;
    target_odom_.y() = msgPtr->pose.pose.position.y;
    target_odom_.z() = msgPtr->pose.pose.position.z;
    target_lock_.clear();
  }

  // NOTE just for global map in simulation
  void map_call_back(const sensor_msgs::PointCloud2ConstPtr& msgPtr) {
    if (map_recieved_) {
      return;
    }
    pcl::PointCloud<pcl::PointXYZ> point_cloud;
    pcl::fromROSMsg(*msgPtr, point_cloud);
    for (const auto& pt : point_cloud) {
      Eigen::Vector3d p(pt.x, pt.y, pt.z);
      gridmap_.setOcc(p);
    }
    gridmap_.inflate(inflate_size_);
    ROS_WARN("[mapping] GLOBAL MAP REVIEVED!");
    map_recieved_ = true;
    return;
  }
  void global_map_timer_callback(const ros::TimerEvent& event) {
    if (!map_recieved_) {
      return;
    }
    quadrotor_msgs::OccMap3d gridmap_msg;
    gridmap_.to_msg(gridmap_msg);
    gridmap_inflate_pub_.publish(gridmap_msg);
  }

  void init(ros::NodeHandle& nh) {
    // set parameters of mapping
    // cam2body_R_ << 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0;
    // cam2body_p_.setZero();
    std::vector<double> tmp;
    if (nh.param<std::vector<double>>("cam2body_R", tmp, std::vector<double>())) {
      cam2body_R_ = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(tmp.data(), 3, 3);
    }
    if (nh.param<std::vector<double>>("cam2body_p", tmp, std::vector<double>())) {
      cam2body_p_ = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(tmp.data(), 3, 1);
    }
    // std::cout << "R: \n" << cam2body_R_ << std::endl;
    // std::cout << "p: \n" << cam2body_p_ << std::endl;
    double res;
    Eigen::Vector3d map_size;
    // NOTE whether to use global map
    nh.getParam("use_global_map", use_global_map_);
    if (use_global_map_) {
      double x, y, z, res;
      nh.getParam("x_length", x);
      nh.getParam("y_length", y);
      nh.getParam("z_length", z);
      nh.getParam("resolution", res);
      nh.getParam("inflate_size", inflate_size_);
      gridmap_.setup(res, Eigen::Vector3d(x, y, z), 10, true);
    } else {
      // camera parameters
      nh.getParam("camera_rate", camConfig_.rate);
      nh.getParam("camera_range", camConfig_.range);
      nh.getParam("cam_width", camConfig_.width);
      nh.getParam("cam_height", camConfig_.height);
      nh.getParam("cam_fx", camConfig_.fx);
      nh.getParam("cam_fy", camConfig_.fy);
      nh.getParam("cam_cx", camConfig_.cx);
      nh.getParam("cam_cy", camConfig_.cy);
      nh.getParam("depth_scaling_factor", camConfig_.depth_scaling_factor);
      // mapping parameters
      nh.getParam("down_sample_factor", down_sample_factor_);
      nh.getParam("resolution", res);
      nh.getParam("local_x", map_size.x());
      nh.getParam("local_y", map_size.y());
      nh.getParam("local_z", map_size.z());
      nh.getParam("inflate_size", inflate_size_);
      gridmap_.setup(res, map_size, camConfig_.range);
      // depth filter parameters
      nh.getParam("depth_filter_tolerance", depth_filter_tolerance_);
      nh.getParam("depth_filter_mindist", depth_filter_mindist_);
      nh.getParam("depth_filter_margin", depth_filter_margin_);
      // raycasting parameters
      int p_min, p_max, p_hit, p_mis, p_occ, p_def;
      nh.getParam("p_min", p_min);
      nh.getParam("p_max", p_max);
      nh.getParam("p_hit", p_hit);
      nh.getParam("p_mis", p_mis);
      nh.getParam("p_occ", p_occ);
      nh.getParam("p_def", p_def);
      gridmap_.setupP(p_min, p_max, p_hit, p_mis, p_occ, p_def);
    }
    gridmap_.inflate_size = inflate_size_;
    // use mask parameter
    nh.getParam("use_mask", use_mask_);

    gridmap_inflate_pub_ = nh.advertise<quadrotor_msgs::OccMap3d>("gridmap_inflate", 1);

    local_pc_pub_ = nh.advertise<sensor_msgs::PointCloud2>("local_pointcloud", 1);
    pcl_pub_ = nh.advertise<sensor_msgs::PointCloud2>("mask_cloud", 10);

    if (use_global_map_) {
      map_pc_sub_ = nh.subscribe<sensor_msgs::PointCloud2>("global_map", 1, &Nodelet::map_call_back, this);
      global_map_timer_ = nh.createTimer(ros::Duration(1.0), &Nodelet::global_map_timer_callback, this);
    } else {
      depth_sub_.subscribe(nh, "depth", 1);
      odom_sub_.subscribe(nh, "odom", 50);
      depth_odom_sync_Ptr_ = std::make_shared<ImageOdomSynchronizer>(ImageOdomSyncPolicy(100), depth_sub_, odom_sub_);
      depth_odom_sync_Ptr_->registerCallback(boost::bind(&Nodelet::depth_odom_callback, this, _1, _2));
    }

    if (use_mask_) {
      target_odom_sub_ = nh.subscribe<nav_msgs::Odometry>("target", 1, &Nodelet::target_odom_callback, this, ros::TransportHints().tcpNoDelay());
    }
  }

 public:
  void onInit(void) {
    ros::NodeHandle nh(getMTPrivateNodeHandle());
    initThread_ = std::thread(std::bind(&Nodelet::init, this, nh));
  }
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace mapping

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(mapping::Nodelet, nodelet::Nodelet);