#pragma once
#include "so3_quadrotor/geometry_utils.hpp"

namespace so3_quadrotor {

// data types
struct Config {
  double          g;      // gravity
  double          mass;
  Eigen::Matrix3d J;      // Inertia
  double          kf;
  double          km;
  double          arm_length;
  double          motor_time_constant; // unit: sec
  double          max_rpm;
  double          min_rpm;
};
struct Control {
  double rpm[4];
};
struct Cmd {
  float force[3] = {0,0,0};
  float qx, qy, qz, qw;
  float kR[3] = {0.0, 0.0, 0.0};
  float kOm[3] = {0.0, 0.0, 0.0};
  float corrections[3] = {0,0,0};
  float current_yaw = 0;
  bool  use_external_yaw = false;
};

class Quadrotor {
 private:
  // parameters
  Config config_;
  // state
  struct State {
    Eigen::Vector3d x = Eigen::Vector3d::Zero();
    Eigen::Vector3d v = Eigen::Vector3d::Zero();
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d omega = Eigen::Vector3d::Zero();
    Eigen::Vector4d motor_rpm = Eigen::Vector4d::Zero();
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    inline State operator+(const State &t) const {
      State sum;
      sum.x = x + t.x;
      sum.v = v + t.v;
      sum.R = R + t.R;
      sum.omega = omega + t.omega;
      sum.motor_rpm = motor_rpm + t.motor_rpm;
      return sum;
    }
    inline State operator*(const double &t) const {
      State mul;
      mul.x = x * t;
      mul.v = v * t;
      mul.R = R * t;
      mul.omega = omega * t;
      mul.motor_rpm = motor_rpm * t;
      return mul;
    }
    inline State operator/(const double &t) const {
      State mul;
      mul.x = x / t;
      mul.v = v / t;
      mul.R = R / t;
      mul.omega = omega / t;
      mul.motor_rpm = motor_rpm / t;
      return mul;
    }
  } state_;
  Eigen::Vector4d  input_ = Eigen::Vector4d::Zero();
 public:
  const Config &config;
  const State &state;
  Quadrotor(const Config &conf) : config_(conf), 
    config(config_), state(state_) {};
  ~Quadrotor() {};

  // Inputs are desired RPM for the motors
  // Rotor numbering is:
  //   *1*    Front
  // 3     4
  //    2
  // with 1 and 2 clockwise and 3 and 4 counter-clockwise (looking from top)
  inline void setInput(double u1, double u2, double u3, double u4) {
    input_(0) = u1;
    input_(1) = u2;
    input_(2) = u3;
    input_(3) = u4;
    for (size_t i=0; i<4; ++i) {
      if (std::isnan(input_(i))) {
        printf("so3_quadrotor: NAN input!\n");
      }
      input_(i) = input_(i) < config_.max_rpm ? input_(i) : config_.max_rpm;
      input_(i) = input_(i) > config_.min_rpm ? input_(i) : config_.min_rpm;
    }
  }

  // calculate dot of state
  inline State diff(const State &state) const {
    State state_dot;
    // Re-orthonormalize R (polar decomposition)
    Eigen::LLT<Eigen::Matrix3d> llt(state.R.transpose() * state.R);
    Eigen::Matrix3d             P = llt.matrixL();
    Eigen::Matrix3d             R = state.R * P.inverse();

    Eigen::Matrix3d omega_vee(Eigen::Matrix3d::Zero());

    omega_vee(2, 1) = state.omega(0);
    omega_vee(1, 2) = -state.omega(0);
    omega_vee(0, 2) = state.omega(1);
    omega_vee(2, 0) = -state.omega(1);
    omega_vee(1, 0) = state.omega(2);
    omega_vee(0, 1) = -state.omega(2);

    Eigen::Vector4d motor_rpm_sq = state.motor_rpm.array().square();

    double thrust = config_.kf * motor_rpm_sq.sum();

    Eigen::Vector3d moments;
    moments(0) = config_.kf * (motor_rpm_sq(2) - motor_rpm_sq(3)) * config_.arm_length;
    moments(1) = config_.kf * (motor_rpm_sq(1) - motor_rpm_sq(0)) * config_.arm_length;
    moments(2) = config_.km * (motor_rpm_sq(0) + motor_rpm_sq(1) - motor_rpm_sq(2) -
                        motor_rpm_sq(3));

    double resistance = 0.1 *                                        // C
                        3.14159265 * (config_.arm_length) * (config_.arm_length) * // S
                        state.v.norm() * state.v.norm();
    Eigen::Vector3d vnorm = state.v.norm() > 0 ? state.v.normalized() : state.v;
    state_dot.x = state.v;
    state_dot.v = -Eigen::Vector3d(0, 0, config_.g) + thrust * R.col(2) / config_.mass - resistance * vnorm / config_.mass;
    state_dot.R = R * omega_vee;
    state_dot.omega = config_.J.inverse() * (moments - state.omega.cross(config_.J * state.omega));
    state_dot.motor_rpm = (input_ - state.motor_rpm) / config_.motor_time_constant;
    return state_dot;
  }
  // Runs the actual dynamics simulation with a time step of dt
  inline void step(const double &dt) {
    // Runge–Kutta
    State k1 = diff(state_);
    State k2 = diff(state_+k1*dt/2);
    State k3 = diff(state_+k2*dt/2);
    State k4 = diff(state_+k3*dt);
    state_ = state_ + (k1+k2*2+k3*2+k4) * dt/6;
  }
  // get control from cmd
  inline Control getControl(const Cmd& cmd) {
    double         kf = config_.kf - cmd.corrections[0];
    double         km = config_.km / kf * kf;
    double          d = config_.arm_length;
    Eigen::Matrix3f J = config_.J.cast<float>();
    float     I[3][3] = { { J(0, 0), J(0, 1), J(0, 2) },
                          { J(1, 0), J(1, 1), J(1, 2) },
                          { J(2, 0), J(2, 1), J(2, 2) } };
    // rotation, may use external yaw
    Eigen::Vector3d ypr = uav_utils::R_to_ypr(state_.R);
    if (cmd.use_external_yaw) {
      ypr[0] = cmd.current_yaw;
    }
    Eigen::Matrix3d R; 
    R = Eigen::AngleAxisd(ypr[0], Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(ypr[1], Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(ypr[2], Eigen::Vector3d::UnitX());
    float R11 = R(0, 0);
    float R12 = R(0, 1);
    float R13 = R(0, 2);
    float R21 = R(1, 0);
    float R22 = R(1, 1);
    float R23 = R(1, 2);
    float R31 = R(2, 0);
    float R32 = R(2, 1);
    float R33 = R(2, 2);
    float Om1 = state_.omega(0);
    float Om2 = state_.omega(1);
    float Om3 = state_.omega(2);
    float Rd11 =
      cmd.qw * cmd.qw + cmd.qx * cmd.qx - cmd.qy * cmd.qy - cmd.qz * cmd.qz;
    float Rd12 = 2 * (cmd.qx * cmd.qy - cmd.qw * cmd.qz);
    float Rd13 = 2 * (cmd.qx * cmd.qz + cmd.qw * cmd.qy);
    float Rd21 = 2 * (cmd.qx * cmd.qy + cmd.qw * cmd.qz);
    float Rd22 =
      cmd.qw * cmd.qw - cmd.qx * cmd.qx + cmd.qy * cmd.qy - cmd.qz * cmd.qz;
    float Rd23 = 2 * (cmd.qy * cmd.qz - cmd.qw * cmd.qx);
    float Rd31 = 2 * (cmd.qx * cmd.qz - cmd.qw * cmd.qy);
    float Rd32 = 2 * (cmd.qy * cmd.qz + cmd.qw * cmd.qx);
    float Rd33 =
      cmd.qw * cmd.qw - cmd.qx * cmd.qx - cmd.qy * cmd.qy + cmd.qz * cmd.qz;

    float Psi = 0.5f * (3.0f - (Rd11 * R11 + Rd21 * R21 + Rd31 * R31 +
                                Rd12 * R12 + Rd22 * R22 + Rd32 * R32 +
                                Rd13 * R13 + Rd23 * R23 + Rd33 * R33));
    float force = 0;
    if (Psi < 1.0f) // Position control stability guaranteed only when Psi < 1
      force = cmd.force[0] * R13 + cmd.force[1] * R23 + cmd.force[2] * R33;

    float eR1 = 0.5f * (R12 * Rd13 - R13 * Rd12 + R22 * Rd23 - R23 * Rd22 +
                        R32 * Rd33 - R33 * Rd32);
    float eR2 = 0.5f * (R13 * Rd11 - R11 * Rd13 - R21 * Rd23 + R23 * Rd21 -
                        R31 * Rd33 + R33 * Rd31);
    float eR3 = 0.5f * (R11 * Rd12 - R12 * Rd11 + R21 * Rd22 - R22 * Rd21 +
                        R31 * Rd32 - R32 * Rd31);

    float eOm1 = Om1;
    float eOm2 = Om2;
    float eOm3 = Om3;

    float in1 = Om2 * (I[2][0] * Om1 + I[2][1] * Om2 + I[2][2] * Om3) -
                Om3 * (I[1][0] * Om1 + I[1][1] * Om2 + I[1][2] * Om3);
    float in2 = Om3 * (I[0][0] * Om1 + I[0][1] * Om2 + I[0][2] * Om3) -
                Om1 * (I[2][0] * Om1 + I[2][1] * Om2 + I[2][2] * Om3);
    float in3 = Om1 * (I[1][0] * Om1 + I[1][1] * Om2 + I[1][2] * Om3) -
                Om2 * (I[0][0] * Om1 + I[0][1] * Om2 + I[0][2] * Om3);
    float M1 = -cmd.kR[0] * eR1 - cmd.kOm[0] * eOm1 + in1;
    float M2 = -cmd.kR[1] * eR2 - cmd.kOm[1] * eOm2 + in2;
    float M3 = -cmd.kR[2] * eR3 - cmd.kOm[2] * eOm3 + in3;

    float w_sq[4];
    w_sq[0] = force / (4 * kf) - M2 / (2 * d * kf) + M3 / (4 * km);
    w_sq[1] = force / (4 * kf) + M2 / (2 * d * kf) + M3 / (4 * km);
    w_sq[2] = force / (4 * kf) + M1 / (2 * d * kf) - M3 / (4 * km);
    w_sq[3] = force / (4 * kf) - M1 / (2 * d * kf) - M3 / (4 * km);

    Control control;
    for (int i = 0; i < 4; i++) {
      if (w_sq[i] < 0) w_sq[i] = 0;
      control.rpm[i] = sqrtf(w_sq[i]);
    }
    return control;
  }

  // set initial state
  inline void setPos(const Eigen::Vector3d &pos) {
    state_.x = pos;
  }
  inline void setYpr(const Eigen::Vector3d &ypr) {
    state_.R = uav_utils::ypr_to_R(ypr);
  }
  inline void setRpm(const Eigen::Vector4d &rpm) {
    state_.motor_rpm = rpm;
  }
  // get values of state
  inline double getGrav() const {
    return config_.g * config_.mass;
  }
  inline Eigen::Vector3d getPos() const {
    return state_.x;
  }
  inline Eigen::Vector3d getVel() const {
    return state_.v;
  }
  inline Eigen::Vector3d getAcc() const {
    Eigen::LLT<Eigen::Matrix3d> llt(state_.R.transpose() * state_.R);
    Eigen::Matrix3d             P = llt.matrixL();
    Eigen::Matrix3d             R = state_.R * P.inverse();
    double resistance = 0.1 *                                        // C
                        3.14159265 * (config_.arm_length) * (config_.arm_length) * // S
                        state_.v.norm() * state_.v.norm();
    Eigen::Vector3d vnorm = state_.v.normalized();
    Eigen::Vector4d motor_rpm_sq = state_.motor_rpm.array().square();
    double thrust = config_.kf * motor_rpm_sq.sum();
    return -Eigen::Vector3d(0, 0, config_.g) + thrust * R.col(2) / config_.mass - resistance * vnorm / config_.mass;
  }
  inline Eigen::Quaterniond getQuat() const {
    return Eigen::Quaterniond(state_.R);
  }
  inline Eigen::Vector3d getOmega() const {
    return state_.omega;
  }

};

} // namespace so3_quadrotor
