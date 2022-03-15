#include <traj_opt/traj_opt.h>

#include <random>
#include <traj_opt/geoutils.hpp>
#include <traj_opt/lbfgs_raw.hpp>

namespace traj_opt {

static bool landing_ = false;

// SECTION  variables transformation and gradient transmission
static double expC2(double t) {
  return t > 0.0 ? ((0.5 * t + 1.0) * t + 1.0)
                 : 1.0 / ((0.5 * t - 1.0) * t + 1.0);
}
static double logC2(double T) {
  return T > 1.0 ? (sqrt(2.0 * T - 1.0) - 1.0) : (1.0 - sqrt(2.0 / T - 1.0));
}
static void forwardT(const Eigen::Ref<const Eigen::VectorXd>& t, const double& sT, Eigen::Ref<Eigen::VectorXd> vecT) {
  int M = t.size();
  for (int i = 0; i < M; ++i) {
    vecT(i) = expC2(t(i));
  }
  vecT(M) = 0.0;
  vecT /= 1.0 + vecT.sum();
  vecT(M) = 1.0 - vecT.sum();
  vecT *= sT;
  return;
}
static void backwardT(const Eigen::Ref<const Eigen::VectorXd>& vecT, Eigen::Ref<Eigen::VectorXd> t) {
  int M = t.size();
  t = vecT.head(M) / vecT(M);
  for (int i = 0; i < M; ++i) {
    t(i) = logC2(vecT(i));
  }
  return;
}
static void addLayerTGrad(const Eigen::Ref<const Eigen::VectorXd>& t,
                          const double& sT,
                          const Eigen::Ref<const Eigen::VectorXd>& gradT,
                          Eigen::Ref<Eigen::VectorXd> gradt) {
  int Ms1 = t.size();
  Eigen::VectorXd gFree = sT * gradT.head(Ms1);
  double gTail = sT * gradT(Ms1);
  Eigen::VectorXd dExpTau(Ms1);
  double expTauSum = 0.0, gFreeDotExpTau = 0.0;
  double denSqrt, expTau;
  for (int i = 0; i < Ms1; i++) {
    if (t(i) > 0) {
      expTau = (0.5 * t(i) + 1.0) * t(i) + 1.0;
      dExpTau(i) = t(i) + 1.0;
      expTauSum += expTau;
      gFreeDotExpTau += expTau * gFree(i);
    } else {
      denSqrt = (0.5 * t(i) - 1.0) * t(i) + 1.0;
      expTau = 1.0 / denSqrt;
      dExpTau(i) = (1.0 - t(i)) / (denSqrt * denSqrt);
      expTauSum += expTau;
      gFreeDotExpTau += expTau * gFree(i);
    }
  }
  denSqrt = expTauSum + 1.0;
  gradt = (gFree.array() - gTail) * dExpTau.array() / denSqrt -
          (gFreeDotExpTau - gTail * expTauSum) * dExpTau.array() / (denSqrt * denSqrt);
}

static void forwardP(const Eigen::Ref<const Eigen::VectorXd>& p,
                     const std::vector<Eigen::MatrixXd>& cfgPolyVs,
                     Eigen::MatrixXd& inP) {
  int M = cfgPolyVs.size();
  Eigen::VectorXd q;
  int j = 0, k;
  for (int i = 0; i < M; ++i) {
    k = cfgPolyVs[i].cols() - 1;
    q = 2.0 / (1.0 + p.segment(j, k).squaredNorm()) * p.segment(j, k);
    inP.col(i) = cfgPolyVs[i].rightCols(k) * q.cwiseProduct(q) +
                 cfgPolyVs[i].col(0);
    j += k;
  }
  return;
}
static double objectiveNLS(void* ptrPOBs,
                           const double* x,
                           double* grad,
                           const int n) {
  const Eigen::MatrixXd& pobs = *(Eigen::MatrixXd*)ptrPOBs;
  Eigen::Map<const Eigen::VectorXd> p(x, n);
  Eigen::Map<Eigen::VectorXd> gradp(grad, n);

  double qnsqr = p.squaredNorm();
  double qnsqrp1 = qnsqr + 1.0;
  double qnsqrp1sqr = qnsqrp1 * qnsqrp1;
  Eigen::VectorXd r = 2.0 / qnsqrp1 * p;

  Eigen::Vector3d delta = pobs.rightCols(n) * r.cwiseProduct(r) +
                          pobs.col(1) - pobs.col(0);
  double cost = delta.squaredNorm();
  Eigen::Vector3d gradR3 = 2 * delta;

  Eigen::VectorXd gdr = pobs.rightCols(n).transpose() * gradR3;
  gdr = gdr.array() * r.array() * 2.0;
  gradp = gdr * 2.0 / qnsqrp1 -
          p * 4.0 * gdr.dot(p) / qnsqrp1sqr;

  return cost;
}

static void backwardP(const Eigen::Ref<const Eigen::MatrixXd>& inP,
                      const std::vector<Eigen::MatrixXd>& cfgPolyVs,
                      Eigen::VectorXd& p) {
  int M = inP.cols();
  int j = 0, k;

  // Parameters for tiny nonlinear least squares
  double minSqrD;
  lbfgs::lbfgs_parameter_t nls_params;
  lbfgs::lbfgs_load_default_parameters(&nls_params);
  nls_params.g_epsilon = FLT_EPSILON;
  nls_params.max_iterations = 128;

  Eigen::MatrixXd pobs;
  for (int i = 0; i < M; i++) {
    k = cfgPolyVs[i].cols() - 1;
    p.segment(j, k).setConstant(1.0 / (sqrt(k + 1.0) + 1.0));
    pobs.resize(3, k + 2);
    pobs << inP.col(i), cfgPolyVs[i];
    lbfgs::lbfgs_optimize(k,
                          p.data() + j,
                          &minSqrD,
                          &objectiveNLS,
                          nullptr,
                          nullptr,
                          &pobs,
                          &nls_params);
    j += k;
  }
  return;
}
static void addLayerPGrad(const Eigen::Ref<const Eigen::VectorXd>& p,
                          const std::vector<Eigen::MatrixXd>& cfgPolyVs,
                          const Eigen::Ref<const Eigen::MatrixXd>& gradInPs,
                          Eigen::Ref<Eigen::VectorXd> grad) {
  int M = gradInPs.cols();

  int j = 0, k;
  double qnsqr, qnsqrp1, qnsqrp1sqr;
  Eigen::VectorXd q, r, gdr;
  for (int i = 0; i < M; i++) {
    k = cfgPolyVs[i].cols() - 1;
    q = p.segment(j, k);
    qnsqr = q.squaredNorm();
    qnsqrp1 = qnsqr + 1.0;
    qnsqrp1sqr = qnsqrp1 * qnsqrp1;
    r = 2.0 / qnsqrp1 * q;
    gdr = cfgPolyVs[i].rightCols(k).transpose() * gradInPs.col(i);
    gdr = gdr.array() * r.array() * 2.0;

    grad.segment(j, k) = gdr * 2.0 / qnsqrp1 -
                         q * 4.0 * gdr.dot(q) / qnsqrp1sqr;
    j += k;
  }
  return;
}
// !SECTION variables transformation and gradient transmission

// SECTION object function
static inline double objectiveFunc(void* ptrObj,
                                   const double* x,
                                   double* grad,
                                   const int n) {
  TrajOpt& obj = *(TrajOpt*)ptrObj;

  Eigen::Map<const Eigen::VectorXd> t(x, obj.dim_t_);
  Eigen::Map<const Eigen::VectorXd> p(x + obj.dim_t_, obj.dim_p_);
  Eigen::Map<Eigen::VectorXd> gradt(grad, obj.dim_t_);
  Eigen::Map<Eigen::VectorXd> gradp(grad + obj.dim_t_, obj.dim_p_);
  double deltaT = x[obj.dim_t_ + obj.dim_p_];

  Eigen::VectorXd T(obj.N_);
  Eigen::MatrixXd P(3, obj.N_ - 1);
  // T_sigma = T_s + deltaT^2
  double sumT = obj.sum_T_ + deltaT * deltaT;
  forwardT(t, sumT, T);
  forwardP(p, obj.cfgVs_, P);

  obj.jerkOpt_.generate(P, T);
  double cost = obj.jerkOpt_.getTrajJerkCost();
  obj.jerkOpt_.calGrads_CT();
  obj.addTimeIntPenalty(cost);
  obj.addTimeCost(cost);
  obj.jerkOpt_.calGrads_PT();
  grad[obj.dim_t_ + obj.dim_p_] = obj.jerkOpt_.gdT.dot(T) / sumT + obj.rhoT_;
  cost += obj.rhoT_ * deltaT * deltaT;
  grad[obj.dim_t_ + obj.dim_p_] *= 2 * deltaT;
  addLayerTGrad(t, sumT, obj.jerkOpt_.gdT, gradt);
  addLayerPGrad(p, obj.cfgVs_, obj.jerkOpt_.gdP, gradp);

  return cost;
}
// !SECTION object function

static inline int earlyExit(void* ptrObj,
                            const double* x,
                            const double* grad,
                            const double fx,
                            const double xnorm,
                            const double gnorm,
                            const double step,
                            int n,
                            int k,
                            int ls) {
  return k > 1e3;
}

bool TrajOpt::extractVs(const std::vector<Eigen::MatrixXd>& hPs,
                        std::vector<Eigen::MatrixXd>& vPs) const {
  const int M = hPs.size() - 1;

  vPs.clear();
  vPs.reserve(2 * M + 1);

  int nv;
  Eigen::MatrixXd curIH, curIV, curIOB;
  for (int i = 0; i < M; i++) {
    if (!geoutils::enumerateVs(hPs[i], curIV)) {
      return false;
    }
    nv = curIV.cols();
    curIOB.resize(3, nv);
    curIOB << curIV.col(0), curIV.rightCols(nv - 1).colwise() - curIV.col(0);
    vPs.push_back(curIOB);

    curIH.resize(6, hPs[i].cols() + hPs[i + 1].cols());
    curIH << hPs[i], hPs[i + 1];
    if (!geoutils::enumerateVs(curIH, curIV)) {
      return false;
    }
    nv = curIV.cols();
    curIOB.resize(3, nv);
    curIOB << curIV.col(0), curIV.rightCols(nv - 1).colwise() - curIV.col(0);
    vPs.push_back(curIOB);
  }

  if (!geoutils::enumerateVs(hPs.back(), curIV)) {
    return false;
  }
  nv = curIV.cols();
  curIOB.resize(3, nv);
  curIOB << curIV.col(0), curIV.rightCols(nv - 1).colwise() - curIV.col(0);
  vPs.push_back(curIOB);

  return true;
}

TrajOpt::TrajOpt(ros::NodeHandle& nh) : nh_(nh) {
  // nh.getParam("N", N_);
  nh.getParam("K", K_);
  // load dynamic paramters
  nh.getParam("vmax", vmax_);
  nh.getParam("amax", amax_);
  nh.getParam("rhoT", rhoT_);
  nh.getParam("rhoP", rhoP_);
  nh.getParam("rhoTracking", rhoTracking_);
  nh.getParam("rhosVisibility", rhosVisibility_);
  nh.getParam("theta_clearance", theta_clearance_);
  nh.getParam("rhoV", rhoV_);
  nh.getParam("rhoA", rhoA_);
  nh.getParam("tracking_dur", tracking_dur_);
  nh.getParam("tracking_dist", tracking_dist_);
  nh.getParam("tracking_dt", tracking_dt_);
  nh.getParam("clearance_d", clearance_d_);
  nh.getParam("tolerance_d", tolerance_d_);
}

void TrajOpt::setBoundConds(const Eigen::MatrixXd& iniState,
                            const Eigen::MatrixXd& finState) {
  Eigen::MatrixXd initS = iniState;
  Eigen::MatrixXd finalS = finState;
  double tempNorm = initS.col(1).norm();
  initS.col(1) *= tempNorm > vmax_ ? (vmax_ / tempNorm) : 1.0;
  tempNorm = finalS.col(1).norm();
  finalS.col(1) *= tempNorm > vmax_ ? (vmax_ / tempNorm) : 1.0;
  tempNorm = initS.col(2).norm();
  initS.col(2) *= tempNorm > amax_ ? (amax_ / tempNorm) : 1.0;
  tempNorm = finalS.col(2).norm();
  finalS.col(2) *= tempNorm > amax_ ? (amax_ / tempNorm) : 1.0;

  Eigen::VectorXd T(N_);
  T.setConstant(sum_T_ / N_);
  backwardT(T, t_);
  Eigen::MatrixXd P(3, N_ - 1);
  for (int i = 0; i < N_ - 1; ++i) {
    int k = cfgVs_[i].cols() - 1;
    P.col(i) = cfgVs_[i].rightCols(k).rowwise().sum() / (1.0 + k) + cfgVs_[i].col(0);
  }
  backwardP(P, cfgVs_, p_);
  jerkOpt_.reset(initS, finalS, N_);
  return;
}

int TrajOpt::optimize(const double& delta) {
  // Setup for L-BFGS solver
  lbfgs::lbfgs_parameter_t lbfgs_params;
  lbfgs::lbfgs_load_default_parameters(&lbfgs_params);
  lbfgs_params.mem_size = 16;
  lbfgs_params.past = 3;
  lbfgs_params.g_epsilon = 1e-10;
  lbfgs_params.min_step = 1e-32;
  lbfgs_params.delta = delta;
  Eigen::Map<Eigen::VectorXd> t(x_, dim_t_);
  Eigen::Map<Eigen::VectorXd> p(x_ + dim_t_, dim_p_);
  t = t_;
  p = p_;
  double minObjective;
  auto ret = lbfgs::lbfgs_optimize(dim_t_ + dim_p_ + 1, x_, &minObjective,
                                   &objectiveFunc, nullptr,
                                   &earlyExit, this, &lbfgs_params);
  std::cout << "\033[32m"
            << "ret: " << ret << "\033[0m" << std::endl;
  t_ = t;
  p_ = p;
  return ret;
}

bool TrajOpt::generate_traj(const Eigen::MatrixXd& iniState,
                            const Eigen::MatrixXd& finState,
                            const std::vector<Eigen::Vector3d>& target_predcit,
                            const std::vector<Eigen::Vector3d>& visible_ps,
                            const std::vector<double>& thetas,
                            const std::vector<Eigen::MatrixXd>& hPolys,
                            Trajectory& traj) {
  landing_ = false;
  cfgHs_ = hPolys;
  if (cfgHs_.size() == 1) {
    cfgHs_.push_back(cfgHs_[0]);
  }
  if (!extractVs(cfgHs_, cfgVs_)) {
    ROS_ERROR("extractVs fail!");
    return false;
  }
  N_ = 2 * cfgHs_.size();
  // NOTE wonderful trick
  sum_T_ = tracking_dur_;

  // NOTE: one corridor two pieces
  dim_t_ = N_ - 1;
  dim_p_ = 0;
  for (const auto& cfgV : cfgVs_) {
    dim_p_ += cfgV.cols() - 1;
  }
  // std::cout << "dim_p_: " << dim_p_ << std::endl;
  p_.resize(dim_p_);
  t_.resize(dim_t_);
  x_ = new double[dim_p_ + dim_t_ + 1];
  Eigen::VectorXd T(N_);
  Eigen::MatrixXd P(3, N_ - 1);

  tracking_ps_ = target_predcit;
  tracking_visible_ps_ = visible_ps;
  tracking_thetas_ = thetas;

  setBoundConds(iniState, finState);
  x_[dim_p_ + dim_t_] = 0.1;
  int opt_ret = optimize();
  if (opt_ret < 0) {
    return false;
  }
  double sumT = sum_T_ + x_[dim_p_ + dim_t_] * x_[dim_p_ + dim_t_];
  forwardT(t_, sumT, T);
  forwardP(p_, cfgVs_, P);
  jerkOpt_.generate(P, T);
  // std::cout << "P: \n" << P << std::endl;
  // std::cout << "T: " << T.transpose() << std::endl;
  traj = jerkOpt_.getTraj();
  delete[] x_;
  return true;
}

// NOTE just for landing the car of YTK
bool TrajOpt::generate_traj(const Eigen::MatrixXd& iniState,
                            const Eigen::MatrixXd& finState,
                            const std::vector<Eigen::Vector3d>& target_predcit,
                            const std::vector<Eigen::MatrixXd>& hPolys,
                            Trajectory& traj) {
  landing_ = true;
  cfgHs_ = hPolys;
  if (cfgHs_.size() == 1) {
    cfgHs_.push_back(cfgHs_[0]);
  }
  if (!extractVs(cfgHs_, cfgVs_)) {
    ROS_ERROR("extractVs fail!");
    return false;
  }
  N_ = 2 * cfgHs_.size();
  // NOTE wonderful trick
  sum_T_ = tracking_dur_;

  // NOTE: one corridor two pieces
  dim_t_ = N_ - 1;
  dim_p_ = 0;
  for (const auto& cfgV : cfgVs_) {
    dim_p_ += cfgV.cols() - 1;
  }
  // std::cout << "dim_p_: " << dim_p_ << std::endl;
  p_.resize(dim_p_);
  t_.resize(dim_t_);
  x_ = new double[dim_p_ + dim_t_ + 1];
  Eigen::VectorXd T(N_);
  Eigen::MatrixXd P(3, N_ - 1);

  tracking_ps_ = target_predcit;

  setBoundConds(iniState, finState);
  x_[dim_p_ + dim_t_] = 0.1;
  int opt_ret = optimize();
  if (opt_ret < 0) {
    return false;
  }
  double sumT = sum_T_ + x_[dim_p_ + dim_t_] * x_[dim_p_ + dim_t_];
  forwardT(t_, sumT, T);
  forwardP(p_, cfgVs_, P);
  jerkOpt_.generate(P, T);
  // std::cout << "P: \n" << P << std::endl;
  // std::cout << "T: " << T.transpose() << std::endl;
  traj = jerkOpt_.getTraj();
  delete[] x_;
  return true;
}

void TrajOpt::addTimeIntPenalty(double& cost) {
  Eigen::Vector3d pos, vel, acc, jer;
  Eigen::Vector3d grad_tmp;
  double cost_tmp;
  Eigen::Matrix<double, 6, 1> beta0, beta1, beta2, beta3;
  double s1, s2, s3, s4, s5;
  double step, alpha;
  Eigen::Matrix<double, 6, 3> gradViolaPc, gradViolaVc, gradViolaAc;
  double gradViolaPt, gradViolaVt, gradViolaAt;
  double omg;

  int innerLoop;
  for (int i = 0; i < N_; ++i) {
    const auto& c = jerkOpt_.b.block<6, 3>(i * 6, 0);
    step = jerkOpt_.T1(i) / K_;
    s1 = 0.0;
    innerLoop = K_ + 1;

    const auto& hPoly = cfgHs_[i / 2];
    for (int j = 0; j < innerLoop; ++j) {
      s2 = s1 * s1;
      s3 = s2 * s1;
      s4 = s2 * s2;
      s5 = s4 * s1;
      beta0 << 1.0, s1, s2, s3, s4, s5;
      beta1 << 0.0, 1.0, 2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s4;
      beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3;
      beta3 << 0.0, 0.0, 0.0, 6.0, 24.0 * s1, 60.0 * s2;
      alpha = 1.0 / K_ * j;
      pos = c.transpose() * beta0;
      vel = c.transpose() * beta1;
      acc = c.transpose() * beta2;
      jer = c.transpose() * beta3;

      omg = (j == 0 || j == innerLoop - 1) ? 0.5 : 1.0;

      if (grad_cost_p_corridor(pos, hPoly, grad_tmp, cost_tmp)) {
        gradViolaPc = beta0 * grad_tmp.transpose();
        gradViolaPt = alpha * grad_tmp.transpose() * vel;
        jerkOpt_.gdC.block<6, 3>(i * 6, 0) += omg * step * gradViolaPc;
        jerkOpt_.gdT(i) += omg * (cost_tmp / K_ + step * gradViolaPt);
        cost += omg * step * cost_tmp;
      }
      if (grad_cost_v(vel, grad_tmp, cost_tmp)) {
        gradViolaVc = beta1 * grad_tmp.transpose();
        gradViolaVt = alpha * grad_tmp.dot(acc);
        jerkOpt_.gdC.block<6, 3>(i * 6, 0) += omg * step * gradViolaVc;
        jerkOpt_.gdT(i) += omg * (cost_tmp / K_ + step * gradViolaVt);
        cost += omg * step * cost_tmp;
      }
      if (grad_cost_a(acc, grad_tmp, cost_tmp)) {
        gradViolaAc = beta2 * grad_tmp.transpose();
        gradViolaAt = alpha * grad_tmp.dot(jer);
        jerkOpt_.gdC.block<6, 3>(i * 6, 0) += omg * step * gradViolaAc;
        jerkOpt_.gdT(i) += omg * (cost_tmp / K_ + step * gradViolaAt);
        cost += omg * step * cost_tmp;
      }

      s1 += step;
    }
  }
}

void TrajOpt::addTimeCost(double& cost) {
  const auto& T = jerkOpt_.T1;
  int piece = 0;
  int M = tracking_ps_.size() * 4 / 5;
  double t = 0;
  double t_pre = 0;

  double step = tracking_dt_;
  Eigen::Matrix<double, 6, 1> beta0, beta1;
  double s1, s2, s3, s4, s5;
  Eigen::Vector3d pos, vel;
  Eigen::Vector3d grad_tmp;
  double cost_tmp;
  Eigen::Matrix<double, 6, 3> gradViolaPc;

  for (int i = 0; i < M; ++i) {
    double rho = exp2(-3.0 * i / M);
    while (t - t_pre > T(piece)) {
      t_pre += T(piece);
      piece++;
    }
    s1 = t - t_pre;
    s2 = s1 * s1;
    s3 = s2 * s1;
    s4 = s2 * s2;
    s5 = s4 * s1;
    beta0 << 1.0, s1, s2, s3, s4, s5;
    beta1 << 0.0, 1.0, 2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s4;
    const auto& c = jerkOpt_.b.block<6, 3>(piece * 6, 0);
    pos = c.transpose() * beta0;
    vel = c.transpose() * beta1;
    Eigen::Vector3d target_p = tracking_ps_[i];

    if (landing_) {
      if (grad_cost_p_landing(pos, target_p, grad_tmp, cost_tmp)) {
        gradViolaPc = beta0 * grad_tmp.transpose();
        cost += rho * step * cost_tmp;
        jerkOpt_.gdC.block<6, 3>(piece * 6, 0) += rho * step * gradViolaPc;
        if (piece > 0) {
          jerkOpt_.gdT.head(piece).array() += -rho * step * grad_tmp.dot(vel);
        }
      }
    } else {
      if (grad_cost_p_tracking(pos, target_p, grad_tmp, cost_tmp)) {
        gradViolaPc = beta0 * grad_tmp.transpose();
        cost += rho * step * cost_tmp;
        jerkOpt_.gdC.block<6, 3>(piece * 6, 0) += rho * step * gradViolaPc;
        if (piece > 0) {
          jerkOpt_.gdT.head(piece).array() += -rho * step * grad_tmp.dot(vel);
        }
      }
      // TODO occlusion
      if (grad_cost_visibility(pos, target_p, tracking_visible_ps_[i], tracking_thetas_[i],
                               grad_tmp, cost_tmp)) {
        gradViolaPc = beta0 * grad_tmp.transpose();
        cost += rho * step * cost_tmp;
        jerkOpt_.gdC.block<6, 3>(piece * 6, 0) += rho * step * gradViolaPc;
        if (piece > 0) {
          jerkOpt_.gdT.head(piece).array() += -rho * step * grad_tmp.dot(vel);
        }
      }
    }

    t += step;
  }
}

bool TrajOpt::grad_cost_p_corridor(const Eigen::Vector3d& p,
                                   const Eigen::MatrixXd& hPoly,
                                   Eigen::Vector3d& gradp,
                                   double& costp) {
  // return false;
  bool ret = false;
  gradp.setZero();
  costp = 0;
  for (int i = 0; i < hPoly.cols(); ++i) {
    Eigen::Vector3d norm_vec = hPoly.col(i).head<3>();
    double pen = norm_vec.dot(p - hPoly.col(i).tail<3>() + clearance_d_ * norm_vec);
    if (pen > 0) {
      double pen2 = pen * pen;
      gradp += rhoP_ * 3 * pen2 * norm_vec;
      costp += rhoP_ * pen2 * pen;
      ret = true;
    }
  }
  return ret;
}

// y1 = -x^4 + 2*x0 * x^3
// y2 = 2x0^3 x - x0^4
// static double penF(const double& x, double& grad) {
//   static double x0 = 0.1;
//   static double x02 = x0 * x0;
//   static double x03 = x0 * x02;
//   static double x04 = x02 * x02;
//   if (x < x0) {
//     double x2 = x * x;
//     double x3 = x * x2;
//     grad = x2 * (6 * x0 - 4 * x);
//     return x3 * (2 * x0 - x);
//   } else {
//     grad = 2 * x03;
//     return 2 * x03 * x - x04;
//   }
// }
static double penF(const double& x, double& grad) {
  static double eps = 0.05;
  static double eps2 = eps * eps;
  static double eps3 = eps * eps2;
  if (x < 2 * eps) {
    double x2 = x * x;
    double x3 = x * x2;
    double x4 = x2 * x2;
    grad = 12 / eps2 * x2 - 4 / eps3 * x3;
    return 4 / eps2 * x3 - x4 / eps3;
  } else {
    grad = 16;
    return 16 * (x - eps);
  }
}

static double penF2(const double& x, double& grad) {
  double x2 = x * x;
  grad = 3 * x2;
  return x * x2;
}

bool TrajOpt::grad_cost_p_tracking(const Eigen::Vector3d& p,
                                   const Eigen::Vector3d& target_p,
                                   Eigen::Vector3d& gradp,
                                   double& costp) {
  // return false;
  double upper = tracking_dist_ + tolerance_d_;
  double lower = tracking_dist_ - tolerance_d_;
  upper = upper * upper;
  lower = lower * lower;

  Eigen::Vector3d dp = (p - target_p);
  double dr2 = dp.head(2).squaredNorm();
  double dz2 = dp.z() * dp.z();

  bool ret;
  gradp.setZero();
  costp = 0;

  double pen = dr2 - upper;
  if (pen > 0) {
    double grad;
    costp += penF(pen, grad);
    gradp.head(2) += 2 * grad * dp.head(2);
    ret = true;
  } else {
    pen = lower - dr2;
    if (pen > 0) {
      double pen2 = pen * pen;
      gradp.head(2) -= 6 * pen2 * dp.head(2);
      costp += pen2 * pen;
      ret = true;
    }
  }
  pen = dz2 - tolerance_d_ * tolerance_d_;
  if (pen > 0) {
    double pen2 = pen * pen;
    gradp.z() += 6 * pen2 * dp.z();
    costp += pen * pen2;
    ret = true;
  }

  gradp *= rhoTracking_;
  costp *= rhoTracking_;

  return ret;
}

bool TrajOpt::grad_cost_p_landing(const Eigen::Vector3d& p,
                                  const Eigen::Vector3d& target_p,
                                  Eigen::Vector3d& gradp,
                                  double& costp) {
  Eigen::Vector3d dp = (p - target_p);
  double dr2 = dp.head(2).squaredNorm();
  double dz2 = dp.z() * dp.z();

  bool ret;
  gradp.setZero();
  costp = 0;

  double pen = dr2 - tolerance_d_ * tolerance_d_;
  if (pen > 0) {
    double pen2 = pen * pen;
    gradp.head(2) += 6 * pen2 * dp.head(2);
    costp += pen * pen2;
    ret = true;
  }
  pen = dz2 - tolerance_d_ * tolerance_d_;
  if (pen > 0) {
    double pen2 = pen * pen;
    gradp.z() += 6 * pen2 * dp.z();
    costp += pen * pen2;
    ret = true;
  }

  gradp *= rhoTracking_;
  costp *= rhoTracking_;

  return ret;
}

bool TrajOpt::grad_cost_visibility(const Eigen::Vector3d& p,
                                   const Eigen::Vector3d& center,
                                   const Eigen::Vector3d& vis_p,
                                   const double& theta,
                                   Eigen::Vector3d& gradp,
                                   double& costp) {
  Eigen::Vector3d a = p - center;
  Eigen::Vector3d b = vis_p - center;
  double inner_product = a.dot(b);
  double norm_a = a.norm();
  double norm_b = b.norm();
  double theta_less = theta - theta_clearance_ > 0 ? theta - theta_clearance_ : 0;
  double cosTheta = cos(theta_less);
  double pen = cosTheta - inner_product / norm_a / norm_b;
  if (pen > 0) {
    double grad = 0;
    costp = penF2(pen, grad);
    gradp = grad * -(norm_a * b - inner_product / norm_a * a) / norm_a / norm_a / norm_b;
    // gradp = grad * (norm_b * cosTheta / norm_a * a - b);
    gradp *= rhosVisibility_;
    costp *= rhosVisibility_;
    return true;
  } else {
    return false;
  }
}

bool TrajOpt::grad_cost_v(const Eigen::Vector3d& v,
                          Eigen::Vector3d& gradv,
                          double& costv) {
  double vpen = v.squaredNorm() - vmax_ * vmax_;
  if (vpen > 0) {
    gradv = rhoV_ * 6 * vpen * vpen * v;
    costv = rhoV_ * vpen * vpen * vpen;
    return true;
  }
  return false;
}

bool TrajOpt::grad_cost_a(const Eigen::Vector3d& a,
                          Eigen::Vector3d& grada,
                          double& costa) {
  double apen = a.squaredNorm() - amax_ * amax_;
  if (apen > 0) {
    grada = rhoA_ * 6 * apen * apen * a;
    costa = rhoA_ * apen * apen * apen;
    return true;
  }
  return false;
}

}  // namespace traj_opt
