// Copyright (c) 2021 Samsung Research America
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <math.h>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <limits>

#include "gtest/gtest.h"
#include "rclcpp/rclcpp.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"
#include "nav2_ros_common/lifecycle_node.hpp"
#include "path_utils/path_utils.hpp"
#include "nav2_regulated_pure_pursuit_controller/regulated_pure_pursuit_controller.hpp"
#include "nav2_costmap_2d/costmap_filters/filter_values.hpp"
#include "nav2_core/controller_exceptions.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav2_controller/plugins/feasible_path_handler.hpp"

class TestGoalChecker : public nav2_core::GoalChecker
{
public:
  void initialize(
    const nav2::LifecycleNode::WeakPtr &,
    const std::string &,
    const std::shared_ptr<nav2_costmap_2d::Costmap2DROS>) override
  {}

  void reset() override {}

  bool isGoalReached(
    const geometry_msgs::msg::Pose &,
    const geometry_msgs::msg::Pose &,
    const geometry_msgs::msg::Twist &,
    const nav_msgs::msg::Path &) override
  {
    return false;
  }

  bool getTolerances(
    geometry_msgs::msg::Pose & pose_tolerance,
    geometry_msgs::msg::Twist & vel_tolerance) override
  {
    pose_tolerance.position.x = 0.25;
    vel_tolerance = geometry_msgs::msg::Twist();
    return true;
  }
};

class BasicAPIRPP : public nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController
{
public:
  BasicAPIRPP()
  : nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController() {}

  double getSpeed() {return params_->max_linear_vel;}

  std::unique_ptr<geometry_msgs::msg::PointStamped> createCarrotMsgWrapper(
    const geometry_msgs::msg::PoseStamped & carrot_pose)
  {
    return createCarrotMsg(carrot_pose);
  }

  void setVelocityScaledLookAhead() {params_->use_velocity_scaled_lookahead_dist = true;}
  void setCostRegulationScaling() {params_->use_cost_regulated_linear_velocity_scaling = true;}
  void resetVelocityRegulationScaling() {params_->use_regulated_linear_velocity_scaling = false;}

  double getLookAheadDistanceWrapper(const geometry_msgs::msg::Twist & twist)
  {
    return getLookAheadDistance(twist);
  }

  bool shouldRotateToPathWrapper(
    const geometry_msgs::msg::PoseStamped & carrot_pose, double & angle_to_path)
  {
    double x_vel_sign = 1.0;
    return shouldRotateToPath(carrot_pose, angle_to_path, x_vel_sign);
  }

  bool shouldRotateToGoalHeadingWrapper(const geometry_msgs::msg::PoseStamped & carrot_pose)
  {
    return shouldRotateToGoalHeading(carrot_pose);
  }

  double getVelocitySignWrapper(
    const nav_msgs::msg::Path & path,
    double lookahead_dist = 0.0,
    double carrot_x = 0.0)
  {
    geometry_msgs::msg::PoseStamped carrot_pose;
    carrot_pose.pose.position.x = carrot_x;
    return getVelocitySign(path, carrot_pose, lookahead_dist);
  }

  void setAllowReversing(bool allow_reversing)
  {
    params_->allow_reversing = allow_reversing;
  }

  void setUsePathSegmentDirectionForReversing(bool use_path_segment_direction)
  {
    params_->use_path_segment_direction_for_reversing = use_path_segment_direction;
  }

  void setMinimumTurningRadius(double radius)
  {
    params_->minimum_turning_radius = radius;
  }

  void setMaxAngularVel(double v)
  {
    params_->max_angular_vel = v;
  }

  void setMinAngularVel(double v)
  {
    params_->min_angular_vel = v;
  }

  void setUseDynamicWindow(bool v)
  {
    params_->use_dynamic_window = v;
  }

  void setUseCollisionDetection(bool v)
  {
    params_->use_collision_detection = v;
  }

  nav2_regulated_pure_pursuit_controller::Parameters * paramsHandle()
  {
    return params_;
  }

  void rotateToHeadingWrapper(
    double & linear_vel, double & angular_vel,
    const double & angle_to_path, const geometry_msgs::msg::Twist & curr_speed)
  {
    return rotateToHeading(linear_vel, angular_vel, angle_to_path, curr_speed);
  }

  void applyConstraintsWrapper(
    const double & curvature, const geometry_msgs::msg::Twist & curr_speed,
    const double & pose_cost, const nav_msgs::msg::Path & path, double & linear_vel, double & sign)
  {
    return applyConstraints(
      curvature, curr_speed, pose_cost, path,
      linear_vel, sign);
  }

  bool isCollisionImminentWrapper(
    const geometry_msgs::msg::PoseStamped & robot_pose,
    const double & linear_vel, const double & angular_vel,
    const double & carrot_dist, const double & dist_to_path_end)
  {
    return collision_checker_->isCollisionImminent(
      robot_pose, linear_vel, angular_vel, carrot_dist, dist_to_path_end);
  }
};

TEST(RegulatedPurePursuitTest, basicAPI)
{
  auto node = std::make_shared<nav2::LifecycleNode>("testRPP");
  std::string name = "PathFollower";
  auto tf = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  auto costmap = std::make_shared<nav2_costmap_2d::Costmap2DROS>("fake_costmap");

  // instantiate
  auto ctrl = std::make_shared<BasicAPIRPP>();
  costmap->on_configure(rclcpp_lifecycle::State());
  ctrl->configure(node, name, tf, costmap);
  ctrl->activate();
  ctrl->deactivate();
  ctrl->cleanup();

  // set speed limit
  const double base_speed = ctrl->getSpeed();
  EXPECT_EQ(ctrl->getSpeed(), base_speed);
  ctrl->setSpeedLimit(0.51, false);
  EXPECT_EQ(ctrl->getSpeed(), 0.51);
  ctrl->setSpeedLimit(nav2_costmap_2d::NO_SPEED_LIMIT, false);
  EXPECT_EQ(ctrl->getSpeed(), base_speed);
  ctrl->setSpeedLimit(30, true);
  EXPECT_EQ(ctrl->getSpeed(), base_speed * 0.3);
  ctrl->setSpeedLimit(nav2_costmap_2d::NO_SPEED_LIMIT, true);
  EXPECT_EQ(ctrl->getSpeed(), base_speed);
}

TEST(RegulatedPurePursuitTest, createCarrotMsg)
{
  auto ctrl = std::make_shared<BasicAPIRPP>();
  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = "Hi!";
  pose.pose.position.x = 1.0;
  pose.pose.position.y = 12.0;
  pose.pose.orientation.w = 0.5;

  auto rtn = ctrl->createCarrotMsgWrapper(pose);
  EXPECT_EQ(rtn->header.frame_id, std::string("Hi!"));
  EXPECT_EQ(rtn->point.x, 1.0);
  EXPECT_EQ(rtn->point.y, 12.0);
  EXPECT_EQ(rtn->point.z, 0.01);
}

TEST(RegulatedPurePursuitTest, lookaheadAPI)
{
  auto ctrl = std::make_shared<BasicAPIRPP>();
  auto node = std::make_shared<nav2::LifecycleNode>("testRPP");
  std::string name = "PathFollower";
  auto tf = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  auto costmap = std::make_shared<nav2_costmap_2d::Costmap2DROS>("fake_costmap");
  rclcpp_lifecycle::State state;
  costmap->on_configure(state);
  ctrl->configure(node, name, tf, costmap);

  geometry_msgs::msg::Twist twist;

  // test getLookAheadDistance
  double rtn = ctrl->getLookAheadDistanceWrapper(twist);
  EXPECT_EQ(rtn, 0.6);  // default lookahead_dist

  // shouldn't be a function of speed
  twist.linear.x = 10.0;
  rtn = ctrl->getLookAheadDistanceWrapper(twist);
  EXPECT_EQ(rtn, 0.6);

  // now it should be a function of velocity, max out
  ctrl->setVelocityScaledLookAhead();
  rtn = ctrl->getLookAheadDistanceWrapper(twist);
  EXPECT_EQ(rtn, 0.9);  // 10 speed maxes out at max_lookahead_dist

  // check normal range
  twist.linear.x = 0.35;
  rtn = ctrl->getLookAheadDistanceWrapper(twist);
  EXPECT_NEAR(rtn, 0.525, 0.0001);  // 1.5 * 0.35

  // check minimum range
  twist.linear.x = 0.0;
  rtn = ctrl->getLookAheadDistanceWrapper(twist);
  EXPECT_EQ(rtn, 0.3);
}

TEST(RegulatedPurePursuitTest, rotateTests)
{
  // --------------------------
  // Non-Stateful Configuration
  // --------------------------
  auto ctrl = std::make_shared<BasicAPIRPP>();
  auto node = std::make_shared<nav2::LifecycleNode>("testRPP");
  nav2::declare_parameter_if_not_declared(
    node, "PathFollower.stateful", rclcpp::ParameterValue(false));

  std::string name = "PathFollower";
  auto tf = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  auto costmap = std::make_shared<nav2_costmap_2d::Costmap2DROS>("fake_costmap");
  rclcpp_lifecycle::State state;
  costmap->on_configure(state);
  ctrl->configure(node, name, tf, costmap);

  // shouldRotateToPath
  geometry_msgs::msg::PoseStamped carrot;
  double angle_to_path_rtn;
  EXPECT_EQ(ctrl->shouldRotateToPathWrapper(carrot, angle_to_path_rtn), false);

  carrot.pose.position.x = 0.5;
  carrot.pose.position.y = 0.25;
  EXPECT_EQ(ctrl->shouldRotateToPathWrapper(carrot, angle_to_path_rtn), false);

  carrot.pose.position.x = 0.5;
  carrot.pose.position.y = 1.0;
  EXPECT_EQ(ctrl->shouldRotateToPathWrapper(carrot, angle_to_path_rtn), true);

  // shouldRotateToGoalHeading
  carrot.pose.position.x = 0.0;
  carrot.pose.position.y = 0.0;
  EXPECT_EQ(ctrl->shouldRotateToGoalHeadingWrapper(carrot), true);

  carrot.pose.position.x = 0.0;
  carrot.pose.position.y = 0.24;
  EXPECT_EQ(ctrl->shouldRotateToGoalHeadingWrapper(carrot), true);

  carrot.pose.position.x = 0.0;
  carrot.pose.position.y = 0.26;
  EXPECT_EQ(ctrl->shouldRotateToGoalHeadingWrapper(carrot), false);

  // rotateToHeading
  double lin_v = 10.0;
  double ang_v = 0.5;
  double angle_to_path = 0.4;
  geometry_msgs::msg::Twist curr_speed;
  curr_speed.angular.z = 1.75;

  // basic full speed at a speed
  ctrl->rotateToHeadingWrapper(lin_v, ang_v, angle_to_path, curr_speed);
  EXPECT_EQ(lin_v, 0.0);
  EXPECT_EQ(ang_v, 1.6);  // hit slow down limit

  // negative direction
  angle_to_path = -0.4;
  curr_speed.angular.z = -1.75;
  ctrl->rotateToHeadingWrapper(lin_v, ang_v, angle_to_path, curr_speed);
  EXPECT_EQ(ang_v, -1.6);  // hit slow down limit

  // kinematic clamping, no speed, some speed accelerating, some speed decelerating
  angle_to_path = 0.4;
  curr_speed.angular.z = 0.0;
  ctrl->rotateToHeadingWrapper(lin_v, ang_v, angle_to_path, curr_speed);
  EXPECT_NEAR(ang_v, 0.16, 0.01);

  curr_speed.angular.z = 1.0;
  ctrl->rotateToHeadingWrapper(lin_v, ang_v, angle_to_path, curr_speed);
  EXPECT_NEAR(ang_v, 1.16, 0.01);

  angle_to_path = -0.4;
  curr_speed.angular.z = 1.0;
  ctrl->rotateToHeadingWrapper(lin_v, ang_v, angle_to_path, curr_speed);
  EXPECT_NEAR(ang_v, 0.84, 0.01);

  // -----------------------
  // Stateful Configuration
  // -----------------------
  node->set_parameter(
    rclcpp::Parameter("PathFollower.stateful", true));

  ctrl->configure(node, name, tf, costmap);

  // Start just outside tolerance
  carrot.pose.position.x = 0.0;
  carrot.pose.position.y = 0.26;
  EXPECT_EQ(ctrl->shouldRotateToGoalHeadingWrapper(carrot), false);

  // Enter tolerance (should set internal flag)
  carrot.pose.position.y = 0.24;
  EXPECT_EQ(ctrl->shouldRotateToGoalHeadingWrapper(carrot), true);

  // Move outside tolerance again - still expect true (due to persistent state)
  carrot.pose.position.y = 0.26;
  EXPECT_EQ(ctrl->shouldRotateToGoalHeadingWrapper(carrot), true);
}

TEST(RegulatedPurePursuitTest, applyConstraints)
{
  auto ctrl = std::make_shared<BasicAPIRPP>();
  auto node = std::make_shared<nav2::LifecycleNode>("testRPP");
  std::string name = "PathFollower";
  auto tf = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  auto costmap = std::make_shared<nav2_costmap_2d::Costmap2DROS>("fake_costmap");
  rclcpp_lifecycle::State state;
  costmap->on_configure(state);

  constexpr double approach_velocity_scaling_dist = 0.6;
  nav2::declare_parameter_if_not_declared(
    node,
    name + ".approach_velocity_scaling_dist",
    rclcpp::ParameterValue(approach_velocity_scaling_dist));

  ctrl->configure(node, name, tf, costmap);

  auto no_approach_path = path_utils::generate_path(
    geometry_msgs::msg::PoseStamped(), 0.1, {
    std::make_unique<path_utils::Straight>(approach_velocity_scaling_dist + 1.0)
  });

  double curvature = 0.5;
  geometry_msgs::msg::Twist curr_speed;
  double pose_cost = 0.0;
  double linear_vel = 0.0;
  double sign = 1.0;

  // test curvature regulation (default)
  curr_speed.linear.x = 0.25;
  ctrl->applyConstraintsWrapper(
    curvature, curr_speed, pose_cost, no_approach_path,
    linear_vel, sign);
  EXPECT_EQ(linear_vel, 0.25);  // min set speed

  linear_vel = 1.0;
  curvature = 0.7407;
  curr_speed.linear.x = 0.5;
  ctrl->applyConstraintsWrapper(
    curvature, curr_speed, pose_cost, no_approach_path,
    linear_vel, sign);
  EXPECT_NEAR(linear_vel, 0.5, 0.01);  // lower by curvature

  linear_vel = 1.0;
  curvature = 1000.0;
  curr_speed.linear.x = 0.25;
  ctrl->applyConstraintsWrapper(
    curvature, curr_speed, pose_cost, no_approach_path,
    linear_vel, sign);
  EXPECT_NEAR(linear_vel, 0.25, 0.01);  // min out by curvature

  // Approach velocity scaling on a path with no distance left
  auto approach_path = path_utils::generate_path(
    geometry_msgs::msg::PoseStamped(), 0.1, {
    std::make_unique<path_utils::Straight>(0.0)
  });

  linear_vel = 1.0;
  curvature = 0.0;
  curr_speed.linear.x = 0.25;
  ctrl->applyConstraintsWrapper(
    curvature, curr_speed, pose_cost, approach_path,
    linear_vel, sign);
  EXPECT_NEAR(linear_vel, 0.05, 0.01);  // min out on min approach velocity

  // now try with cost regulation (turn off velocity and only cost)
  // ctrl->setCostRegulationScaling();
  // ctrl->resetVelocityRegulationScaling();
  // curvature = 0.0;

  // min changeable cost
  // pose_cost = 1;
  // linear_vel = 0.5;
  // curr_speed.linear.x = 0.5;
  // ctrl->applyConstraintsWrapper(
  //   dist_error, lookahead_dist, curvature, curr_speed, pose_cost, linear_vel);
  // EXPECT_NEAR(linear_vel, 0.498, 0.01);

  // max changing cost
  // pose_cost = 127;
  // curr_speed.linear.x = 0.255;
  // ctrl->applyConstraintsWrapper(
  //   dist_error, lookahead_dist, curvature, curr_speed, pose_cost, linear_vel);
  // EXPECT_NEAR(linear_vel, 0.255, 0.01);

  // over max cost thresh
  // pose_cost = 200;
  // curr_speed.linear.x = 0.25;
  // ctrl->applyConstraintsWrapper(
  //   dist_error, lookahead_dist, curvature, curr_speed, pose_cost, linear_vel);
  // EXPECT_NEAR(linear_vel, 0.25, 0.01);

  // test kinematic clamping
  // pose_cost = 200;
  // curr_speed.linear.x = 1.0;
  // ctrl->applyConstraintsWrapper(
  //   dist_error, lookahead_dist, curvature, curr_speed, pose_cost, linear_vel);
  // EXPECT_NEAR(linear_vel, 0.5, 0.01);
}

TEST(RegulatedPurePursuitTest, testDynamicParameter)
{
  auto node = std::make_shared<nav2::LifecycleNode>("Smactest");
  auto costmap = std::make_shared<nav2_costmap_2d::Costmap2DROS>("global_costmap");
  costmap->on_configure(rclcpp_lifecycle::State());
  auto ctrl =
    std::make_unique<nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController>();
  auto tf = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  ctrl->configure(node, "test", tf, costmap);
  ctrl->activate();

  auto rec_param = std::make_shared<rclcpp::AsyncParametersClient>(
    node->get_node_base_interface(), node->get_node_topics_interface(),
    node->get_node_graph_interface(),
    node->get_node_services_interface());

  auto results = rec_param->set_parameters_atomically(
    {rclcpp::Parameter("test.max_linear_vel", 1.0),
      rclcpp::Parameter("test.min_linear_vel", -1.0),
      rclcpp::Parameter("test.max_angular_vel", 2.0),
      rclcpp::Parameter("test.min_angular_vel", -2.0),
      rclcpp::Parameter("test.max_linear_accel", 2.0),
      rclcpp::Parameter("test.max_linear_decel", -2.0),
      rclcpp::Parameter("test.max_angular_accel", 3.0),
      rclcpp::Parameter("test.max_angular_decel", -3.0),
      rclcpp::Parameter("test.lookahead_dist", 7.0),
      rclcpp::Parameter("test.max_lookahead_dist", 7.0),
      rclcpp::Parameter("test.min_lookahead_dist", 6.0),
      rclcpp::Parameter("test.lookahead_time", 1.8),
      rclcpp::Parameter("test.rotate_to_heading_angular_vel", 18.0),
      rclcpp::Parameter("test.min_approach_linear_velocity", 1.0),
      rclcpp::Parameter("test.approach_velocity_scaling_dist", 0.8),
      rclcpp::Parameter("test.max_allowed_time_to_collision_up_to_carrot", 2.0),
      rclcpp::Parameter("test.min_distance_to_obstacle", 2.0),
      rclcpp::Parameter("test.cost_scaling_dist", 2.0),
      rclcpp::Parameter("test.cost_scaling_gain", 4.0),
      rclcpp::Parameter("test.regulated_linear_scaling_min_radius", 10.0),
      rclcpp::Parameter("test.rotate_to_heading_min_angle", 0.7),
      rclcpp::Parameter("test.regulated_linear_scaling_min_speed", 4.0),
      rclcpp::Parameter("test.use_velocity_scaled_lookahead_dist", false),
      rclcpp::Parameter("test.use_regulated_linear_velocity_scaling", false),
      rclcpp::Parameter("test.use_cost_regulated_linear_velocity_scaling", false),
      rclcpp::Parameter("test.inflation_cost_scaling_factor", 1.0),
      rclcpp::Parameter("test.allow_reversing", false),
      rclcpp::Parameter("test.use_path_segment_direction_for_reversing", true),
      rclcpp::Parameter("test.minimum_turning_radius", 6.0),
      rclcpp::Parameter("test.use_rotate_to_heading", false),
      rclcpp::Parameter("test.stateful", false),
      rclcpp::Parameter("test.use_dynamic_window", true),
      rclcpp::Parameter("test.allow_obstacle_checking_beyond_goal", false)});

  rclcpp::spin_until_future_complete(
    node->get_node_base_interface(),
    results);

  EXPECT_EQ(node->get_parameter("test.max_linear_vel").as_double(), 1.0);
  EXPECT_EQ(node->get_parameter("test.min_linear_vel").as_double(), -1.0);
  EXPECT_EQ(node->get_parameter("test.max_angular_vel").as_double(), 2.0);
  EXPECT_EQ(node->get_parameter("test.min_angular_vel").as_double(), -2.0);
  EXPECT_EQ(node->get_parameter("test.max_linear_accel").as_double(), 2.0);
  EXPECT_EQ(node->get_parameter("test.max_linear_decel").as_double(), -2.0);
  EXPECT_EQ(node->get_parameter("test.max_angular_accel").as_double(), 3.0);
  EXPECT_EQ(node->get_parameter("test.max_angular_decel").as_double(), -3.0);
  EXPECT_EQ(node->get_parameter("test.lookahead_dist").as_double(), 7.0);
  EXPECT_EQ(node->get_parameter("test.max_lookahead_dist").as_double(), 7.0);
  EXPECT_EQ(node->get_parameter("test.min_lookahead_dist").as_double(), 6.0);
  EXPECT_EQ(node->get_parameter("test.lookahead_time").as_double(), 1.8);
  EXPECT_EQ(node->get_parameter("test.rotate_to_heading_angular_vel").as_double(), 18.0);
  EXPECT_EQ(node->get_parameter("test.min_approach_linear_velocity").as_double(), 1.0);
  EXPECT_EQ(node->get_parameter("test.approach_velocity_scaling_dist").as_double(), 0.8);
  EXPECT_EQ(
    node->get_parameter(
      "test.max_allowed_time_to_collision_up_to_carrot").as_double(), 2.0);
  EXPECT_EQ(node->get_parameter("test.min_distance_to_obstacle").as_double(), 2.0);
  EXPECT_EQ(node->get_parameter("test.cost_scaling_dist").as_double(), 2.0);
  EXPECT_EQ(node->get_parameter("test.cost_scaling_gain").as_double(), 4.0);
  EXPECT_EQ(node->get_parameter("test.regulated_linear_scaling_min_radius").as_double(), 10.0);
  EXPECT_EQ(node->get_parameter("test.rotate_to_heading_min_angle").as_double(), 0.7);
  EXPECT_EQ(node->get_parameter("test.regulated_linear_scaling_min_speed").as_double(), 4.0);
  EXPECT_EQ(node->get_parameter("test.use_velocity_scaled_lookahead_dist").as_bool(), false);
  EXPECT_EQ(node->get_parameter("test.use_regulated_linear_velocity_scaling").as_bool(), false);
  EXPECT_EQ(node->get_parameter("test.inflation_cost_scaling_factor").as_double(), 1.0);
  EXPECT_EQ(
    node->get_parameter(
      "test.use_cost_regulated_linear_velocity_scaling").as_bool(), false);
  EXPECT_EQ(node->get_parameter("test.allow_reversing").as_bool(), false);
  EXPECT_EQ(
    node->get_parameter(
      "test.use_path_segment_direction_for_reversing").as_bool(), true);
  EXPECT_EQ(node->get_parameter("test.minimum_turning_radius").as_double(), 6.0);
  EXPECT_EQ(node->get_parameter("test.use_rotate_to_heading").as_bool(), false);
  EXPECT_EQ(node->get_parameter("test.stateful").as_bool(), false);
  EXPECT_EQ(node->get_parameter("test.use_dynamic_window").as_bool(), true);
  EXPECT_EQ(node->get_parameter("test.allow_obstacle_checking_beyond_goal").as_bool(), false);

  // Should fail
  auto results2 = rec_param->set_parameters_atomically(
    {rclcpp::Parameter("test.inflation_cost_scaling_factor", -1.0)});

  rclcpp::spin_until_future_complete(
    node->get_node_base_interface(),
    results2);

  auto results3 = rec_param->set_parameters_atomically(
    {rclcpp::Parameter("test.use_rotate_to_heading", true)});

  rclcpp::spin_until_future_complete(
    node->get_node_base_interface(),
    results3);

  auto results4 = rec_param->set_parameters_atomically(
    {rclcpp::Parameter("test.allow_reversing", false),
      rclcpp::Parameter("test.use_rotate_to_heading", true),
      rclcpp::Parameter("test.allow_reversing", true)});

  rclcpp::spin_until_future_complete(
    node->get_node_base_interface(),
    results4);
}

TEST(RegulatedPurePursuitTest, reversingDirectionUsesCurrentPathSegmentDirection)
{
  auto ctrl = std::make_shared<BasicAPIRPP>();
  auto node = std::make_shared<nav2::LifecycleNode>("testRPPPathDirection");
  std::string name = "PathFollower";
  auto tf = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  auto costmap = std::make_shared<nav2_costmap_2d::Costmap2DROS>("fake_costmap");
  rclcpp_lifecycle::State state;
  costmap->on_configure(state);

  ctrl->configure(node, name, tf, costmap);
  ctrl->setAllowReversing(true);
  ctrl->setUsePathSegmentDirectionForReversing(true);

  nav_msgs::msg::Path path;
  auto add_pose = [&path](double x, double y, double yaw) {
    geometry_msgs::msg::PoseStamped pose;
    pose.pose.position.x = x;
    pose.pose.position.y = y;
    pose.pose.orientation.z = std::sin(yaw / 2.0);
    pose.pose.orientation.w = std::cos(yaw / 2.0);
    path.poses.push_back(pose);
  };
  add_pose(0.0, 0.0, M_PI / 2.0);
  add_pose(0.0, 1.0, M_PI / 2.0);
  add_pose(0.0, 0.4, M_PI / 2.0);

  EXPECT_EQ(ctrl->getVelocitySignWrapper(path, 0.2, -0.08), 1.0);
  EXPECT_EQ(ctrl->getVelocitySignWrapper(path, 0.8, 0.08), 1.0);
  EXPECT_EQ(ctrl->getVelocitySignWrapper(path, 1.2, 0.08), -1.0);

  nav_msgs::msg::Path reverse_path;
  auto add_reverse_pose = [&reverse_path](double x, double y, double yaw) {
    geometry_msgs::msg::PoseStamped pose;
    pose.pose.position.x = x;
    pose.pose.position.y = y;
    pose.pose.orientation.z = std::sin(yaw / 2.0);
    pose.pose.orientation.w = std::cos(yaw / 2.0);
    reverse_path.poses.push_back(pose);
  };
  add_reverse_pose(0.0, 1.0, M_PI / 2.0);
  add_reverse_pose(0.0, 0.4, M_PI / 2.0);

  EXPECT_EQ(ctrl->getVelocitySignWrapper(reverse_path, 0.2, 0.08), -1.0);
  EXPECT_EQ(ctrl->getVelocitySignWrapper(reverse_path, 0.2, -0.08), -1.0);

  ctrl->setUsePathSegmentDirectionForReversing(false);
  EXPECT_EQ(ctrl->getVelocitySignWrapper(path, 1.2, -0.08), -1.0);
  EXPECT_EQ(ctrl->getVelocitySignWrapper(reverse_path, 0.2, 0.08), 1.0);

  ctrl->setAllowReversing(false);
  EXPECT_EQ(ctrl->getVelocitySignWrapper(reverse_path, 0.2, -0.08), 1.0);

  ctrl->cleanup();
}

TEST(RegulatedPurePursuitTest, computeVelocityByDWPP)
{
  auto ctrl = std::make_shared<BasicAPIRPP>();
  auto node = std::make_shared<nav2::LifecycleNode>("testRPP");
  std::string name = "PathFollower";
  auto tf = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  auto costmap = std::make_shared<nav2_costmap_2d::Costmap2DROS>("fake_costmap");
  rclcpp_lifecycle::State state;
  costmap->on_configure(state);

  // Enable DWPP so computeVelocityCommands executes the dynamic window branch.
  nav2::declare_parameter_if_not_declared(
    node, name + ".use_dynamic_window", rclcpp::ParameterValue(false));
  node->set_parameter(rclcpp::Parameter(name + ".use_dynamic_window", true));

  // Disable collision detection to simplify test.
  nav2::declare_parameter_if_not_declared(
    node, name + ".use_collision_detection", rclcpp::ParameterValue(true));
  node->set_parameter(rclcpp::Parameter(name + ".use_collision_detection", false));

  ctrl->configure(node, name, tf, costmap);
  ctrl->activate();
  nav2_controller::FeasiblePathHandler path_handler;
  path_handler.initialize(node, node->get_logger(), "path_handler", costmap, tf);

  auto stamp = node->get_clock()->now();

  // Simple straight path ahead of the robot in map frame.
  geometry_msgs::msg::PoseStamped start_pose;
  start_pose.header.frame_id = "map";
  start_pose.header.stamp = stamp;
  start_pose.pose.orientation.w = 1.0;
  auto plan = path_utils::generate_path(
    start_pose, 0.1,
    {std::make_unique<path_utils::Straight>(2.0)});
  ctrl->newPathReceived(plan);
  path_handler.setPlan(plan);

  // Provide transform into base frame so the controller can compute commands.
  geometry_msgs::msg::TransformStamped map_to_base;
  map_to_base.header.stamp = stamp;
  map_to_base.header.frame_id = "map";
  map_to_base.child_frame_id = "base_link";
  map_to_base.transform.rotation.w = 1.0;
  tf->setTransform(map_to_base, "dwpp-controller-test");

  geometry_msgs::msg::PoseStamped robot_pose;
  robot_pose.header.frame_id = "base_link";
  robot_pose.header.stamp = stamp;
  robot_pose.pose.orientation.w = 1.0;
  geometry_msgs::msg::Twist current_speed;
  TestGoalChecker checker;

  auto [closest_point, pruned_plan_end] = path_handler.findPlanSegment(robot_pose);
  nav_msgs::msg::Path transformed_global_plan = path_handler.transformLocalPlan(
    closest_point, pruned_plan_end);
  auto goal = path_handler.getTransformedGoal(robot_pose.header.stamp);
  auto cmd_vel = ctrl->computeVelocityCommands(
    robot_pose, current_speed, &checker, transformed_global_plan, goal);

  EXPECT_EQ(cmd_vel.twist.linear.x, 0.125);
  EXPECT_EQ(cmd_vel.twist.angular.z, 0.0);

  ctrl->deactivate();
  ctrl->cleanup();
}

// Shared helper: run one computeVelocityCommands cycle on a given plan and return the twist.
// Uses the FeasiblePathHandler exactly like computeVelocityByDWPP does above.
static geometry_msgs::msg::TwistStamped runControllerOnce(
  BasicAPIRPP & ctrl,
  nav2::LifecycleNode::SharedPtr node,
  std::shared_ptr<tf2_ros::Buffer> tf,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap,
  const nav_msgs::msg::Path & plan)
{
  nav2_controller::FeasiblePathHandler path_handler;
  path_handler.initialize(node, node->get_logger(), "path_handler", costmap, tf);

  auto stamp = node->get_clock()->now();
  ctrl.newPathReceived(plan);
  path_handler.setPlan(plan);

  geometry_msgs::msg::TransformStamped map_to_base;
  map_to_base.header.stamp = stamp;
  map_to_base.header.frame_id = "map";
  map_to_base.child_frame_id = "base_link";
  map_to_base.transform.rotation.w = 1.0;
  tf->setTransform(map_to_base, "min-turn-radius-test");

  geometry_msgs::msg::PoseStamped robot_pose;
  robot_pose.header.frame_id = "base_link";
  robot_pose.header.stamp = stamp;
  robot_pose.pose.orientation.w = 1.0;
  geometry_msgs::msg::Twist current_speed;
  TestGoalChecker checker;

  auto [closest_point, pruned_plan_end] = path_handler.findPlanSegment(robot_pose);
  nav_msgs::msg::Path transformed_global_plan = path_handler.transformLocalPlan(
    closest_point, pruned_plan_end);
  auto goal = path_handler.getTransformedGoal(robot_pose.header.stamp);
  return ctrl.computeVelocityCommands(
    robot_pose, current_speed, &checker, transformed_global_plan, goal);
}

TEST(RegulatedPurePursuitTest, minimumTurningRadiusClampsCurvature)
{
  // A sharp left arc of radius 0.5 m; robot at origin heading +x; lookahead ~0.6 m.
  // Without a clamp, |kappa| ~= 2.0. With minimum_turning_radius = 1.0, |kappa| <= 1.0.
  auto ctrl = std::make_shared<BasicAPIRPP>();
  auto node = std::make_shared<nav2::LifecycleNode>("testRPPClamp");
  std::string name = "PathFollower";
  auto tf = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  auto costmap = std::make_shared<nav2_costmap_2d::Costmap2DROS>("fake_costmap");
  costmap->on_configure(rclcpp_lifecycle::State());
  ctrl->configure(node, name, tf, costmap);
  ctrl->activate();

  ctrl->setUseCollisionDetection(false);

  geometry_msgs::msg::PoseStamped start_pose;
  start_pose.header.frame_id = "map";
  start_pose.header.stamp = node->get_clock()->now();
  start_pose.pose.orientation.w = 1.0;
  auto plan = path_utils::generate_path(
    start_pose, 0.05,
    {std::make_unique<path_utils::LeftTurn>(0.5)});

  // First: unclamped (minimum_turning_radius == 0.0). Expect |kappa| ~ 2 (== 1/0.5).
  ctrl->setMinimumTurningRadius(0.0);
  auto cmd_unclamped = runControllerOnce(*ctrl, node, tf, costmap, plan);
  const double kappa_unclamped =
    cmd_unclamped.twist.angular.z / cmd_unclamped.twist.linear.x;
  EXPECT_GT(std::fabs(kappa_unclamped), 1.0 + 1e-3);  // curvature above 1/R

  // Second: clamped to 1/1.0 = 1.0
  ctrl->setMinimumTurningRadius(1.0);
  auto cmd_clamped = runControllerOnce(*ctrl, node, tf, costmap, plan);
  ASSERT_GT(std::fabs(cmd_clamped.twist.linear.x), 1e-6);
  const double kappa_clamped =
    cmd_clamped.twist.angular.z / cmd_clamped.twist.linear.x;
  EXPECT_LE(std::fabs(kappa_clamped), 1.0 + 1e-6);

  ctrl->deactivate();
  ctrl->cleanup();
}

TEST(RegulatedPurePursuitTest, minimumTurningRadiusOmegaCapReducesV)
{
  // Force the omega cap to bind: set max_angular_vel below v_max * (1/R) so that the
  // natural |omega| = |v * kappa| exceeds max_angular_vel. Verify that (a) v is reduced
  // to omega_cap / |kappa| and (b) the curvature omega/v is preserved (== 1/R clamp).
  auto ctrl = std::make_shared<BasicAPIRPP>();
  auto node = std::make_shared<nav2::LifecycleNode>("testRPPOmegaCap");
  std::string name = "PathFollower";
  auto tf = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  auto costmap = std::make_shared<nav2_costmap_2d::Costmap2DROS>("fake_costmap");
  costmap->on_configure(rclcpp_lifecycle::State());
  ctrl->configure(node, name, tf, costmap);
  ctrl->activate();

  ctrl->setUseCollisionDetection(false);

  // Regulation curvature will be clamped to 1/0.5 = 2.0. RPP's curvature/approach
  // speed regulation determines the pre-cap v (it is well above 0.025 for this arc),
  // so a cap of 0.05 rad/s is guaranteed to bind regardless of regulation internals:
  // v must be reduced to 0.05 / 2.0 = 0.025 with curvature preserved exactly.
  ctrl->setMinimumTurningRadius(0.5);
  ctrl->setMaxAngularVel(0.05);
  ctrl->setMinAngularVel(-0.05);

  geometry_msgs::msg::PoseStamped start_pose;
  start_pose.header.frame_id = "map";
  start_pose.header.stamp = node->get_clock()->now();
  start_pose.pose.orientation.w = 1.0;
  // Radius-0.25 left arc gives raw |kappa| ~ 4, well above the clamp of 2.
  auto plan = path_utils::generate_path(
    start_pose, 0.05,
    {std::make_unique<path_utils::LeftTurn>(0.25)});

  auto cmd = runControllerOnce(*ctrl, node, tf, costmap, plan);
  ASSERT_GT(std::fabs(cmd.twist.linear.x), 1e-6);

  const double kappa_cmd = cmd.twist.angular.z / cmd.twist.linear.x;
  // Curvature preserved at the clamp value 1/R = 2.0.
  EXPECT_NEAR(std::fabs(kappa_cmd), 2.0, 1e-6);
  // |omega| at its cap.
  EXPECT_NEAR(std::fabs(cmd.twist.angular.z), 0.05, 1e-6);
  // |v| reduced to omega_cap / |kappa| = 0.05 / 2.0 = 0.025.
  EXPECT_NEAR(cmd.twist.linear.x, 0.025, 1e-6);

  ctrl->deactivate();
  ctrl->cleanup();
}

TEST(RegulatedPurePursuitTest, minimumTurningRadiusDefaultOffBitIdentical)
{
  // With minimum_turning_radius = 0.0 (default) the produced command must be identical
  // to the pre-change behavior. We check invariance against a curved path in both the
  // regular and dynamic-window branches by recording the command with the feature
  // explicitly disabled and comparing to the same command produced fresh.
  auto make_and_run = [&](bool use_dw, double min_radius) {
      auto ctrl = std::make_shared<BasicAPIRPP>();
      auto node = std::make_shared<nav2::LifecycleNode>("testRPPDefaultOff");
      std::string name = "PathFollower";
      auto tf = std::make_shared<tf2_ros::Buffer>(node->get_clock());
      auto costmap = std::make_shared<nav2_costmap_2d::Costmap2DROS>("fake_costmap");
      costmap->on_configure(rclcpp_lifecycle::State());
      ctrl->configure(node, name, tf, costmap);
      ctrl->activate();
      ctrl->setUseCollisionDetection(false);
      ctrl->setUseDynamicWindow(use_dw);
      ctrl->setMinimumTurningRadius(min_radius);
      // Wide omega envelope in both runs so the omega-cap logic cannot bind and the
      // only difference under test is the feature gate itself.
      ctrl->setMaxAngularVel(100.0);
      ctrl->setMinAngularVel(-100.0);
      geometry_msgs::msg::PoseStamped start_pose;
      start_pose.header.frame_id = "map";
      start_pose.header.stamp = node->get_clock()->now();
      start_pose.pose.orientation.w = 1.0;
      auto plan = path_utils::generate_path(
        start_pose, 0.05,
        {std::make_unique<path_utils::LeftTurn>(0.5)});
      auto cmd = runControllerOnce(*ctrl, node, tf, costmap, plan);
      ctrl->deactivate();
      ctrl->cleanup();
      return cmd;
    };

  // Regular (non-DWPP) path: value with the feature off should match the value with
  // a tiny minimum_turning_radius (1e-6 m -> kappa_max = 1e6 1/m, clamp never binds).
  auto cmd_off_regular = make_and_run(false, 0.0);
  auto cmd_permissive_regular = make_and_run(false, 1e-6);
  EXPECT_DOUBLE_EQ(cmd_off_regular.twist.linear.x, cmd_permissive_regular.twist.linear.x);
  EXPECT_DOUBLE_EQ(cmd_off_regular.twist.angular.z, cmd_permissive_regular.twist.angular.z);

  // Dynamic-window path: same invariant.
  auto cmd_off_dwpp = make_and_run(true, 0.0);
  auto cmd_permissive_dwpp = make_and_run(true, 1e-6);
  EXPECT_DOUBLE_EQ(cmd_off_dwpp.twist.linear.x, cmd_permissive_dwpp.twist.linear.x);
  EXPECT_DOUBLE_EQ(cmd_off_dwpp.twist.angular.z, cmd_permissive_dwpp.twist.angular.z);
}

TEST(RegulatedPurePursuitTest, minimumTurningRadiusRampsFromStandstillDWPP)
{
  // Closed-loop regression against an absorbing zero-velocity state: with the
  // production configuration (dynamic window + curvature clamp active), repeated
  // control cycles from standstill must ramp |v| up. The DWPP solver seeds its
  // window from the internally stored last commanded velocity, so if any cycle
  // collapses the command to (0, 0) the stall is permanent — this test spins the
  // loop without resetting the plan to reproduce that feedback.
  auto ctrl = std::make_shared<BasicAPIRPP>();
  auto node = std::make_shared<nav2::LifecycleNode>("testRPPStandstillRamp");
  std::string name = "PathFollower";
  auto tf = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  auto costmap = std::make_shared<nav2_costmap_2d::Costmap2DROS>("fake_costmap");
  costmap->on_configure(rclcpp_lifecycle::State());
  ctrl->configure(node, name, tf, costmap);
  ctrl->activate();

  ctrl->setUseCollisionDetection(false);
  ctrl->setUseDynamicWindow(true);
  ctrl->setMinimumTurningRadius(6.0);
  auto * params = ctrl->paramsHandle();
  params->max_linear_vel = 0.5;
  params->min_linear_vel = -0.5;
  params->max_angular_vel = 0.0833;
  params->min_angular_vel = -0.0833;
  params->max_linear_accel = 1.0;
  params->max_linear_decel = -1.0;
  params->max_angular_accel = 0.25;
  params->max_angular_decel = -0.25;
  params->regulated_linear_scaling_min_radius = 12.0;
  params->regulated_linear_scaling_min_speed = 0.3;
  params->lookahead_dist = 0.7;
  params->use_velocity_scaled_lookahead_dist = false;

  geometry_msgs::msg::PoseStamped start_pose;
  start_pose.header.frame_id = "map";
  start_pose.header.stamp = node->get_clock()->now();
  start_pose.pose.orientation.w = 1.0;
  // Minimum-radius left arc: the tightest feasible production geometry.
  auto plan = path_utils::generate_path(
    start_pose, 0.05,
    {std::make_unique<path_utils::LeftTurn>(6.0)});

  nav2_controller::FeasiblePathHandler path_handler;
  path_handler.initialize(node, node->get_logger(), "path_handler", costmap, tf);
  ctrl->newPathReceived(plan);
  path_handler.setPlan(plan);

  geometry_msgs::msg::TransformStamped map_to_base;
  map_to_base.header.stamp = node->get_clock()->now();
  map_to_base.header.frame_id = "map";
  map_to_base.child_frame_id = "base_link";
  map_to_base.transform.rotation.w = 1.0;
  tf->setTransform(map_to_base, "standstill-ramp-test", true);

  geometry_msgs::msg::PoseStamped robot_pose;
  robot_pose.header.frame_id = "base_link";
  robot_pose.header.stamp = node->get_clock()->now();
  robot_pose.pose.orientation.w = 1.0;
  geometry_msgs::msg::Twist current_speed;
  TestGoalChecker checker;

  auto run_cycles = [&](const nav_msgs::msg::Path & p, int n) {
      nav2_controller::FeasiblePathHandler handler2;
      handler2.initialize(node, node->get_logger(), "path_handler2", costmap, tf);
      ctrl->newPathReceived(p);
      handler2.setPlan(p);
      geometry_msgs::msg::Twist speed;
      double max_v = 0.0;
      for (int cycle = 0; cycle < n; ++cycle) {
        auto [cp, ppe] = handler2.findPlanSegment(robot_pose);
        nav_msgs::msg::Path tp = handler2.transformLocalPlan(cp, ppe);
        auto g = handler2.getTransformedGoal(robot_pose.header.stamp);
        auto cmd = ctrl->computeVelocityCommands(robot_pose, speed, &checker, tp, g);
        speed = cmd.twist;
        max_v = std::max(max_v, std::fabs(cmd.twist.linear.x));
      }
      return max_v;
    };

  double max_abs_v = 0.0;
  for (int cycle = 0; cycle < 40; ++cycle) {
    auto [closest_point, pruned_plan_end] = path_handler.findPlanSegment(robot_pose);
    nav_msgs::msg::Path transformed_global_plan = path_handler.transformLocalPlan(
      closest_point, pruned_plan_end);
    auto goal = path_handler.getTransformedGoal(robot_pose.header.stamp);
    auto cmd = ctrl->computeVelocityCommands(
      robot_pose, current_speed, &checker, transformed_global_plan, goal);
    current_speed = cmd.twist;
    max_abs_v = std::max(max_abs_v, std::fabs(cmd.twist.linear.x));
  }
  // A healthy first cycle alone reaches ~max_linear_accel * dt = 0.05 m/s.
  EXPECT_GT(max_abs_v, 0.04);

  // Regression for the field-observed zero-lock: a nearly straight path whose
  // curvature falls in [1e-6, 1e-3) — below the DWPP straightness threshold but
  // above the old projection epsilon. The DWPP solver returns omega = 0 there;
  // a projection that caps v by the RETURNED omega collapses the command to
  // (0, 0) permanently. The huge-radius arc below has kappa ~= 5e-4.
  auto near_straight_plan = path_utils::generate_path(
    start_pose, 0.5,
    {std::make_unique<path_utils::LeftTurn>(2000.0)});
  const double max_v_near_straight = run_cycles(near_straight_plan, 40);
  EXPECT_GT(max_v_near_straight, 0.04);

  ctrl->deactivate();
  ctrl->cleanup();
}

TEST(RegulatedPurePursuitTest, minimumTurningRadiusInDynamicWindow)
{
  // Same clamping invariant must hold on the DWPP exit path.
  auto ctrl = std::make_shared<BasicAPIRPP>();
  auto node = std::make_shared<nav2::LifecycleNode>("testRPPClampDWPP");
  std::string name = "PathFollower";
  auto tf = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  auto costmap = std::make_shared<nav2_costmap_2d::Costmap2DROS>("fake_costmap");
  costmap->on_configure(rclcpp_lifecycle::State());
  ctrl->configure(node, name, tf, costmap);
  ctrl->activate();
  ctrl->setUseCollisionDetection(false);
  ctrl->setUseDynamicWindow(true);
  ctrl->setMinimumTurningRadius(1.0);

  geometry_msgs::msg::PoseStamped start_pose;
  start_pose.header.frame_id = "map";
  start_pose.header.stamp = node->get_clock()->now();
  start_pose.pose.orientation.w = 1.0;
  auto plan = path_utils::generate_path(
    start_pose, 0.05,
    {std::make_unique<path_utils::LeftTurn>(0.5)});

  auto cmd = runControllerOnce(*ctrl, node, tf, costmap, plan);
  if (std::fabs(cmd.twist.linear.x) > 1e-6) {
    const double kappa_cmd = cmd.twist.angular.z / cmd.twist.linear.x;
    EXPECT_LE(std::fabs(kappa_cmd), 1.0 + 1e-6);
  } else {
    // Fallback: DWPP corner-picked. Curvature-preserving cap should have kept |omega|
    // in step with |v * kappa|; with v ~ 0 that requires omega ~ 0.
    EXPECT_NEAR(cmd.twist.angular.z, 0.0, 1e-6);
  }

  ctrl->deactivate();
  ctrl->cleanup();
}

TEST(RegulatedPurePursuitTest, testObstacleBeyondGoal)
{
  auto ctrl = std::make_shared<BasicAPIRPP>();
  auto node = std::make_shared<nav2::LifecycleNode>("testRPP");
  std::string name = "PathFollower";
  auto tf = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  auto costmap = std::make_shared<nav2_costmap_2d::Costmap2DROS>("fake_costmap");
  rclcpp_lifecycle::State state;
  costmap->on_configure(state);

  nav2::declare_parameter_if_not_declared(
    node, name + ".use_velocity_scaled_lookahead_dist", rclcpp::ParameterValue(true));
  nav2::declare_parameter_if_not_declared(
    node, name + ".max_lookahead_dist", rclcpp::ParameterValue(2.0));
  nav2::declare_parameter_if_not_declared(
    node, name + ".min_distance_to_obstacle", rclcpp::ParameterValue(1.5));
  nav2::declare_parameter_if_not_declared(
    node, name + ".use_collision_detection", rclcpp::ParameterValue(true));
  nav2::declare_parameter_if_not_declared(
    node, name + ".allow_obstacle_checking_beyond_goal", rclcpp::ParameterValue(false));

  ctrl->configure(node, name, tf, costmap);
  ctrl->activate();

  auto * raw_costmap = costmap->getCostmap();
  double resolution = raw_costmap->getResolution();
  double origin_x = raw_costmap->getOriginX();
  double origin_y = raw_costmap->getOriginY();
  unsigned int size_x = raw_costmap->getSizeInCellsX();
  unsigned int size_y = raw_costmap->getSizeInCellsY();

  double robot_x = origin_x + (size_x * resolution) / 2.0;
  double robot_y = origin_y + (size_y * resolution) / 2.0;

  geometry_msgs::msg::PoseStamped start_pose;
  start_pose.header.frame_id = costmap->getGlobalFrameID();
  start_pose.header.stamp = node->get_clock()->now();
  start_pose.pose.position.x = robot_x;
  start_pose.pose.position.y = robot_y;
  start_pose.pose.orientation.w = 1.0;
  auto plan = path_utils::generate_path(
    start_pose, 0.05,
    {std::make_unique<path_utils::Straight>(0.15)});
  ctrl->newPathReceived(plan);

  // "Place" a lethal obstacle 1.0m ahead (beyond path end, within min_distance_to_obstacle)
  unsigned int obs_mx, obs_my;
  raw_costmap->worldToMap(robot_x + 1.0, robot_y, obs_mx, obs_my);
  raw_costmap->setCost(obs_mx, obs_my, nav2_costmap_2d::LETHAL_OBSTACLE);
  raw_costmap->setCost(obs_mx, obs_my + 1, nav2_costmap_2d::LETHAL_OBSTACLE);
  raw_costmap->setCost(obs_mx + 1, obs_my, nav2_costmap_2d::LETHAL_OBSTACLE);
  raw_costmap->setCost(obs_mx + 1, obs_my + 1, nav2_costmap_2d::LETHAL_OBSTACLE);

  geometry_msgs::msg::PoseStamped robot_pose;
  robot_pose.header.frame_id = costmap->getGlobalFrameID();
  robot_pose.header.stamp = node->get_clock()->now();
  robot_pose.pose.position.x = robot_x;
  robot_pose.pose.position.y = robot_y;
  robot_pose.pose.orientation.w = 1.0;

  double linear_vel = 0.5;
  double angular_vel = 0.0;
  double carrot_dist = 0.15;
  double dist_to_path_end = 0.15;

  // When disabled: obstacle beyond goal should be ignored
  EXPECT_FALSE(
    ctrl->isCollisionImminentWrapper(
      robot_pose, linear_vel, angular_vel, carrot_dist, dist_to_path_end));

  node->set_parameter(
    rclcpp::Parameter(name + ".allow_obstacle_checking_beyond_goal", true));
  ctrl->configure(node, name, tf, costmap);
  ctrl->activate();
  ctrl->newPathReceived(plan);

  // Enabled: obstacle beyond goal should be detected
  EXPECT_TRUE(
    ctrl->isCollisionImminentWrapper(
      robot_pose, linear_vel, angular_vel, carrot_dist, dist_to_path_end));

  ctrl->deactivate();
  ctrl->cleanup();
}

TEST(RegulatedPurePursuitTest, testParameterWarnings)
{
  auto ctrl = std::make_shared<BasicAPIRPP>();
  auto node = std::make_shared<nav2::LifecycleNode>("testRPP");
  std::string name = "PathFollower";
  auto tf = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  auto costmap = std::make_shared<nav2_costmap_2d::Costmap2DROS>("fake_costmap");
  rclcpp_lifecycle::State state;
  costmap->on_configure(state);

  // min_distance_to_obstacle > lookahead_dist (fixed lookahead)
  nav2::declare_parameter_if_not_declared(
    node, name + ".use_collision_detection", rclcpp::ParameterValue(true));
  nav2::declare_parameter_if_not_declared(
    node, name + ".use_velocity_scaled_lookahead_dist", rclcpp::ParameterValue(false));
  nav2::declare_parameter_if_not_declared(
    node, name + ".lookahead_dist", rclcpp::ParameterValue(0.6));
  nav2::declare_parameter_if_not_declared(
    node, name + ".min_distance_to_obstacle", rclcpp::ParameterValue(1.5));
  nav2::declare_parameter_if_not_declared(
    node, name + ".allow_obstacle_checking_beyond_goal", rclcpp::ParameterValue(false));
  ctrl->configure(node, name, tf, costmap);
  ctrl->cleanup();

  // min_distance_to_obstacle > max_lookahead_dist (velocity scaled)
  node->set_parameter(rclcpp::Parameter(name + ".use_velocity_scaled_lookahead_dist", true));
  node->set_parameter(rclcpp::Parameter(name + ".max_lookahead_dist", 1.0));
  node->set_parameter(rclcpp::Parameter(name + ".min_distance_to_obstacle", 2.0));
  ctrl->configure(node, name, tf, costmap);
  ctrl->cleanup();

  // allow_obstacle_checking_beyond_goal without velocity scaled lookahead
  node->set_parameter(rclcpp::Parameter(name + ".use_velocity_scaled_lookahead_dist", false));
  node->set_parameter(rclcpp::Parameter(name + ".allow_obstacle_checking_beyond_goal", true));
  node->set_parameter(rclcpp::Parameter(name + ".min_distance_to_obstacle", 1.0));
  ctrl->configure(node, name, tf, costmap);
  ctrl->cleanup();

  // allow_obstacle_checking_beyond_goal with min_distance_to_obstacle <= 0.0
  node->set_parameter(rclcpp::Parameter(name + ".use_velocity_scaled_lookahead_dist", true));
  node->set_parameter(rclcpp::Parameter(name + ".min_distance_to_obstacle", -1.0));
  ctrl->configure(node, name, tf, costmap);
  ctrl->cleanup();
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  rclcpp::init(0, nullptr);

  int result = RUN_ALL_TESTS();

  rclcpp::shutdown();

  return result;
}
