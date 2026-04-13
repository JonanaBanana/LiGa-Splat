#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp_components/register_node_macro.hpp>

namespace liga_splat {
class PathPublisher : public rclcpp::Node
{
public:
  PathPublisher(const rclcpp::NodeOptions & options)
  : Node("path_publisher", options)
  {
    declare_parameter<std::string>("odom_topic", "/odom");
    declare_parameter<std::string>("path_topic", "/liga_splat/path");
    declare_parameter<std::string>("frame_id", "World");
    declare_parameter<int>("max_path_length", 10000);

    odom_topic_ = get_parameter("odom_topic").as_string();
    path_topic_ = get_parameter("path_topic").as_string();
    frame_id_ = get_parameter("frame_id").as_string();
    max_path_length_ = get_parameter("max_path_length").as_int();

    RCLCPP_INFO(this->get_logger(),
      "PathPublisher initialized with odom topic: '%s', path topic: '%s', frame id: '%s'",
      odom_topic_.c_str(), path_topic_.c_str(), frame_id_.c_str());

    subscription_ = this->create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, 10,
      std::bind(&PathPublisher::odom_callback, this, std::placeholders::_1));

    publisher_ = this->create_publisher<nav_msgs::msg::Path>(path_topic_, 10);

    path_.header.frame_id = frame_id_;
  }

private:
  void odom_callback(const nav_msgs::msg::Odometry & msg)
  {
    geometry_msgs::msg::PoseStamped pose;
    pose.header = msg.header;
    pose.pose = msg.pose.pose;

    path_.header.stamp = msg.header.stamp;
    path_.poses.push_back(pose);
    if (static_cast<int>(path_.poses.size()) > max_path_length_)
        path_.poses.erase(path_.poses.begin());

    publisher_->publish(path_);
  }

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subscription_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher_;

  std::string odom_topic_;
  std::string path_topic_;
  std::string frame_id_;
  int max_path_length_;

  nav_msgs::msg::Path path_;
};

RCLCPP_COMPONENTS_REGISTER_NODE(liga_splat::PathPublisher)
} // namespace liga_splat