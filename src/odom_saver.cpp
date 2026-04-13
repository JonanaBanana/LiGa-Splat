#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp_components/register_node_macro.hpp>

namespace airlab_lidar_3dgs {
class OdomSaver : public rclcpp::Node
{
public:
  OdomSaver(const rclcpp::NodeOptions & options)
  : Node("odom_saver", options)
  {
    const char* home_env = std::getenv("HOME");
    std::string home = home_env ? home_env : "/tmp";

    declare_parameter<std::string>("odom_topic", "/isaacsim/odom");
    declare_parameter<std::string>("output_file", home + "/dataset/airlab_3dgs/timestamps/odom.csv");
    declare_parameter<int>("save_interval", 1);

    odom_topic_ = get_parameter("odom_topic").as_string();
    output_file_ = get_parameter("output_file").as_string();
    save_interval_ = get_parameter("save_interval").as_int();

    RCLCPP_INFO(this->get_logger(),
      "OdomSaver initialized: topic='%s', interval=%d, output='%s'",
      odom_topic_.c_str(), save_interval_, output_file_.c_str());

    subscription_ = this->create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, 10,
      std::bind(&OdomSaver::odom_callback, this, std::placeholders::_1));
  }

  ~OdomSaver()
  {
    if (entries_.empty())
    {
      RCLCPP_INFO(this->get_logger(), "No odometry data recorded. Skipping CSV export.");
      return;
    }

    std::string csv_path = output_file_;
    std::ofstream csv(csv_path);
    if (!csv.is_open())
    {
      RCLCPP_ERROR(this->get_logger(), "Failed to open '%s' for writing.", csv_path.c_str());
      return;
    }

    csv << "index,timestamp_sec,timestamp_nanosec,"
        << "px,py,pz,"
        << "qx,qy,qz,qw\n";

    csv << std::fixed << std::setprecision(12);
    for (const auto & e : entries_)
    {
      csv << e.index << ","
          << e.stamp_sec << ","
          << e.stamp_nanosec << ","
          << e.px << "," << e.py << "," << e.pz << ","
          << e.qx << "," << e.qy << "," << e.qz << "," << e.qw << "\n";
    }
    csv.close();

    RCLCPP_INFO(this->get_logger(),
      "Saved %zu odometry entries to '%s'", entries_.size(), csv_path.c_str());
  }

private:
  struct OdomEntry
  {
    int index;
    int64_t stamp_sec;
    uint32_t stamp_nanosec;
    double px, py, pz;
    double qx, qy, qz, qw;
  };

  void odom_callback(const nav_msgs::msg::Odometry & msg)
  {
    frame_count_++;

    if (frame_count_ % save_interval_ != 0)
      return;

    const auto & pos = msg.pose.pose.position;
    const auto & ori = msg.pose.pose.orientation;

    entries_.push_back({
      static_cast<int>(entries_.size()),
      static_cast<int64_t>(msg.header.stamp.sec),
      msg.header.stamp.nanosec,
      pos.x, pos.y, pos.z,
      ori.x, ori.y, ori.z, ori.w
    });

    //RCLCPP_INFO(this->get_logger(),
    //  "Recorded pose %zu (frame %d): [%.3f, %.3f, %.3f]",
    //  entries_.size(), frame_count_, pos.x, pos.y, pos.z);
  }

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subscription_;

  std::string odom_topic_;
  std::string output_file_;
  int save_interval_;

  int frame_count_ = 0;
  std::vector<OdomEntry> entries_;
};

RCLCPP_COMPONENTS_REGISTER_NODE(airlab_lidar_3dgs::OdomSaver)
} // namespace airlab_lidar_3dgs