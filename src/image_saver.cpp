#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <opencv2/imgproc.hpp>

namespace liga_splat {
class ImageSaver : public rclcpp::Node
{
public:
  ImageSaver(const rclcpp::NodeOptions & options)
  : Node("image_saver", options)
  {
    const char* home_env = std::getenv("HOME");
    std::string home = home_env ? home_env : "/tmp";

    declare_parameter<std::string>("image_topic", "/isaacsim/rgb");
    declare_parameter<int>("save_interval", 10);
    declare_parameter<std::string>("image_dir", home + "/dataset/airlab_3dgs/distorted/images");
    declare_parameter<std::string>("timestamp_file", home + "/dataset/airlab_3dgs/timestamps/image.csv");
    declare_parameter<std::string>("image_prefix", "frame");
    image_topic_ = get_parameter("image_topic").as_string();
    save_interval_ = get_parameter("save_interval").as_int();
    image_dir_ = get_parameter("image_dir").as_string();
    timestamp_file_ = get_parameter("timestamp_file").as_string();
    image_prefix_ = get_parameter("image_prefix").as_string();

    RCLCPP_INFO(this->get_logger(),
      "ImageSaver initialized: topic='%s', interval=%d, output='%s'",
      image_topic_.c_str(), save_interval_, image_dir_.c_str());
    RCLCPP_INFO(this->get_logger(),
      "Timestamp CSV will be saved to '%s'", timestamp_file_.c_str());

    // PNG compression level 0 = uncompressed
    png_params_.push_back(cv::IMWRITE_PNG_COMPRESSION);
    png_params_.push_back(0);

    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      image_topic_, 10,
      std::bind(&ImageSaver::image_callback, this, std::placeholders::_1));
  }

  ~ImageSaver()
  {
    if (timestamps_.empty())
    {
      RCLCPP_INFO(this->get_logger(), "No images were saved. Skipping CSV export.");
      return;
    }

    std::string csv_path = timestamp_file_;
    std::ofstream csv(csv_path);
    if (!csv.is_open())
    {
      RCLCPP_ERROR(this->get_logger(), "Failed to open '%s' for writing.", csv_path.c_str());
      return;
    }

    csv << "index,filename,timestamp_sec,timestamp_nanosec\n";
    for (const auto & entry : timestamps_)
    {
      csv << entry.index << ","
          << entry.filename << ","
          << entry.stamp_sec << ","
          << entry.stamp_nanosec << "\n";
    }
    csv.close();

    RCLCPP_INFO(this->get_logger(),
      "Saved %zu timestamps to '%s'", timestamps_.size(), csv_path.c_str());
  }

private:
  struct TimestampEntry
  {
    int index;
    std::string filename;
    int64_t stamp_sec;
    uint32_t stamp_nanosec;
  };

  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
  {
    frame_count_++;

    if (frame_count_ % save_interval_ != 0)
      return;

    // Convert ROS image to OpenCV
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvShare(msg);
    }
    catch (const cv_bridge::Exception & e)
    {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }

    // Build filename with zero-padded index
    int save_index = static_cast<int>(timestamps_.size());
    char name_buf[64];
    std::snprintf(name_buf, sizeof(name_buf), "%s_%06d.png", image_prefix_.c_str(), save_index);
    std::string filename(name_buf);
    std::string filepath = image_dir_ + "/" + filename;

    // Save as uncompressed PNG
    cv::Mat save_image;
    if (msg->encoding == "rgb8")
      cv::cvtColor(cv_ptr->image, save_image, cv::COLOR_RGB2BGR);
    else
      save_image = cv_ptr->image;

    // Then use save_image instead of cv_ptr->image in imwrite:
    if (!cv::imwrite(filepath, save_image, png_params_))
    {
      RCLCPP_ERROR(this->get_logger(), "Failed to save image to '%s'", filepath.c_str());
      return;
    }

    // Record timestamp
    timestamps_.push_back({
      save_index,
      filename,
      static_cast<int64_t>(msg->header.stamp.sec),
      msg->header.stamp.nanosec
    });

    //RCLCPP_INFO(this->get_logger(), "Saved '%s' (frame %d)", filename.c_str(), frame_count_);
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;

  std::string image_topic_;
  int save_interval_;
  std::string image_dir_;
  std::string timestamp_file_;
  std::string image_prefix_;

  int frame_count_ = 0;
  std::vector<int> png_params_;
  std::vector<TimestampEntry> timestamps_;
};

RCLCPP_COMPONENTS_REGISTER_NODE(liga_splat::ImageSaver)
} // namespace liga_splat