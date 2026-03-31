#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/pcl_base.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include "pcl_conversions/pcl_conversions.h"
#include <rclcpp_components/register_node_macro.hpp>

namespace airlab_lidar_3dgs {
class PointCloudAccumulator : public rclcpp::Node
{
  public:
    PointCloudAccumulator(const rclcpp::NodeOptions & options)
    : Node("point_cloud_accumulator", options)
    {
      // Load ROS2 parameters
      declare_parameter<std::string>("input_topic",  "/isaacsim/lidar");
      declare_parameter<std::string>("output_topic", "/airlab_lidar_3dgs/accumulated_point_cloud");
      declare_parameter<std::string>("frame_id", "World");
      in_topic_ = get_parameter("input_topic").as_string();
      out_topic_ = get_parameter("output_topic").as_string();
      frame_id_ = get_parameter("frame_id").as_string();

      RCLCPP_INFO(this->get_logger(), "Initialized PointCloudAccumulator with input topic: '%s', output topic: '%s', frame id: '%s'", 
      in_topic_.c_str(), out_topic_.c_str(), frame_id_.c_str());

      subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      in_topic_, 10, std::bind(&PointCloudAccumulator::point_cloud_callback, this, std::placeholders::_1));
      publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(out_topic_, 10);
    }

  private:
    void point_cloud_callback(const sensor_msgs::msg::PointCloud2 & msg)
    { 
      //Convert pointcloud2 topic into pcl point cloud
      pcl::fromROSMsg(msg, *buffer_cloud_);
      
      // Transform the point cloud to the world frame if necessary

      // Concatenate the new point cloud with the existing one
      *output_cloud_ += *buffer_cloud_;
      if (output_cloud_->size() > 1000000)
      {
        RCLCPP_INFO(this->get_logger(), "Accumulated point cloud size exceeded 1 million points. Publishing and clearing.");
        sensor_msgs::msg::PointCloud2 out_msg;
        pcl::toROSMsg(*output_cloud_, out_msg);
        out_msg.header.frame_id = frame_id_;
        out_msg.header.stamp = msg.header.stamp;
        publisher_->publish(out_msg);
        output_cloud_->clear();
      }
    }
    // ROS2 publisher and subscriber
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;

    // ROS2 parameters
    std::string in_topic_;
    std::string out_topic_;
    std::string frame_id_;

    // Initialize empty point clouds
    typedef pcl::PointXYZ PointT_; //Define the point cloud point type
    pcl::PointCloud<PointT_>::Ptr output_cloud_ {new pcl::PointCloud<PointT_>()};
    pcl::PointCloud<PointT_>::Ptr buffer_cloud_ {new pcl::PointCloud<PointT_>()};
};

RCLCPP_COMPONENTS_REGISTER_NODE(airlab_lidar_3dgs::PointCloudAccumulator)
} // namespace airlab_lidar_3dgs