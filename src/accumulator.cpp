#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/pcl_base.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include "pcl_conversions/pcl_conversions.h"
#include <rclcpp_components/register_node_macro.hpp>

namespace liga_splat {
class PointCloudAccumulator : public rclcpp::Node
{
public:
    PointCloudAccumulator(const rclcpp::NodeOptions & options)
    : Node("point_cloud_accumulator", options)
    {
        declare_parameter<std::string>("input_topic",   "/isaacsim/lidar");
        declare_parameter<std::string>("output_topic",  "/liga_splat/accumulated_point_cloud");
        declare_parameter<std::string>("frame_id",      "World");
        declare_parameter<int>("max_points",      500000); // downsample + publish when cloud exceeds this
        declare_parameter<int>("publish_interval", 100);   // also publish every N frames (0 = disabled)
        declare_parameter<double>("leaf_size",     0.05);  // voxel leaf size used before publishing

        in_topic_         = get_parameter("input_topic").as_string();
        out_topic_        = get_parameter("output_topic").as_string();
        frame_id_         = get_parameter("frame_id").as_string();
        max_points_       = get_parameter("max_points").as_int();
        publish_interval_ = get_parameter("publish_interval").as_int();
        leaf_size_        = static_cast<float>(get_parameter("leaf_size").as_double());

        RCLCPP_INFO(this->get_logger(),
            "PointCloudAccumulator: in='%s', out='%s', max_points=%d, publish_interval=%d, leaf_size=%.3f",
            in_topic_.c_str(), out_topic_.c_str(), max_points_, publish_interval_, leaf_size_);

        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            in_topic_, 10,
            std::bind(&PointCloudAccumulator::point_cloud_callback, this, std::placeholders::_1));
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(out_topic_, 10);
    }

private:
    void publish_and_clear(const rclcpp::Time & stamp)
    {
        sensor_msgs::msg::PointCloud2 out_msg;
        pcl::toROSMsg(*output_cloud_, out_msg);
        out_msg.header.frame_id = frame_id_;
        out_msg.header.stamp    = stamp;
        publisher_->publish(out_msg);

        RCLCPP_INFO(this->get_logger(),
            "Published %zu points to GlobalProcessor. Clearing accumulator.",
            output_cloud_->size());

        output_cloud_->clear();
        frame_count_ = 0;
    }

    void point_cloud_callback(const sensor_msgs::msg::PointCloud2 & msg)
    {
        pcl::fromROSMsg(msg, *buffer_cloud_);
        *output_cloud_ += *buffer_cloud_;
        frame_count_++;

        bool size_exceeded = static_cast<int>(output_cloud_->size()) >= max_points_;
        bool interval_hit  = (publish_interval_ > 0) && (frame_count_ % publish_interval_ == 0);

        if (size_exceeded)
        {
            // Downsample in place before publishing to keep message size manageable
            auto t0 = std::chrono::steady_clock::now();
            vg_.setInputCloud(output_cloud_);
            vg_.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
            vg_.filter(*output_cloud_);
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

            RCLCPP_INFO(this->get_logger(),
                "Size threshold reached. After voxel filter: %zu points (%.2fs).",
                output_cloud_->size(), elapsed);

            publish_and_clear(msg.header.stamp);
        }
        else if (interval_hit)
        {
            // Periodic publish so the GlobalProcessor always receives recent data,
            // even if the size threshold is never reached (e.g. short sessions).
            publish_and_clear(msg.header.stamp);
        }
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;

    std::string in_topic_;
    std::string out_topic_;
    std::string frame_id_;
    int   max_points_;
    int   publish_interval_;
    float leaf_size_;
    int   frame_count_ = 0;

    typedef pcl::PointXYZ PointT_;
    pcl::PointCloud<PointT_>::Ptr output_cloud_ {new pcl::PointCloud<PointT_>()};
    pcl::PointCloud<PointT_>::Ptr buffer_cloud_  {new pcl::PointCloud<PointT_>()};
    pcl::VoxelGrid<PointT_> vg_;
};

RCLCPP_COMPONENTS_REGISTER_NODE(liga_splat::PointCloudAccumulator)
} // namespace liga_splat
