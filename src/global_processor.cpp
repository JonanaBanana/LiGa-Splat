#include <chrono>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/pcl_base.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include "pcl_conversions/pcl_conversions.h"
#include <rclcpp_components/register_node_macro.hpp>

namespace airlab_lidar_3dgs {
class GlobalProcessor : public rclcpp::Node
{
public:
    GlobalProcessor(const rclcpp::NodeOptions & options)
    : Node("global_processor", options)
    {
        const char* home_env = std::getenv("HOME");
        std::string home = home_env ? home_env : "/tmp";

        declare_parameter<std::string>("input_topic",    "/airlab_lidar_3dgs/accumulated_point_cloud");
        declare_parameter<std::string>("output_topic",   "/airlab_lidar_3dgs/global_point_cloud");
        declare_parameter<std::string>("frame_id",       "World");
        declare_parameter<std::string>("output_location", home + "/dataset/airlab_3dgs/pcd/input.pcd");
        declare_parameter<float>("leaf_size",            0.05);
        declare_parameter<int>("downsample_interval",    5);       // downsample global cloud every N batches
        declare_parameter<int>("max_global_points",      5000000); // hard cap — triggers aggressive downsample

        in_topic_           = get_parameter("input_topic").as_string();
        out_topic_          = get_parameter("output_topic").as_string();
        frame_id_           = get_parameter("frame_id").as_string();
        output_location_    = get_parameter("output_location").as_string();
        leaf_size_          = get_parameter("leaf_size").as_double();
        downsample_interval_= get_parameter("downsample_interval").as_int();
        max_global_points_  = get_parameter("max_global_points").as_int();

        RCLCPP_INFO(this->get_logger(),
            "GlobalProcessor: in='%s', out='%s', leaf=%.3f, ds_interval=%d, max_pts=%d, save='%s'",
            in_topic_.c_str(), out_topic_.c_str(), leaf_size_,
            downsample_interval_, max_global_points_, output_location_.c_str());

        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            in_topic_, 10,
            std::bind(&GlobalProcessor::point_cloud_callback, this, std::placeholders::_1));
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(out_topic_, 10);
    }

    ~GlobalProcessor()
    {
        if (output_cloud_ && !output_cloud_->empty())
        {
            RCLCPP_INFO(this->get_logger(),
                "Shutting down. Downsampling %zu global points and saving to '%s'.",
                output_cloud_->size(), output_location_.c_str());

            sor_.setInputCloud(output_cloud_);
            sor_.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
            sor_.filter(*output_cloud_);

            pcl::io::savePCDFileBinary(output_location_, *output_cloud_);

            RCLCPP_INFO(this->get_logger(),
                "Saved %zu points to '%s'.", output_cloud_->size(), output_location_.c_str());
        }
    }

private:
    void downsample_global(float leaf)
    {
        auto t0 = std::chrono::steady_clock::now();
        sor_.setInputCloud(output_cloud_);
        sor_.setLeafSize(leaf, leaf, leaf);
        sor_.filter(*output_cloud_);
        double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        RCLCPP_INFO(this->get_logger(),
            "Global downsample (leaf=%.3f): %zu points remaining (%.2fs).",
            leaf, output_cloud_->size(), elapsed);
    }

    void point_cloud_callback(const sensor_msgs::msg::PointCloud2 & msg)
    {
        count_++;

        // Voxel-filter the incoming batch before adding to global map
        pcl::fromROSMsg(msg, *buffer_cloud_);
        {
            auto t0 = std::chrono::steady_clock::now();
            sor_.setInputCloud(buffer_cloud_);
            sor_.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
            sor_.filter(*buffer_cloud_);
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            RCLCPP_INFO(this->get_logger(),
                "Batch %d filtered: %d -> %zu points (%.2fs).",
                count_, msg.height * msg.width, buffer_cloud_->size(), elapsed);
        }

        *output_cloud_ += *buffer_cloud_;

        // Periodic downsampling of global cloud to keep memory bounded
        if (count_ % downsample_interval_ == 0)
        {
            RCLCPP_INFO(this->get_logger(),
                "Periodic downsample (every %d batches). Global size before: %zu.",
                downsample_interval_, output_cloud_->size());
            downsample_global(leaf_size_);
            pcl::io::savePCDFileBinary(output_location_, *output_cloud_);
            RCLCPP_INFO(this->get_logger(), "Saved to '%s'.", output_location_.c_str());
        }

        // Hard cap: if still too large, apply a more aggressive pass to stay within budget
        if (static_cast<int>(output_cloud_->size()) > max_global_points_)
        {
            float aggressive_leaf = leaf_size_ * 2.0f;
            RCLCPP_WARN(this->get_logger(),
                "Global cloud (%zu pts) exceeds max_global_points (%d). "
                "Applying aggressive downsample (leaf=%.3f).",
                output_cloud_->size(), max_global_points_, aggressive_leaf);
            downsample_global(aggressive_leaf);
        }

        // Publish current global cloud for visualisation
        sensor_msgs::msg::PointCloud2 out_msg;
        pcl::toROSMsg(*output_cloud_, out_msg);
        out_msg.header.frame_id = frame_id_;
        out_msg.header.stamp    = msg.header.stamp;
        publisher_->publish(out_msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;

    std::string in_topic_;
    std::string out_topic_;
    std::string output_location_;
    std::string frame_id_;
    double leaf_size_;
    int downsample_interval_;
    int max_global_points_;
    int count_ = 0;

    typedef pcl::PointXYZ PointT_;
    pcl::PointCloud<PointT_>::Ptr output_cloud_ {new pcl::PointCloud<PointT_>()};
    pcl::PointCloud<PointT_>::Ptr buffer_cloud_  {new pcl::PointCloud<PointT_>()};
    pcl::VoxelGrid<PointT_> sor_;
};

RCLCPP_COMPONENTS_REGISTER_NODE(airlab_lidar_3dgs::GlobalProcessor)
} // namespace airlab_lidar_3dgs
