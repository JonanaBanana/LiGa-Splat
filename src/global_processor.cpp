#include <chrono>
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

      // Load ROS2 parameters
      declare_parameter<std::string>("input_topic",  "/airlab_lidar_3dgs/accumulated_point_cloud");
      declare_parameter<std::string>("output_topic", "/airlab_lidar_3dgs/global_point_cloud");
      declare_parameter<float>("leaf_size", 0.05);
      declare_parameter<std::string>("output_location", "/home/airlab/dataset/global_point_cloud.pcd");
      declare_parameter<std::string>("frame_id", "World");
      in_topic_ = get_parameter("input_topic").as_string();
      out_topic_ = get_parameter("output_topic").as_string();
      leaf_size_ = get_parameter("leaf_size").as_double();
      output_location_ = get_parameter("output_location").as_string();
      frame_id_ = get_parameter("frame_id").as_string();

      RCLCPP_INFO(this->get_logger(), "Global Processor initialized with input topic: '%s', output topic: '%s', leaf size: '%.2f', output location: '%s', frame id: '%s'", 
      in_topic_.c_str(), out_topic_.c_str(), leaf_size_, output_location_.c_str(), frame_id_.c_str());

      subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      in_topic_, 10, std::bind(&GlobalProcessor::point_cloud_callback, this, std::placeholders::_1));
      
      publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(out_topic_, 10);
    }
    ~GlobalProcessor()
    {
      if (output_cloud_ && !output_cloud_->empty())
      {
        RCLCPP_INFO(this->get_logger(), "Shutting down. Downsampling and saving %zu global points to '%s'",
                    output_cloud_->size(), output_location_.c_str());
        sor_.setInputCloud(output_cloud_);
        sor_.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
        sor_.filter(*output_cloud_);
        RCLCPP_INFO(this->get_logger(), "Downsampling complete. Remaining points: '%d'. Saving to file...", (int)output_cloud_->size());
        pcl::io::savePCDFileASCII(output_location_, *output_cloud_);
      }
    }

  private:
    void point_cloud_callback(const sensor_msgs::msg::PointCloud2 & msg)
    { 
        count_++;
        pcl::fromROSMsg(msg, *buffer_cloud_);
        sor_.setInputCloud(buffer_cloud_);
        sor_.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
        time_t start = clock();       
        sor_.filter(*buffer_cloud_);
        time_t end = clock();
        double elapsed = double(end - start) / CLOCKS_PER_SEC;
        RCLCPP_INFO(this->get_logger(), "Voxel grid filter applied. Original points: '%d', Filtered points: '%d', Time taken: '%.3f' seconds.", 
        (int)msg.height * (int)msg.width, (int)buffer_cloud_->size(), elapsed);
        *output_cloud_ += *buffer_cloud_;
        //RCLCPP_INFO(this->get_logger(), "Global point cloud now has '%d' points.", (int)output_cloud_->size());
        if (count_ > 5){
            RCLCPP_INFO(this->get_logger(), "Processed 10 point clouds. Downsampling global point cloud...");
            sor_.setInputCloud(output_cloud_);
            sor_.setLeafSize(0.05f, 0.05f, 0.05f);
            time_t start = clock();
            sor_.filter(*output_cloud_);
            time_t end = clock();
            double elapsed = double(end - start) / CLOCKS_PER_SEC;
            RCLCPP_INFO(this->get_logger(), "Global downsampling complete. Remaining points: '%d', Time taken: '%.3f' seconds.", (int)output_cloud_->size(), elapsed);
            pcl::io::savePCDFileASCII (output_location_, *output_cloud_);
            RCLCPP_INFO(this->get_logger(), "Saved Output to file at: '%s'", output_location_.c_str());
            count_ = 0;
        }

        sensor_msgs::msg::PointCloud2 out_msg;
        pcl::toROSMsg(*output_cloud_, out_msg);
        out_msg.header.frame_id = frame_id_;
        out_msg.header.stamp = msg.header.stamp;
        publisher_->publish(out_msg);
    }
    //ROS2 publisher and subscriber
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    //ROS2 parameters
    std::string in_topic_;
    std::string out_topic_;
    std::string output_location_;
    std::string frame_id_;
    double leaf_size_;

    // Initialize empty point cloud cloud
    typedef pcl::PointXYZ PointT_; //Define the point cloud point type
    pcl::PointCloud<PointT_>::Ptr output_cloud_ {new pcl::PointCloud<PointT_>()};
    pcl::PointCloud<PointT_>::Ptr buffer_cloud_ {new pcl::PointCloud<PointT_>()};

    //voxel grid filter
    pcl::VoxelGrid<PointT_> sor_;

    // counter to track how many point clouds have been processed before downsampling the global cloud
    int count_ = 0;



};

RCLCPP_COMPONENTS_REGISTER_NODE(airlab_lidar_3dgs::GlobalProcessor)
} // namespace airlab_lidar_3dgs