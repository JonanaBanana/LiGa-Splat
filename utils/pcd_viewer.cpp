#include <iostream>
#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <path_to_pcd_file>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];

    // Try loading as XYZRGB first, fall back to XYZ
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_rgb(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>());

    pcl::PCLPointCloud2 header;
    pcl::io::loadPCDFile(filename, header);

    bool has_rgb = false;
    for (const auto& field : header.fields)
    {
        if (field.name == "rgb" || field.name == "rgba")
        {
            has_rgb = true;
            break;
        }
    }

    if (has_rgb)
    {
        if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(filename, *cloud_rgb) == -1)
        {
            std::cerr << "Error: Could not load file '" << filename << "'" << std::endl;
            return 1;
        }
        std::cout << "Loaded " << cloud_rgb->size() << " points (XYZRGB) from " << filename << std::endl;
    }
    else
    {
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud_xyz) == -1)
        {
            std::cerr << "Error: Could not load file '" << filename << "'" << std::endl;
            return 1;
        }
        std::cout << "Loaded " << cloud_xyz->size() << " points (XYZ) from " << filename << std::endl;
    }

    // Set up the visualizer
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("PCD Viewer"));
    viewer->setBackgroundColor(0.05, 0.05, 0.05);
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    if (has_rgb)
    {
        std::cout << "Loaded " << cloud_rgb->size() << " points (XYZRGB) from " << filename << std::endl;
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_handler(cloud_rgb);
        viewer->addPointCloud<pcl::PointXYZRGB>(cloud_rgb, rgb_handler, "cloud");
    }
    else
    {
        std::cout << "Loaded " << cloud_xyz->size() << " points (XYZ) from " << filename << std::endl;
        viewer->addPointCloud<pcl::PointXYZ>(cloud_xyz, "cloud");
        viewer->setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_COLOR, 0.2, 0.8, 0.2, "cloud");
    }

    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");

    std::cout << "\nControls:" << std::endl;
    std::cout << "  Mouse drag   - Rotate" << std::endl;
    std::cout << "  Scroll       - Zoom" << std::endl;
    std::cout << "  Shift+drag   - Pan" << std::endl;
    std::cout << "  r            - Reset camera" << std::endl;
    std::cout << "  +/-          - Increase/decrease point size" << std::endl;
    std::cout << "  q            - Quit" << std::endl;
    
    viewer->setCameraPosition(-10, 0, 3, 15, 0, 3, 0, 0, 1);

    viewer->spin();

    return 0;
}