#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>
#include <vtkObject.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

// Map a normalized value [0,1] to an RGB color using a turbo-like colormap
void rainbow_colormap(float t, uint8_t &r, uint8_t &g, uint8_t &b)
{
    t = std::clamp(t, 0.0f, 1.0f);

    // HSV with hue from 270 (blue) down to 0 (red) as height increases
    float hue = (1.0f - t) * 270.0f;
    float s = 1.0f, v = 1.0f;

    float c = v * s;
    float x = c * (1.0f - std::fabs(std::fmod(hue / 60.0f, 2.0f) - 1.0f));
    float m = v - c;

    float rf, gf, bf;
    if      (hue < 60)  { rf = c; gf = x; bf = 0; }
    else if (hue < 120) { rf = x; gf = c; bf = 0; }
    else if (hue < 180) { rf = 0; gf = c; bf = x; }
    else if (hue < 240) { rf = 0; gf = x; bf = c; }
    else                { rf = x; gf = 0; bf = c; }

    r = static_cast<uint8_t>((rf + m) * 255.0f);
    g = static_cast<uint8_t>((gf + m) * 255.0f);
    b = static_cast<uint8_t>((bf + m) * 255.0f);
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <pcd_file> [color_mode]" << std::endl;
        std::cerr << "  color_mode: rgb (default), z" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    std::string color_mode = (argc >= 3) ? argv[2] : "rgb";

    if (color_mode != "rgb" && color_mode != "z")
    {
        std::cerr << "Unknown color mode '" << color_mode << "'. Use: rgb, or z," << std::endl;
        return 1;
    }

    // Load the cloud as XYZ
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *cloud) == -1)
    {
        std::cerr << "\033[31m" << "Error: Could not load file '" << filename << "'" << "\033[0m" << std::endl;
        return 1;
    }

    std::cout << "Loaded " << cloud->size() << " points from " << filename << std::endl;
    std::cout << "Color mode: " << color_mode << std::endl;

    // Build a colored cloud
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    colored_cloud->resize(cloud->size());

    if (color_mode == "z")
    {
        // Find Z range
        float z_min = std::numeric_limits<float>::max();
        float z_max = std::numeric_limits<float>::lowest();
        for (const auto & pt : cloud->points)
        {
            if (std::isfinite(pt.z))
            {
                z_min = std::min(z_min, pt.z);
                z_max = std::max(z_max, pt.z);
            }
        }

        float z_range = z_max - z_min;
        if (z_range < 1e-6f) z_range = 1.0f;

        std::cout << "Z range: [" << z_min << ", " << z_max << "]" << std::endl;

        for (size_t i = 0; i < cloud->size(); i++)
        {
            const auto & pt = cloud->points[i];
            auto & cpt = colored_cloud->points[i];
            cpt.x = pt.x; cpt.y = pt.y; cpt.z = pt.z;

            float t = (pt.z - z_min) / z_range;
            rainbow_colormap(t, cpt.r, cpt.g, cpt.b);
        }
    }
    else // rgb — load native colors if present, otherwise flat color
    {
        // Load into PCLPointCloud2 first to inspect field names.
        // A typed load (PointXYZ) silently drops any fields not in the type,
        // so we must check the raw header before deciding which loader to use.
        pcl::PCLPointCloud2 header;
        pcl::io::loadPCDFile(filename, header);

        bool has_rgb = false;
        for (const auto & field : header.fields)
        {
            if (field.name == "rgb" || field.name == "rgba")
            {
                has_rgb = true;
                break;
            }
        }

        if (has_rgb)
        {
            pcl::io::loadPCDFile<pcl::PointXYZRGB>(filename, *colored_cloud);
            std::cout << "Using native RGB colors." << std::endl;
        }
        else
        {
            for (size_t i = 0; i < cloud->size(); i++)
            {
                auto & cpt = colored_cloud->points[i];
                cpt.x = cloud->points[i].x;
                cpt.y = cloud->points[i].y;
                cpt.z = cloud->points[i].z;
                cpt.r = 50; cpt.g = 200; cpt.b = 50;
            }
            std::cout << "No RGB data in file. Using default green." << std::endl;
        }
    }

    // Suppress VTK deprecation warnings
    vtkObject::GlobalWarningDisplayOff();

    // Set up viewer
    pcl::visualization::PCLVisualizer viewer("PCD Viewer");
    viewer.setBackgroundColor(0.05, 0.05, 0.05);
    viewer.addCoordinateSystem(1.0);
    viewer.initCameraParameters();

    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_handler(colored_cloud);
    viewer.addPointCloud<pcl::PointXYZRGB>(colored_cloud, rgb_handler, "cloud");
    viewer.setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");

    std::cout << "\nControls:" << std::endl;
    std::cout << "  Mouse drag   - Rotate" << std::endl;
    std::cout << "  Scroll       - Zoom" << std::endl;
    std::cout << "  Shift+drag   - Pan" << std::endl;
    std::cout << "  r            - Reset camera" << std::endl;
    std::cout << "  +/-          - Increase/decrease point size" << std::endl;
    std::cout << "  q            - Quit" << std::endl;

    viewer.setCameraPosition(-10, 0, 3, 15, 0, 3, 0, 0, 1);

    viewer.spin();

    return 0;
}