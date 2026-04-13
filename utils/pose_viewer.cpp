#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/PolygonMesh.h>
#include <pcl/conversions.h>
#include <vtkObject.h>

struct Pose
{
    double timestamp;
    double px, py, pz;
    double qx, qy, qz, qw;
};

double combine_timestamp(int64_t sec, uint32_t nsec)
{
    return static_cast<double>(sec) + static_cast<double>(nsec) * 1e-9;
}

void rainbow_colormap(float t, uint8_t &r, uint8_t &g, uint8_t &b)
{
    t = std::clamp(t, 0.0f, 1.0f);

    float hue = (1.0f - t) * 270.0f;
    float c = 1.0f;
    float x = c * (1.0f - std::fabs(std::fmod(hue / 60.0f, 2.0f) - 1.0f));

    float rf, gf, bf;
    if      (hue < 60)  { rf = c; gf = x; bf = 0; }
    else if (hue < 120) { rf = x; gf = c; bf = 0; }
    else if (hue < 180) { rf = 0; gf = c; bf = x; }
    else if (hue < 240) { rf = 0; gf = x; bf = c; }
    else                { rf = x; gf = 0; bf = c; }

    r = static_cast<uint8_t>(rf * 255.0f);
    g = static_cast<uint8_t>(gf * 255.0f);
    b = static_cast<uint8_t>(bf * 255.0f);
}

std::vector<Pose> read_odometry_csv(const std::string & path)
{
    std::vector<Pose> poses;
    std::ifstream file(path);
    if (!file.is_open()) return poses;

    std::string line;
    std::getline(file, line); // skip header

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string token;
        Pose p;
        int64_t sec;
        uint32_t nsec;

        std::getline(ss, token, ','); // index
        std::getline(ss, token, ','); sec = std::stoll(token);
        std::getline(ss, token, ','); nsec = std::stoul(token);
        std::getline(ss, token, ','); p.px = std::stod(token);
        std::getline(ss, token, ','); p.py = std::stod(token);
        std::getline(ss, token, ','); p.pz = std::stod(token);
        std::getline(ss, token, ','); p.qx = std::stod(token);
        std::getline(ss, token, ','); p.qy = std::stod(token);
        std::getline(ss, token, ','); p.qz = std::stod(token);
        std::getline(ss, token, ','); p.qw = std::stod(token);

        p.timestamp = combine_timestamp(sec, nsec);
        poses.push_back(p);
    }
    return poses;
}

std::vector<Pose> read_image_poses_csv(const std::string & path)
{
    std::vector<Pose> poses;
    std::ifstream file(path);
    if (!file.is_open()) return poses;

    std::string line;
    std::getline(file, line); // skip header

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string token;
        Pose p;

        std::getline(ss, token, ','); // index
        std::getline(ss, token, ','); // filename
        std::getline(ss, token, ','); p.timestamp = std::stod(token);
        std::getline(ss, token, ','); p.px = std::stod(token);
        std::getline(ss, token, ','); p.py = std::stod(token);
        std::getline(ss, token, ','); p.pz = std::stod(token);
        std::getline(ss, token, ','); p.qx = std::stod(token);
        std::getline(ss, token, ','); p.qy = std::stod(token);
        std::getline(ss, token, ','); p.qz = std::stod(token);
        std::getline(ss, token, ','); p.qw = std::stod(token);

        poses.push_back(p);
    }
    return poses;
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <data_folder> [--no-pcd] [--frustum-scale <s>]" << std::endl;
        std::cerr << "Reads:" << std::endl;
        std::cerr << "  <data_folder>/timestamps/odom.csv" << std::endl;
        std::cerr << "  <data_folder>/poses.csv" << std::endl;
        std::cerr << "  <data_folder>/pcd/input.pcd" << std::endl;
        return 1;
    }

    std::string data_folder = argv[1];
    if (data_folder.back() != '/') data_folder += '/';

    bool show_pcd = true;
    double frustum_scale = 0.3;
    for (int i = 2; i < argc; i++)
    {
        if (std::string(argv[i]) == "--no-pcd") show_pcd = false;
        else if (std::string(argv[i]) == "--frustum-scale" && i + 1 < argc)
            frustum_scale = std::stod(argv[++i]);
    }

    std::string odom_path = data_folder + "timestamps/odom.csv";
    std::string poses_path = data_folder + "poses.csv";
    std::string pcd_path = data_folder + "pcd/input.pcd";

    // ---- Load data ----
    auto odom = read_odometry_csv(odom_path);
    auto cam_poses = read_image_poses_csv(poses_path);

    std::cout << "Loaded " << odom.size() << " odometry poses" << std::endl;
    std::cout << "Loaded " << cam_poses.size() << " camera poses" << std::endl;

    if (odom.empty() && cam_poses.empty())
    {
        std::cerr << "No data to visualize." << std::endl;
        return 1;
    }

    // ---- Set up viewer ----
    vtkObject::GlobalWarningDisplayOff();
    pcl::visualization::PCLVisualizer viewer("Pose Viewer");
    viewer.setBackgroundColor(0.05, 0.05, 0.05);
    viewer.addCoordinateSystem(1.0);

    // ---- Point cloud with Z-axis rainbow coloring ----
    if (show_pcd)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_path, *cloud) == 0 && !cloud->empty())
        {
            std::cout << "Loaded " << cloud->size() << " points from PCD" << std::endl;

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored(new pcl::PointCloud<pcl::PointXYZRGB>());
            colored->resize(cloud->size());

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
            float z_range = (z_max - z_min > 1e-6f) ? z_max - z_min : 1.0f;

            for (size_t i = 0; i < cloud->size(); i++)
            {
                auto & cpt = colored->points[i];
                cpt.x = cloud->points[i].x;
                cpt.y = cloud->points[i].y;
                cpt.z = cloud->points[i].z;
                rainbow_colormap((cpt.z - z_min) / z_range, cpt.r, cpt.g, cpt.b);
            }

            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler(colored);
            viewer.addPointCloud(colored, handler, "cloud");
            viewer.setPointCloudRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");
        }
        else
        {
            std::cerr << "Warning: Could not load PCD file, skipping." << std::endl;
        }
    }

    // ---- Odometry path ----
    if (!odom.empty())
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr odom_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        for (const auto & p : odom)
        {
            pcl::PointXYZRGB pt;
            pt.x = p.px; pt.y = p.py; pt.z = p.pz;
            pt.r = 80; pt.g = 80; pt.b = 200;
            odom_cloud->push_back(pt);
        }
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler(odom_cloud);
        viewer.addPointCloud(odom_cloud, handler, "odom_points");
        viewer.setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "odom_points");

        // Polyline
        pcl::PointCloud<pcl::PointXYZ>::Ptr path_pts(new pcl::PointCloud<pcl::PointXYZ>());
        for (const auto & p : odom)
            path_pts->push_back(pcl::PointXYZ(p.px, p.py, p.pz));

        pcl::PolygonMesh path_mesh;
        pcl::toPCLPointCloud2(*path_pts, path_mesh.cloud);
        pcl::Vertices strip;
        for (uint32_t i = 0; i < path_pts->size(); i++)
            strip.vertices.push_back(i);
        path_mesh.polygons.push_back(strip);
        viewer.addPolylineFromPolygonMesh(path_mesh, "odom_path");
        viewer.setShapeRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_COLOR, 0.3, 0.3, 0.8, "odom_path");
    }

    // ---- Camera frustums ----
    if (!cam_poses.empty())
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr frustum_pts(new pcl::PointCloud<pcl::PointXYZ>());
        std::vector<pcl::Vertices> lines;

        double aspect = 16.0 / 9.0;

        for (const auto & p : cam_poses)
        {
            double qx = p.qx, qy = p.qy, qz = p.qz, qw = p.qw;

            // Extract world-space axes from the quaternion rotation matrix columns.
            // These are the unit vectors pointing forward, right, and up in world space.
            // Forward (X column of rotation matrix)
            double fx = 1.0 - 2.0 * (qy * qy + qz * qz);
            double fy = 2.0 * (qx * qy + qw * qz);
            double fz = 2.0 * (qx * qz - qw * qy);

            // Right (negative Y column)
            double rx = -2.0 * (qx * qy - qw * qz);
            double ry = -(1.0 - 2.0 * (qx * qx + qz * qz));
            double rz = -2.0 * (qy * qz + qw * qx);

            // Up (Z column)
            double ux = 2.0 * (qx * qz + qw * qy);
            double uy = 2.0 * (qy * qz - qw * qx);
            double uz = 1.0 - 2.0 * (qx * qx + qy * qy);

            // hw/hh: half-width and half-height of the far plane rectangle.
            // d: depth at which the far plane is placed (equals frustum_scale).
            double hw = frustum_scale * aspect * 0.5;
            double hh = frustum_scale * 0.5;
            double d = frustum_scale;

            uint32_t base = frustum_pts->size();

            // Point layout per frustum (base offset):
            //   0 = camera origin (apex)
            //   1 = far top-left,  2 = far top-right
            //   3 = far bot-left,  4 = far bot-right
            frustum_pts->push_back(pcl::PointXYZ(p.px, p.py, p.pz));
            frustum_pts->push_back(pcl::PointXYZ(
                p.px + d*fx - hw*rx + hh*ux,
                p.py + d*fy - hw*ry + hh*uy,
                p.pz + d*fz - hw*rz + hh*uz));
            frustum_pts->push_back(pcl::PointXYZ(
                p.px + d*fx + hw*rx + hh*ux,
                p.py + d*fy + hw*ry + hh*uy,
                p.pz + d*fz + hw*rz + hh*uz));
            frustum_pts->push_back(pcl::PointXYZ(
                p.px + d*fx - hw*rx - hh*ux,
                p.py + d*fy - hw*ry - hh*uy,
                p.pz + d*fz - hw*rz - hh*uz));
            frustum_pts->push_back(pcl::PointXYZ(
                p.px + d*fx + hw*rx - hh*ux,
                p.py + d*fy + hw*ry - hh*uy,
                p.pz + d*fz + hw*rz - hh*uz));

            auto add_line = [&](uint32_t a, uint32_t b) {
                pcl::Vertices v;
                v.vertices.push_back(base + a);
                v.vertices.push_back(base + b);
                lines.push_back(v);
            };

            // 4 rays from apex to corners, then 4 edges closing the far rectangle
            add_line(0, 1); add_line(0, 2); add_line(0, 3); add_line(0, 4);
            add_line(1, 2); add_line(2, 4); add_line(4, 3); add_line(3, 1);
        }

        pcl::PolygonMesh mesh;
        pcl::toPCLPointCloud2(*frustum_pts, mesh.cloud);
        mesh.polygons = lines;
        viewer.addPolylineFromPolygonMesh(mesh, "frustums");
        viewer.setShapeRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.3, 0.3, "frustums");
    }

    std::cout << "\nVisualization:" << std::endl;
    std::cout << "  Rainbow cloud  = point cloud (Z-axis colored)" << std::endl;
    std::cout << "  Blue path      = odometry" << std::endl;
    std::cout << "  Red frustums   = camera poses" << std::endl;

    viewer.spin();

    return 0;
}