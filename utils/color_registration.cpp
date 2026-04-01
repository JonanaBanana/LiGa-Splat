#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <filesystem>
#include <chrono>

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <vtkObject.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// ========================= Configuration =========================
struct Config
{
    // Camera intrinsics
    double f = 1000.0;
    int img_w = 1920;
    int img_h = 1080;
    double px = 960.0;
    double py = 540.0;

    // Filtering
    double voxel_size = 0.03;
    double min_depth = 1.0;
    double max_depth = 400.0;
    bool filter_outliers = true;
    int sor_neighbors = 10;
    double sor_std_ratio = 2.0;

    // Hidden point removal
    double hpr_radius = 100000.0;

    // Background sphere
    bool fill_background = true;
    double sphere_radius = 200.0;
    int sphere_num_pts = 50000;

    // Camera-to-body transform
    Eigen::Matrix4d trans_mat = Eigen::Matrix4d::Identity();

    // Paths (relative to data_folder)
    std::string pcd_file = "pcd/global_point_cloud.pcd";
    std::string poses_file = "poses/image_poses.csv";
    std::string images_dir = "images";
    std::string output_file = "point_cloud_color_information.csv";
    std::string downsampled_file = "pcd/downsampled_point_cloud.pcd";
};

// ========================= Config Parser =========================

Config load_config(const std::string & path)
{
    Config cfg;
    YAML::Node node;

    try {
        node = YAML::LoadFile(path);
    } catch (const YAML::Exception & e) {
        std::cerr << "Error reading config: " << e.what() << std::endl;
        cfg.trans_mat << 0,0,1,0, -1,0,0,0, 0,-1,0,0, 0,0,0,1;
        return cfg;
    }

    cfg.f     = node["focal_length"].as<double>(cfg.f);
    cfg.img_w = node["image_width"].as<int>(cfg.img_w);
    cfg.img_h = node["image_height"].as<int>(cfg.img_h);
    cfg.px    = node["principal_x"].as<double>(cfg.px);
    cfg.py    = node["principal_y"].as<double>(cfg.py);

    cfg.voxel_size      = node["voxel_size"].as<double>(cfg.voxel_size);
    cfg.min_depth       = node["min_depth"].as<double>(cfg.min_depth);
    cfg.max_depth       = node["max_depth"].as<double>(cfg.max_depth);
    cfg.filter_outliers = node["filter_outliers"].as<bool>(cfg.filter_outliers);
    cfg.sor_neighbors   = node["sor_neighbors"].as<int>(cfg.sor_neighbors);
    cfg.sor_std_ratio   = node["sor_std_ratio"].as<double>(cfg.sor_std_ratio);

    cfg.hpr_radius = node["hpr_radius"].as<double>(cfg.hpr_radius);

    cfg.fill_background = node["fill_background"].as<bool>(cfg.fill_background);
    cfg.sphere_radius   = node["sphere_radius"].as<double>(cfg.sphere_radius);
    cfg.sphere_num_pts  = node["sphere_num_points"].as<int>(cfg.sphere_num_pts);

    if (node["trans_mat"])
    {
        auto vals = node["trans_mat"].as<std::vector<double>>();
        if (vals.size() == 16)
            cfg.trans_mat = Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(vals.data());
    }
    else
    {
        cfg.trans_mat << 0,0,1,0, -1,0,0,0, 0,-1,0,0, 0,0,0,1;
    }

    cfg.pcd_file         = node["pcd_file"].as<std::string>(cfg.pcd_file);
    cfg.poses_file       = node["poses_file"].as<std::string>(cfg.poses_file);
    cfg.images_dir       = node["images_dir"].as<std::string>(cfg.images_dir);
    cfg.output_file      = node["output_file"].as<std::string>(cfg.output_file);
    cfg.downsampled_file = node["downsampled_file"].as<std::string>(cfg.downsampled_file);

    return cfg;
}

void print_config(const Config & cfg)
{
    std::cout << "=== Configuration ===" << std::endl;
    std::cout << "  Focal length:    " << cfg.f << std::endl;
    std::cout << "  Resolution:      " << cfg.img_w << "x" << cfg.img_h << std::endl;
    std::cout << "  Principal point:  (" << cfg.px << ", " << cfg.py << ")" << std::endl;
    std::cout << "  Voxel size:      " << cfg.voxel_size << std::endl;
    std::cout << "  Depth range:     [" << cfg.min_depth << ", " << cfg.max_depth << "]" << std::endl;
    std::cout << "  Filter outliers: " << (cfg.filter_outliers ? "yes" : "no") << std::endl;
    std::cout << "  HPR radius:      " << cfg.hpr_radius << std::endl;
    std::cout << "  Background:      " << (cfg.fill_background ? "yes" : "no") << std::endl;
    std::cout << "  PCD file:        " << cfg.pcd_file << std::endl;
    std::cout << "  Poses file:      " << cfg.poses_file << std::endl;
    std::cout << "  Images dir:      " << cfg.images_dir << std::endl;
    std::cout << "  Output file:     " << cfg.output_file << std::endl;
    std::cout << "=====================" << std::endl;
}

// ========================= Data Structures =========================
struct ImagePose
{
    int index;
    std::string filename;
    double timestamp;
    double px, py, pz;
    double qx, qy, qz, qw;
};

struct ColorObservation
{
    int point_id;
    float r, g, b;
};

// ========================= Helpers =========================

Eigen::Matrix4d quat_to_matrix(double px, double py, double pz,
                                double qx, double qy, double qz, double qw)
{
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    Eigen::Quaterniond q(qw, qx, qy, qz);
    q.normalize();
    T.block<3, 3>(0, 0) = q.toRotationMatrix();
    T(0, 3) = px;
    T(1, 3) = py;
    T(2, 3) = pz;
    return T;
}

std::vector<ImagePose> read_image_poses(const std::string & path)
{
    std::vector<ImagePose> poses;
    std::ifstream file(path);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open pose file '" << path << "'" << std::endl;
        return poses;
    }

    std::string line;
    std::getline(file, line); // skip header

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string token;
        ImagePose p;

        std::getline(ss, token, ','); p.index = std::stoi(token);
        std::getline(ss, token, ','); p.filename = token;
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

void generate_sphere_points(Eigen::MatrixXd & points, int start_row,
                            const Eigen::Vector3d & center,
                            double radius, int num_pts)
{
    for (int i = 0; i < num_pts; i++)
    {
        double idx = static_cast<double>(i) + 0.5;
        double phi = std::acos(1.0 - 2.0 * idx / num_pts);
        double theta = M_PI * (1.0 + std::sqrt(5.0)) * idx;

        int row = start_row + i;
        points(row, 0) = std::cos(theta) * std::sin(phi) * radius + center(0);
        points(row, 1) = std::sin(theta) * std::sin(phi) * radius + center(1);
        points(row, 2) = std::cos(phi) * radius + center(2);
    }
}

// OPT 7: Scalar math instead of Eigen::Vector3d per point
std::vector<int> hidden_point_removal(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,
    const Eigen::Vector3d & viewpoint,
    double radius)
{
    int n = static_cast<int>(cloud->size());

    pcl::PointCloud<pcl::PointXYZ>::Ptr flipped(new pcl::PointCloud<pcl::PointXYZ>());
    flipped->resize(n + 1);

    const double vx = viewpoint(0), vy = viewpoint(1), vz = viewpoint(2);

    for (int i = 0; i < n; i++)
    {
        const double px = cloud->points[i].x;
        const double py = cloud->points[i].y;
        const double pz = cloud->points[i].z;

        const double dx = px - vx;
        const double dy = py - vy;
        const double dz = pz - vz;
        const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

        if (dist < 1e-10)
        {
            (*flipped)[i] = cloud->points[i];
            continue;
        }

        const double scale = 2.0 * (radius - dist) / dist;
        (*flipped)[i].x = px + scale * dx;
        (*flipped)[i].y = py + scale * dy;
        (*flipped)[i].z = pz + scale * dz;
    }

    (*flipped)[n].x = vx;
    (*flipped)[n].y = vy;
    (*flipped)[n].z = vz;

    pcl::ConvexHull<pcl::PointXYZ> hull;
    hull.setInputCloud(flipped);
    hull.setDimension(3);

    pcl::PointCloud<pcl::PointXYZ> hull_cloud;
    hull.reconstruct(hull_cloud);

    pcl::PointIndices hull_point_indices;
    hull.getHullPointIndices(hull_point_indices);

    std::vector<int> visible;
    visible.reserve(hull_point_indices.indices.size());
    for (int idx : hull_point_indices.indices)
    {
        if (idx >= 0 && idx < n)
            visible.push_back(idx);
    }

    return visible;
}

// ========================= Main =========================
int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <data_folder> <config_file> [--viz]" << std::endl;
        return 1;
    }

    std::string data_folder = argv[1];
    if (data_folder.back() != '/') data_folder += '/';

    std::string config_path = argv[2];

    bool visualize = false;
    for (int i = 3; i < argc; i++)
    {
        if (std::string(argv[i]) == "--viz") visualize = true;
    }

    Config cfg = load_config(config_path);
    print_config(cfg);

    std::string pcd_path = data_folder + cfg.pcd_file;
    std::string poses_path = data_folder + cfg.poses_file;
    std::string images_dir = data_folder + cfg.images_dir + "/";
    std::string output_path = data_folder + cfg.output_file;
    std::string downsampled_path = data_folder + cfg.downsampled_file;

    // OPT 4: Precompute FOV tangent thresholds
    double fov_x = 2.0 * std::atan2(cfg.img_w, 2.0 * cfg.f);
    double fov_y = 2.0 * std::atan2(cfg.img_h, 2.0 * cfg.f);
    double tan_half_fov_x = std::tan(fov_x * 0.5);
    double tan_half_fov_y = std::tan(fov_y * 0.5);

    double margin_x = cfg.img_w * 0.01;
    double margin_y = cfg.img_h * 0.01;
    double max_proj_x = cfg.img_w - margin_x;
    double max_proj_y = cfg.img_h - margin_y;

    // ---- Load and preprocess point cloud ----
    auto t_start = std::chrono::high_resolution_clock::now();

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_path, *cloud) == -1)
    {
        std::cerr << "Error: Could not load PCD file '" << pcd_path << "'" << std::endl;
        return 1;
    }
    std::cout << "Loaded " << cloud->size() << " points from " << pcd_path << std::endl;

    if (cfg.filter_outliers)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr clean(new pcl::PointCloud<pcl::PointXYZ>());
        for (const auto & pt : cloud->points)
        {
            if (std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z))
                clean->push_back(pt);
        }

        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(clean);
        sor.setMeanK(cfg.sor_neighbors);
        sor.setStddevMulThresh(cfg.sor_std_ratio);
        sor.filter(*clean);
        std::cout << "After outlier removal: " << clean->size() << " points" << std::endl;

        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(clean);
        vg.setLeafSize(cfg.voxel_size, cfg.voxel_size, cfg.voxel_size);
        vg.filter(*cloud);
        std::cout << "After voxel downsample: " << cloud->size() << " points" << std::endl;
    }

    int num_original = static_cast<int>(cloud->size());
    int num_sphere = cfg.fill_background ? cfg.sphere_num_pts : 0;
    int total_points = num_original + num_sphere;

    // OPT 3: Store points as [4 x N] column-major so T * points is a single
    // matrix multiply with no transposes
    Eigen::Matrix<double, 4, Eigen::Dynamic> points_h(4, total_points);
    for (int i = 0; i < num_original; i++)
    {
        points_h(0, i) = cloud->points[i].x;
        points_h(1, i) = cloud->points[i].y;
        points_h(2, i) = cloud->points[i].z;
        points_h(3, i) = 1.0;
    }

    if (cfg.fill_background)
    {
        // Generate sphere points into a temporary [N x 3] block
        Eigen::MatrixXd sphere_pts(num_sphere, 3);
        Eigen::MatrixXd temp_for_gen(total_points, 3);
        for (int i = 0; i < num_original; i++)
        {
            temp_for_gen(i, 0) = points_h(0, i);
            temp_for_gen(i, 1) = points_h(1, i);
            temp_for_gen(i, 2) = points_h(2, i);
        }

        Eigen::Vector3d centroid = temp_for_gen.topRows(num_original).colwise().mean();
        std::cout << "Adding " << num_sphere << " background sphere points around centroid ["
                  << centroid.transpose() << "]" << std::endl;
        generate_sphere_points(temp_for_gen, num_original, centroid, cfg.sphere_radius, num_sphere);

        for (int i = num_original; i < total_points; i++)
        {
            points_h(0, i) = temp_for_gen(i, 0);
            points_h(1, i) = temp_for_gen(i, 1);
            points_h(2, i) = temp_for_gen(i, 2);
            points_h(3, i) = 1.0;
        }
    }

    // OPT 8: Save as binary PCD
    {
        pcl::PointCloud<pcl::PointXYZ> save_cloud;
        save_cloud.resize(total_points);
        for (int i = 0; i < total_points; i++)
        {
            save_cloud[i].x = points_h(0, i);
            save_cloud[i].y = points_h(1, i);
            save_cloud[i].z = points_h(2, i);
        }
        pcl::io::savePCDFileBinary(downsampled_path, save_cloud);
        std::cout << "Saved downsampled point cloud to " << downsampled_path << std::endl;
    }

    auto image_poses = read_image_poses(poses_path);
    if (image_poses.empty())
    {
        std::cerr << "No image poses loaded." << std::endl;
        return 1;
    }
    std::cout << "Loaded " << image_poses.size() << " image poses" << std::endl;

    // ---- Visualization state ----
    Eigen::MatrixXf point_colors = Eigen::MatrixXf::Constant(total_points, 3, 0.1f);
    pcl::PointCloud<pcl::PointXYZRGB> viz_cloud;
    bool viewer_initialized = false;

    // ---- Process each image ----
    std::vector<ColorObservation> all_observations;
    all_observations.reserve(total_points * 2);

    // OPT 3: Pre-allocate camera-frame result as [4 x N] — reused each iteration
    Eigen::Matrix<double, 4, Eigen::Dynamic> cam_points(4, total_points);

    // OPT 6: Pre-allocate buffers, reuse each frame via clear()/resize()
    std::vector<int> fov_indices;
    fov_indices.reserve(total_points / 4);

    pcl::PointCloud<pcl::PointXYZ>::Ptr visible_cloud(new pcl::PointCloud<pcl::PointXYZ>());

    for (size_t img_idx = 0; img_idx < image_poses.size(); img_idx++)
    {
        const auto & pose = image_poses[img_idx];
        auto t_img_start = std::chrono::high_resolution_clock::now();

        // Load image
        std::string img_file = images_dir + pose.filename;
        cv::Mat img = cv::imread(img_file, cv::IMREAD_COLOR);
        if (img.empty())
        {
            std::cerr << "Warning: Could not load image '" << img_file << "', skipping." << std::endl;
            continue;
        }
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        // Build transform
        Eigen::Matrix4d T_world_body = quat_to_matrix(
            pose.px, pose.py, pose.pz,
            pose.qx, pose.qy, pose.qz, pose.qw);

        Eigen::Matrix4d T_world_cam = T_world_body * cfg.trans_mat;
        Eigen::Matrix4d T_cam_world = T_world_cam.inverse();

        // OPT 3: Single matrix multiply, no transposes
        // cam_points [4 x N] = T_cam_world [4 x 4] * points_h [4 x N]
        cam_points.noalias() = T_cam_world * points_h;

        // OPT 4 & 5: FOV filtering with tangent comparison, cache results in cam_points
        fov_indices.clear();

        for (int i = 0; i < total_points; i++)
        {
            double cz = cam_points(2, i);
            if (cz < cfg.min_depth || cz > cfg.max_depth) continue;

            double cx = cam_points(0, i);
            double cy = cam_points(1, i);

            // OPT 4: |cx| < cz * tan(fov/2) is equivalent to |atan2(cx,cz)| < fov/2
            if (std::abs(cx) < cz * tan_half_fov_x && std::abs(cy) < cz * tan_half_fov_y)
                fov_indices.push_back(i);
        }

        if (fov_indices.empty()) continue;

        // ---- Hidden point removal ----
        // OPT 6: Reuse visible_cloud allocation
        visible_cloud->points.resize(fov_indices.size());
        visible_cloud->width = fov_indices.size();
        visible_cloud->height = 1;
        for (size_t j = 0; j < fov_indices.size(); j++)
        {
            int idx = fov_indices[j];
            visible_cloud->points[j].x = cam_points(0, idx);
            visible_cloud->points[j].y = cam_points(1, idx);
            visible_cloud->points[j].z = cam_points(2, idx);
        }

        Eigen::Vector3d camera_origin(0.0, 0.0, 0.0);
        std::vector<int> hpr_indices = hidden_point_removal(
            visible_cloud, camera_origin, cfg.hpr_radius);

        if (hpr_indices.empty()) continue;

        // ---- Project to image plane ----
        int found = 0;
        for (int hi : hpr_indices)
        {
            if (hi < 0 || hi >= static_cast<int>(fov_indices.size())) continue;
            int orig_idx = fov_indices[hi];

            // OPT 5: Reuse already-transformed camera coordinates
            double cx = cam_points(0, orig_idx);
            double cy = cam_points(1, orig_idx);
            double cz = cam_points(2, orig_idx);

            double u = cfg.f * cx / cz + cfg.px;
            double v = cfg.f * cy / cz + cfg.py;

            if (u < margin_x || u >= max_proj_x || v < margin_y || v >= max_proj_y)
                continue;

            int iu = std::clamp(static_cast<int>(std::round(u)), 0, cfg.img_w - 1);
            int iv = std::clamp(static_cast<int>(std::round(v)), 0, cfg.img_h - 1);

            const cv::Vec3b & pixel = img.at<cv::Vec3b>(iv, iu);
            float r = pixel[0] / 255.0f;
            float g = pixel[1] / 255.0f;
            float b = pixel[2] / 255.0f;

            all_observations.push_back({orig_idx, r, g, b});

            if (visualize)
            {
                point_colors(orig_idx, 0) = r;
                point_colors(orig_idx, 1) = g;
                point_colors(orig_idx, 2) = b;
            }

            found++;
        }

        // ---- Update visualization cloud ----
        if (visualize)
        {
            if (!viewer_initialized)
            {
                vtkObject::GlobalWarningDisplayOff();
                viz_cloud.resize(total_points);
                for (int i = 0; i < total_points; i++)
                {
                    viz_cloud[i].x = points_h(0, i);
                    viz_cloud[i].y = points_h(1, i);
                    viz_cloud[i].z = points_h(2, i);
                    viz_cloud[i].r = 25;
                    viz_cloud[i].g = 25;
                    viz_cloud[i].b = 25;
                }
                viewer_initialized = true;
            }

            for (int i = 0; i < total_points; i++)
            {
                viz_cloud[i].r = static_cast<uint8_t>(point_colors(i, 0) * 255);
                viz_cloud[i].g = static_cast<uint8_t>(point_colors(i, 1) * 255);
                viz_cloud[i].b = static_cast<uint8_t>(point_colors(i, 2) * 255);
            }
        }

        auto t_img_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t_img_end - t_img_start).count();

        std::cout << img_idx << "/" << image_poses.size() - 1
                  << " — " << pose.filename
                  << ": " << found << " points colored"
                  << " (" << std::fixed << std::setprecision(2) << elapsed << "s)"
                  << std::endl;
    }

    // ---- Save output CSV ----
    std::ofstream out(output_path);
    if (!out.is_open())
    {
        std::cerr << "Error: Could not open output file '" << output_path << "'" << std::endl;
        return 1;
    }

    out << std::fixed << std::setprecision(8);
    for (const auto & obs : all_observations)
        out << obs.point_id << "," << obs.r << "," << obs.g << "," << obs.b << "\n";

    out.close();

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\nDone! Saved " << all_observations.size()
              << " color observations to '" << output_path << "'"
              << " in " << std::fixed << std::setprecision(1) << total_elapsed << "s"
              << std::endl;

    // ---- Show example point cloud ----
    // ---- This is not final point cloud, just contains last color definitions for each point ----
    // ---- the final colored point cloud will be constructed in 
    if (visualize && viewer_initialized)
    {
        std::cout << "Opening viewer. Close the window to exit." << std::endl;
        pcl::visualization::PCLVisualizer viewer("Color Registration");
        viewer.setBackgroundColor(0.05, 0.05, 0.05);
        viewer.addCoordinateSystem(0.5);

        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> handler(viz_cloud.makeShared());
        viewer.addPointCloud(viz_cloud.makeShared(), handler, "cloud");
        viewer.setPointCloudRenderingProperties(
            pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");

        viewer.spin();
    }
    return 0;
}