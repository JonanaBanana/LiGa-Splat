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
#include <unordered_map>
#include <unordered_set>

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/surface/convex_hull.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <omp.h>

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
    // Empirically good range: 15000–30000 for outdoor scenes with max_depth ~200 m.
    double hpr_radius = 20000.0;

    // Edge point preservation
    bool preserve_edge_points = true;
    double edge_canny_low    = 50.0;
    double edge_canny_high   = 150.0;
    int    edge_dilation_px  = 2;     // dilate edge mask by N pixels before matching
    double edge_voxel_size   = 0.01;  // dedup resolution for edge points (metres); keep small

    // Background sphere
    bool fill_background = true;
    double sphere_radius = 200.0;
    int sphere_num_pts = 50000;

    // Camera-to-body transform (T_body_cam)
    // Set to identity if poses are already in camera frame
    bool poses_are_body_frame = true;
    Eigen::Matrix4d trans_mat = Eigen::Matrix4d::Identity();

    // Paths (relative to data_folder)
    std::string pcd_file = "pcd/input.pcd";
    std::string poses_file = "poses.csv";
    std::string images_dir = "images";
    std::string registration_file = "color_registration.csv";
    std::string downsampled_file = "pcd/downsampled.pcd";
};

// ========================= Config Parser =========================

Config load_config(const std::string & path)
{
    Config cfg;
    YAML::Node node;

    try {
        node = YAML::LoadFile(path);
    } catch (const YAML::Exception & e) {
        std::cerr << "\033[31m" << "Error reading config: " << e.what() << "\033[0m" << std::endl;
        std::cerr << "\033[31m" << "Using default config values." << "\033[0m" << std::endl;
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

    cfg.preserve_edge_points = node["preserve_edge_points"].as<bool>(cfg.preserve_edge_points);
    cfg.edge_canny_low       = node["edge_canny_low"].as<double>(cfg.edge_canny_low);
    cfg.edge_canny_high      = node["edge_canny_high"].as<double>(cfg.edge_canny_high);
    cfg.edge_dilation_px     = node["edge_dilation_px"].as<int>(cfg.edge_dilation_px);
    cfg.edge_voxel_size      = node["edge_voxel_size"].as<double>(cfg.edge_voxel_size);

    cfg.fill_background       = node["fill_background"].as<bool>(cfg.fill_background);
    cfg.sphere_radius         = node["sphere_radius"].as<double>(cfg.sphere_radius);
    cfg.sphere_num_pts        = node["sphere_num_points"].as<int>(cfg.sphere_num_pts);
    cfg.poses_are_body_frame  = node["poses_are_body_frame"].as<bool>(cfg.poses_are_body_frame);

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

    cfg.pcd_file          = node["pcd_file"].as<std::string>(cfg.pcd_file);
    cfg.downsampled_file  = node["downsampled_file"].as<std::string>(cfg.downsampled_file);
    cfg.poses_file        = node["poses_file"].as<std::string>(cfg.poses_file);
    cfg.images_dir        = node["images_dir"].as<std::string>(cfg.images_dir);
    cfg.registration_file = node["registration_file"].as<std::string>(cfg.registration_file);

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
    std::cout << "  Preserve edges:  " << (cfg.preserve_edge_points ? "yes" : "no");
    if (cfg.preserve_edge_points)
        std::cout << "  (Canny [" << cfg.edge_canny_low << ", " << cfg.edge_canny_high
                  << "], dilation=" << cfg.edge_dilation_px << "px"
                  << ", edge voxel=" << cfg.edge_voxel_size << "m)";
    std::cout << std::endl;
    std::cout << "  Background:      " << (cfg.fill_background ? "yes" : "no") << std::endl;
    std::cout << "  PCD file:        " << cfg.pcd_file << std::endl;
    std::cout << "  Poses file:      " << cfg.poses_file << std::endl;
    std::cout << "  Images dir:      " << cfg.images_dir << std::endl;
    std::cout << "  Output file:     " << cfg.registration_file << std::endl;
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

struct ProjPoint
{
    int iu, iv;
    float depth;
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
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <data_folder> [--diag]" << std::endl;
        return 1;
    }
    bool diagnostics = false;
    for (int i = 2; i < argc; i++)
    {
        if (std::string(argv[i]) == "--diag") diagnostics = true;
    }

    std::string data_folder = argv[1];
    if (data_folder.back() != '/') data_folder += '/';

    std::string config_path = data_folder + "config.cfg";

    Config cfg = load_config(config_path);
    print_config(cfg);

    std::string pcd_path = data_folder + cfg.pcd_file;
    std::string poses_path = data_folder + cfg.poses_file;
    std::string images_dir = data_folder + cfg.images_dir + "/";
    std::string output_path = data_folder + cfg.registration_file;
    std::string downsampled_path = data_folder + cfg.downsampled_file;

    // Precompute FOV tangent thresholds
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
        std::cerr << "\033[31m" << "Error: Could not load PCD file '" << pcd_path << "'" << "\033[0m" << std::endl;
        return 1;
    }
    std::cout << "Loaded " << cloud->size() << " points from " << pcd_path << std::endl;

    // full_cloud holds the post-SOR, pre-voxel cloud so we can recover edge points
    // that were discarded by downsampling.  Only populated when preserve_edge_points
    // is true and filter_outliers is true (i.e. voxel downsampling actually ran).
    pcl::PointCloud<pcl::PointXYZ>::Ptr full_cloud(new pcl::PointCloud<pcl::PointXYZ>());

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

        if (cfg.preserve_edge_points)
            *full_cloud = *clean; // snapshot before voxel grid removes points

        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(clean);
        vg.setLeafSize(cfg.voxel_size, cfg.voxel_size, cfg.voxel_size);
        vg.filter(*cloud);
        std::cout << "After voxel downsample: " << cloud->size() << " points" << std::endl;
    }

    int num_original = static_cast<int>(cloud->size());
    int num_sphere = cfg.fill_background ? cfg.sphere_num_pts : 0;
    int total_points = num_original + num_sphere;

    // Store points as [4 x N] column-major so T * points is a single
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
        // A sparse background sphere is added around the scene centroid so that
        // 3DGS has surface support behind the captured area. Without it, the
        // Gaussian splatting optimiser tends to place large "background blobs"
        // at arbitrary distances, which degrades foreground quality.
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

    auto image_poses = read_image_poses(poses_path);
    if (image_poses.empty())
    {
        std::cerr << "No image poses loaded." << std::endl;
        return 1;
    }
    std::cout << "Loaded " << image_poses.size() << " image poses" << std::endl;

    // =========================================================
    // Pass 1: Edge detection — collect full_cloud indices only.
    // No colour sampling here; edge points will receive colour
    // via the regular registration pass after they are added
    // to the cloud, giving them stable multi-view median colour.
    // =========================================================

    // per_image_edge_indices[i] = full_cloud indices visible on edge pixels in image i
    std::vector<std::vector<int>> per_image_edge_indices(image_poses.size());
    // per_image_edge_diag[i]    = projected pixel coords (for diagnostics only)
    std::vector<std::vector<ProjPoint>> per_image_edge_diag(image_poses.size());

    if (cfg.preserve_edge_points && !full_cloud->empty())
    {
        #pragma omp parallel for schedule(dynamic)
        for (size_t img_idx = 0; img_idx < image_poses.size(); img_idx++)
        {
            const auto & pose = image_poses[img_idx];

            std::string img_file = images_dir + pose.filename;
            cv::Mat img = cv::imread(img_file, cv::IMREAD_GRAYSCALE);
            if (img.empty()) continue;

            Eigen::Matrix4d T_world_body = quat_to_matrix(
                pose.px, pose.py, pose.pz,
                pose.qx, pose.qy, pose.qz, pose.qw);
            Eigen::Matrix4d T_cam_world = (cfg.poses_are_body_frame
                                          ? T_world_body * cfg.trans_mat
                                          : T_world_body).inverse();

            // Canny + dilation
            cv::Mat edge_mask;
            cv::Canny(img, edge_mask, cfg.edge_canny_low, cfg.edge_canny_high);
            if (cfg.edge_dilation_px > 0)
            {
                int ks = 2 * cfg.edge_dilation_px + 1;
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {ks, ks});
                cv::dilate(edge_mask, edge_mask, kernel);
            }

            if (diagnostics)
            {
                std::string diag_dir = data_folder + "diagnostics/edge_detection/";
                std::filesystem::create_directories(diag_dir);
                cv::imwrite(diag_dir + pose.filename, edge_mask);
            }

            // FOV-filter full_cloud in camera space
            int num_full = static_cast<int>(full_cloud->size());
            std::vector<int> full_fov_indices;
            full_fov_indices.reserve(num_full / 4);
            pcl::PointCloud<pcl::PointXYZ>::Ptr full_fov_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            full_fov_cloud->reserve(num_full / 4);

            for (int i = 0; i < num_full; i++)
            {
                const auto & pt = full_cloud->points[i];
                double cx = T_cam_world(0,0)*pt.x + T_cam_world(0,1)*pt.y + T_cam_world(0,2)*pt.z + T_cam_world(0,3);
                double cy = T_cam_world(1,0)*pt.x + T_cam_world(1,1)*pt.y + T_cam_world(1,2)*pt.z + T_cam_world(1,3);
                double cz = T_cam_world(2,0)*pt.x + T_cam_world(2,1)*pt.y + T_cam_world(2,2)*pt.z + T_cam_world(2,3);

                if (cz < cfg.min_depth || cz > cfg.max_depth) continue;
                if (std::abs(cx) >= cz * tan_half_fov_x || std::abs(cy) >= cz * tan_half_fov_y) continue;

                full_fov_indices.push_back(i);
                full_fov_cloud->push_back({static_cast<float>(cx),
                                           static_cast<float>(cy),
                                           static_cast<float>(cz)});
            }

            if (full_fov_indices.empty()) continue;

            // HPR — mirrors colour registration to avoid adding occluded points
            Eigen::Vector3d edge_origin(0.0, 0.0, 0.0);
            std::vector<int> full_hpr = hidden_point_removal(
                full_fov_cloud, edge_origin, cfg.hpr_radius);

            // Project surviving points; record those on edge pixels
            for (int hi : full_hpr)
            {
                if (hi < 0 || hi >= static_cast<int>(full_fov_indices.size())) continue;
                int orig_i = full_fov_indices[hi];

                const auto & cp = full_fov_cloud->points[hi];
                double cz = cp.z, cx = cp.x, cy = cp.y;

                double u = cfg.f * cx / cz + cfg.px;
                double v = cfg.f * cy / cz + cfg.py;
                if (u < margin_x || u >= max_proj_x || v < margin_y || v >= max_proj_y) continue;

                int iu = std::clamp(static_cast<int>(std::round(u)), 0, cfg.img_w - 1);
                int iv = std::clamp(static_cast<int>(std::round(v)), 0, cfg.img_h - 1);
                if (edge_mask.at<uint8_t>(iv, iu) == 0) continue;

                per_image_edge_indices[img_idx].push_back(orig_i);
                if (diagnostics)
                    per_image_edge_diag[img_idx].push_back({iu, iv, static_cast<float>(cz)});
            }

        } // end Pass 1 parallel loop
    }

    // =========================================================
    // Between passes: deduplicate edge candidates and expand
    // points_h so Pass 2 colour-registers the new points too.
    // =========================================================
    {
        const double inv_vs = 1.0 / cfg.edge_voxel_size;
        auto make_key = [&](double x, double y, double z) -> int64_t {
            int64_t ix = static_cast<int64_t>(std::floor(x * inv_vs));
            int64_t iy = static_cast<int64_t>(std::floor(y * inv_vs));
            int64_t iz = static_cast<int64_t>(std::floor(z * inv_vs));
            return (ix & 0x1FFFFFLL) | ((iy & 0x1FFFFFLL) << 21) | ((iz & 0x1FFFFFLL) << 42);
        };

        // Collect unique full_cloud indices across all images
        std::unordered_set<int> seen_fc_idx;
        std::vector<int> unique_fc_indices;
        for (const auto & idx_vec : per_image_edge_indices)
            for (int fc_idx : idx_vec)
                if (seen_fc_idx.insert(fc_idx).second)
                    unique_fc_indices.push_back(fc_idx);

        // Deduplicate by edge_voxel_size grid (one edge point per voxel)
        std::unordered_set<int64_t> occupied;
        std::vector<pcl::PointXYZ> new_pts;
        for (int fc_idx : unique_fc_indices)
        {
            const auto & pt = full_cloud->points[fc_idx];
            int64_t key = make_key(pt.x, pt.y, pt.z);
            if (!occupied.insert(key).second) continue;
            new_pts.push_back(pt);
        }

        if (!new_pts.empty())
        {
            // Expand points_h to accommodate new edge points
            int old_total = total_points;
            int extra = static_cast<int>(new_pts.size());
            Eigen::Matrix<double, 4, Eigen::Dynamic> expanded(4, old_total + extra);
            expanded.leftCols(old_total) = points_h;
            for (int i = 0; i < extra; i++)
            {
                expanded(0, old_total + i) = new_pts[i].x;
                expanded(1, old_total + i) = new_pts[i].y;
                expanded(2, old_total + i) = new_pts[i].z;
                expanded(3, old_total + i) = 1.0;
            }
            points_h = std::move(expanded);
            total_points += extra;

            std::cout << "Edge preservation: added " << extra
                      << " points → total " << total_points << std::endl;
        }
    }

    // Save downsampled.pcd (includes edge points)
    {
        pcl::PointCloud<pcl::PointXYZ> save_cloud;
        save_cloud.resize(total_points);
        for (int i = 0; i < total_points; i++)
        {
            save_cloud[i].x = static_cast<float>(points_h(0, i));
            save_cloud[i].y = static_cast<float>(points_h(1, i));
            save_cloud[i].z = static_cast<float>(points_h(2, i));
        }
        pcl::io::savePCDFileBinary(downsampled_path, save_cloud);
        std::cout << "Saved downsampled point cloud (" << total_points
                  << " points) to " << downsampled_path << std::endl;
    }

    // =========================================================
    // Pass 2: Colour registration on the full expanded cloud.
    // Edge points are treated identically to downsampled points;
    // they receive stable multi-view median colour.
    // =========================================================
    std::vector<std::vector<ColorObservation>> per_image_obs(image_poses.size());

    #pragma omp parallel for schedule(dynamic)
    for (size_t img_idx = 0; img_idx < image_poses.size(); img_idx++)
    {
        const auto & pose = image_poses[img_idx];
        auto t_img_start = std::chrono::high_resolution_clock::now();

        // Thread-local working buffers
        Eigen::Matrix<double, 4, Eigen::Dynamic> cam_points(4, total_points);
        std::vector<int> fov_indices;
        fov_indices.reserve(total_points / 4);
        pcl::PointCloud<pcl::PointXYZ>::Ptr visible_cloud(new pcl::PointCloud<pcl::PointXYZ>());

        // Load image
        std::string img_file = images_dir + pose.filename;
        cv::Mat img = cv::imread(img_file, cv::IMREAD_COLOR);
        if (img.empty())
        {
            std::ostringstream msg;
            msg << "Warning: Could not load image '" << img_file << "', skipping.\n";
            #pragma omp critical
            std::cerr << msg.str();
            continue;
        }
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        // Build transform
        Eigen::Matrix4d T_world_body = quat_to_matrix(
            pose.px, pose.py, pose.pz,
            pose.qx, pose.qy, pose.qz, pose.qw);

        Eigen::Matrix4d T_cam_world = (cfg.poses_are_body_frame
                                      ? T_world_body * cfg.trans_mat
                                      : T_world_body).inverse();

        // cam_points [4 x N] = T_cam_world [4 x 4] * points_h [4 x N]
        cam_points.noalias() = T_cam_world * points_h;

        // FOV filtering
        fov_indices.clear();

        for (int i = 0; i < total_points; i++)
        {
            double cz = cam_points(2, i);
            if (cz < cfg.min_depth || cz > cfg.max_depth) continue;

            double cx = cam_points(0, i);
            double cy = cam_points(1, i);

            // Precompute FOV tangent thresholds
            if (std::abs(cx) < cz * tan_half_fov_x && std::abs(cy) < cz * tan_half_fov_y)
                fov_indices.push_back(i);
        }

        if (fov_indices.empty()) continue;

        // ---- Hidden point removal ----
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
        std::vector<ProjPoint> diag_points;

        for (int hi : hpr_indices)
        {
            if (hi < 0 || hi >= static_cast<int>(fov_indices.size())) continue;
            int orig_idx = fov_indices[hi];

            double cx = cam_points(0, orig_idx);
            double cy = cam_points(1, orig_idx);
            double cz = cam_points(2, orig_idx);

            double u = cfg.f * cx / cz + cfg.px;
            double v = cfg.f * cy / cz + cfg.py;

            if (u < margin_x || u >= max_proj_x || v < margin_y || v >= max_proj_y)
                continue;
            // Bilinear interpolation
            int x0 = static_cast<int>(std::floor(u));
            int y0 = static_cast<int>(std::floor(v));
            int x1 = std::min(x0 + 1, cfg.img_w - 1);
            int y1 = std::min(y0 + 1, cfg.img_h - 1);
            x0 = std::max(x0, 0);
            y0 = std::max(y0, 0);

            float fx = static_cast<float>(u - std::floor(u));
            float fy = static_cast<float>(v - std::floor(v));
            float w00 = (1.0f - fx) * (1.0f - fy);
            float w10 = fx           * (1.0f - fy);
            float w01 = (1.0f - fx) * fy;
            float w11 = fx           * fy;

            const auto & p00 = img.at<cv::Vec3b>(y0, x0);
            const auto & p10 = img.at<cv::Vec3b>(y0, x1);
            const auto & p01 = img.at<cv::Vec3b>(y1, x0);
            const auto & p11 = img.at<cv::Vec3b>(y1, x1);

            float r = (w00 * p00[0] + w10 * p10[0] + w01 * p01[0] + w11 * p11[0]) / 255.0f;
            float g = (w00 * p00[1] + w10 * p10[1] + w01 * p01[1] + w11 * p11[1]) / 255.0f;
            float b = (w00 * p00[2] + w10 * p10[2] + w01 * p01[2] + w11 * p11[2]) / 255.0f;

            per_image_obs[img_idx].push_back({orig_idx, r, g, b});

            if (diagnostics)
            {
                diag_points.push_back({x0, y0, static_cast<float>(cz)});
            }

            found++;
        }
        // ---- Diagnostic: registered_points overlay (regular + edge points) ----
        if (diagnostics && !diag_points.empty())
        {
            cv::Mat verify = cv::imread(images_dir + pose.filename);
            if (!verify.empty())
            {
                double d_min = std::numeric_limits<double>::max();
                double d_max = 0.0;
                for (const auto & dp : diag_points)
                {
                    d_min = std::min(d_min, static_cast<double>(dp.depth));
                    d_max = std::max(d_max, static_cast<double>(dp.depth));
                }
                double d_range = (d_max - d_min > 1e-6) ? d_max - d_min : 1.0;

                cv::Mat overlay = verify.clone();
                for (const auto & dp : diag_points)
                {
                    float t = static_cast<float>((dp.depth - d_min) / d_range);
                    float hue = (1.0f - t) * 270.0f;
                    float c = 1.0f;
                    float x = c * (1.0f - std::fabs(std::fmod(hue / 60.0f, 2.0f) - 1.0f));
                    float rf, gf, bf;
                    if      (hue < 60)  { rf = c; gf = x; bf = 0; }
                    else if (hue < 120) { rf = x; gf = c; bf = 0; }
                    else if (hue < 180) { rf = 0; gf = c; bf = x; }
                    else if (hue < 240) { rf = 0; gf = x; bf = c; }
                    else                { rf = x; gf = 0; bf = c; }
                    cv::circle(overlay, cv::Point(dp.iu, dp.iv), 2,
                               cv::Scalar(bf * 255, gf * 255, rf * 255), -1);
                }

                // Edge points as semi-transparent green dots on top
                if (!per_image_edge_diag[img_idx].empty())
                {
                    cv::Mat edge_layer = overlay.clone();
                    for (const auto & dp : per_image_edge_diag[img_idx])
                        cv::circle(edge_layer, cv::Point(dp.iu, dp.iv), 1,
                                   cv::Scalar(0, 255, 0), -1);
                    cv::addWeighted(edge_layer, 0.6, overlay, 0.4, 0, overlay);
                }

                std::string diag_dir = data_folder + "diagnostics/registered_points/";
                std::filesystem::create_directories(diag_dir);
                cv::imwrite(diag_dir + pose.filename, overlay);
            }
        }

        auto t_img_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t_img_end - t_img_start).count();

        std::ostringstream msg;
        msg << img_idx << "/" << image_poses.size() - 1
            << " — " << pose.filename
            << ": " << found << " points colored"
            << " (" << std::fixed << std::setprecision(2) << elapsed << "s)\n";
        #pragma omp critical
        std::cout << msg.str();
    }

    // ---- Merge per-image observations into a single list ----
    // Edge points were added to points_h before Pass 2, so they are already
    // covered by per_image_obs — no separate edge colour list needed.
    std::vector<ColorObservation> all_observations;
    all_observations.reserve(total_points * 2);
    for (const auto & obs_vec : per_image_obs)
        all_observations.insert(all_observations.end(), obs_vec.begin(), obs_vec.end());

    // ---- Save output CSV ----
    std::ofstream out(output_path);
    if (!out.is_open())
    {
        std::cerr  << "\033[31m" << "Error: Could not open output file '" << output_path << "'" << "\033[0m" << std::endl;
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
    return 0;
}