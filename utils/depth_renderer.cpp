#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <filesystem>
#include <chrono>
#include <limits>

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/surface/convex_hull.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <omp.h>

namespace fs = std::filesystem;

// ========================= Configuration =========================
struct Config
{
    // Camera intrinsics
    double f = 1000.0;
    int img_w = 1920;
    int img_h = 1080;
    double px = 960.0;
    double py = 540.0;

    // Depth range
    double min_depth = 1.0;
    double max_depth = 400.0;

    // Outlier filtering (applied to raw input cloud before rendering)
    bool   filter_outliers = true;
    int    sor_neighbors   = 10;
    double sor_std_ratio   = 2.0;

    // Background sphere (denser than registration to fill depth in sky/void regions)
    bool   fill_background      = false;
    double sphere_radius        = 50.0;
    int    depth_sphere_num_pts = 500000;

    // Hidden point removal
    double depth_render_hpr_radius = 40000.0; // intentionally higher than registration to keep more points

    // Dense completion (JBF)
    float jbf_sigma_c = 15.0f; // colour std-dev (0–255 scale); lower = sharper edges, more unfilled voids

    // Camera-to-body transform (T_body_cam)
    bool poses_are_body_frame = true;
    Eigen::Matrix4d trans_mat = Eigen::Matrix4d::Identity();

    // Paths (relative to data_folder)
    std::string pcd_file   = "pcd/input.pcd";
    std::string poses_file = "poses.csv";
    std::string images_dir = "distorted/images";
    std::string depth_dir  = "depth_renders";
};

// ========================= Config Parser =========================
Config load_config(const std::string & path)
{
    Config cfg;
    YAML::Node node;

    try {
        node = YAML::LoadFile(path);
    } catch (const YAML::Exception & e) {
        std::cerr << "\033[31m" <<"Error reading config: " << e.what() << "\033[0m" << std::endl;
        std::cerr << "\033[31m" << "Using default config values." << "\033[0m" << std::endl;
        cfg.trans_mat << 0,0,1,0, -1,0,0,0, 0,-1,0,0, 0,0,0,1;
        return cfg;
    }

    cfg.f     =     node["focal_length"].as<double>(cfg.f);
    cfg.img_w =     node["image_width"].as<int>(cfg.img_w);
    cfg.img_h =     node["image_height"].as<int>(cfg.img_h);
    cfg.px    =     node["principal_x"].as<double>(cfg.px);
    cfg.py    =     node["principal_y"].as<double>(cfg.py);
    cfg.min_depth = node["min_depth"].as<double>(cfg.min_depth);
    cfg.max_depth = node["max_depth"].as<double>(cfg.max_depth);

    cfg.filter_outliers      = node["filter_outliers"].as<bool>(cfg.filter_outliers);
    cfg.sor_neighbors        = node["sor_neighbors"].as<int>(cfg.sor_neighbors);
    cfg.sor_std_ratio        = node["sor_std_ratio"].as<double>(cfg.sor_std_ratio);

    cfg.fill_background      = node["fill_background"].as<bool>(cfg.fill_background);
    cfg.sphere_radius        = node["sphere_radius"].as<double>(cfg.sphere_radius);
    cfg.depth_sphere_num_pts = node["depth_sphere_num_points"].as<int>(cfg.depth_sphere_num_pts);

    cfg.depth_render_hpr_radius = node["depth_render_hpr_radius"].as<double>(cfg.depth_render_hpr_radius);
    cfg.jbf_sigma_c             = node["jbf_sigma_c"].as<float>(cfg.jbf_sigma_c);
    cfg.poses_are_body_frame    = node["poses_are_body_frame"].as<bool>(cfg.poses_are_body_frame);

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

    cfg.pcd_file   = node["pcd_file"].as<std::string>(cfg.pcd_file);
    cfg.poses_file = node["poses_file"].as<std::string>(cfg.poses_file);
    cfg.images_dir = node["images_dir"].as<std::string>(cfg.images_dir);
    cfg.depth_dir  = node["depth_dir"].as<std::string>(cfg.depth_dir);

    return cfg;
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

struct DepthResult
{
    bool        success  = false;
    std::string out_stem;
    float       d_min    = 0.0f;
    float       d_max    = 0.0f;
    double      coverage = 0.0;
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
        std::cerr << "\033[31m" << "Error: Could not open pose file '" << path << "'" << "\033[0m" << std::endl;
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

// Derive output stem from image filename (strip extension)
static std::string stem(const std::string & filename)
{
    auto pos = filename.rfind('.');
    return (pos == std::string::npos) ? filename : filename.substr(0, pos);
}

// ========================= Depth Completion =========================

// Morphological fallback — used when the color guide image cannot be loaded.
cv::Mat complete_depth_map_morphological(const cv::Mat & sparse)
{
    cv::Mat valid_mask = (sparse > 0.0f);
    cv::Mat filled     = sparse.clone();

    auto ellipse = [](int s) {
        return cv::getStructuringElement(cv::MORPH_ELLIPSE, {s, s});
    };

    cv::dilate(filled, filled, ellipse(5));
    cv::dilate(filled, filled, ellipse(15));
    cv::morphologyEx(filled, filled, cv::MORPH_CLOSE, ellipse(61));
    cv::GaussianBlur(filled, filled, {9, 9}, 3.0);

    sparse.copyTo(filled, valid_mask);
    return filled;
}

// Color-guided joint bilateral sparse depth completion.
//
//
// Any residual unfilled pixels (e.g. sky voids that had no color-compatible
// LiDAR source) are closed with a small 7×7 post-dilation.
//
// Parameters:
//   radius  = 4 px    — window half-size; larger → fewer voids, slower, more bleed risk
//   sigma_s = 3.0 px  — spatial roll-off; well matched to radius=4 (edge weight ≈ 0.4)
//   sigma_c           — colour std-dev (0–255 scale); lower = sharper edges / more voids
//                       configurable via jbf_sigma_c in config.cfg (default 15)

cv::Mat complete_depth_map_guided(const cv::Mat & sparse,
                                   const cv::Mat & guide_bgr,
                                   int   radius  = 4,
                                   float sigma_s = 3.0f,
                                   float sigma_c = 3.0f)
{
    cv::Mat valid_mask = (sparse > 0.0f);

    // Precompute spatial Gaussian table [|dy|][|dx|] for dy,dx in [0, radius]
    const int W = radius + 1;
    std::vector<float> spatial_lut(W * W);
    const float inv_2ss = 1.0f / (2.0f * sigma_s * sigma_s);
    for (int dy = 0; dy <= radius; dy++)
        for (int dx = 0; dx <= radius; dx++)
            spatial_lut[dy * W + dx] = std::exp(-(float)(dy * dy + dx * dx) * inv_2ss);

    // Precompute color Gaussian LUT indexed by integer color-diff².
    const float inv_2sc     = 1.0f / (2.0f * sigma_c * sigma_c);
    const int   color_limit = static_cast<int>(9.0f * sigma_c * sigma_c) + 1;
    std::vector<float> color_lut(color_limit);
    for (int i = 0; i < color_limit; i++)
        color_lut[i] = std::exp(-static_cast<float>(i) * inv_2sc);

    cv::Mat out(sparse.size(), CV_32FC1, cv::Scalar(0.0f));

    for (int y = 0; y < sparse.rows; y++)
    {
        const int y0 = std::max(0, y - radius);
        const int y1 = std::min(sparse.rows - 1, y + radius);
        const uchar * gc_row = guide_bgr.ptr<uchar>(y);
        float * out_row = out.ptr<float>(y);

        for (int x = 0; x < sparse.cols; x++)
        {
            const float gc0 = gc_row[x * 3];
            const float gc1 = gc_row[x * 3 + 1];
            const float gc2 = gc_row[x * 3 + 2];

            const int x0 = std::max(0, x - radius);
            const int x1 = std::min(sparse.cols - 1, x + radius);

            float sum_w = 0.0f;
            float sum_d = 0.0f;

            for (int ny = y0; ny <= y1; ny++)
            {
                const int dy = std::abs(ny - y);
                const float * d_row  = sparse.ptr<float>(ny);
                const uchar * gn_row = guide_bgr.ptr<uchar>(ny);

                for (int nx = x0; nx <= x1; nx++)
                {
                    const float d = d_row[nx];
                    if (d <= 0.0f) continue;

                    const float dc0 = gc0 - gn_row[nx * 3];
                    const float dc1 = gc1 - gn_row[nx * 3 + 1];
                    const float dc2 = gc2 - gn_row[nx * 3 + 2];
                    const int cdiff2 = static_cast<int>(dc0*dc0 + dc1*dc1 + dc2*dc2);
                    if (cdiff2 >= color_limit) continue;

                    const int dx = std::abs(nx - x);
                    const float w = spatial_lut[dy * W + dx] * color_lut[cdiff2];
                    sum_w += w;
                    sum_d += w * d;
                }
            }

            if (sum_w > 1e-6f)
                out_row[x] = sum_d / sum_w;
        }
    }

    // Post-fill: small dilation closes residual no-data gaps.
    cv::Mat k7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, {7, 7});
    cv::dilate(out, out, k7);

    // Restore exact LiDAR measurements at all originally valid pixels
    sparse.copyTo(out, valid_mask);
    return out;
}

// ========================= Main =========================
int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <data_folder> [--no-hpr] [--dense] [--diag] [--save-tiff]" << std::endl;
        return 1;
    }

    bool use_hpr          = true;
    bool dense            = false;
    bool save_diagnostics = false;
    bool save_tiff        = false;
    for (int i = 2; i < argc; i++)
    {
        if (std::string(argv[i]) == "--no-hpr")    use_hpr          = false;
        if (std::string(argv[i]) == "--dense")     dense            = true;
        if (std::string(argv[i]) == "--diag")      save_diagnostics = true;
        if (std::string(argv[i]) == "--save-tiff") save_tiff        = true;
    }

    std::string data_folder = argv[1];
    if (data_folder.back() != '/') data_folder += '/';

    Config cfg = load_config(data_folder + "config.cfg");

    // Output paths
    fs::path tiff_dir    = fs::path(data_folder) / cfg.depth_dir;
    fs::path png_dir     = fs::path(data_folder) / "distorted/depth";
    fs::path params_path = fs::path(data_folder) / "distorted/sparse/0/depth_params.json";

    std::cout << "=== Depth Renderer ===" << std::endl;
    std::cout << "  Input cloud:           " << cfg.pcd_file << std::endl;
    std::cout << "  Outlier removal:       " << (cfg.filter_outliers ? "yes" : "no") << std::endl;
    std::cout << "  Background sphere:     " << (cfg.fill_background ? std::to_string(cfg.depth_sphere_num_pts) + " pts, r=" + std::to_string((int)cfg.sphere_radius) + "m" : "no") << std::endl;
    std::cout << "  Resolution:            " << cfg.img_w << "x" << cfg.img_h << std::endl;
    std::cout << "  Focal length:          " << cfg.f << std::endl;
    std::cout << "  Depth range:          [" << cfg.min_depth << ", " << cfg.max_depth << "] m" << std::endl;
    std::cout << "  Hidden point removal:  " << (use_hpr ? "on" : "off") << std::endl;
    std::cout << "  HPR radius:            " << cfg.depth_render_hpr_radius << std::endl;
    std::cout << "  Dense completion:      " << (dense  ? "yes (color-guided JBF)" : "no") << std::endl;
    std::cout << "  Save float32 TIFFs:    " << (save_tiff ? "yes -> " + tiff_dir.string() : "no") << std::endl;
    std::cout << "  Diagnostics:           " << (save_diagnostics ? "yes" : "no") << std::endl;
    std::cout << "  PNG output:            " << png_dir.string() << std::endl;
    std::cout << "  depth_params.json:     " << params_path.string() << std::endl;
    std::cout << "======================" << std::endl;

    // ---- Load and filter point cloud ----
    auto t_start = std::chrono::high_resolution_clock::now();

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(data_folder + cfg.pcd_file, *cloud) == -1)
    {
        std::cerr << "\033[31m" << "Error: Could not load PCD file '"
                  << data_folder + cfg.pcd_file << "'" << "\033[0m" << std::endl;
        return 1;
    }
    std::cout << "Loaded " << cloud->size() << " points from " << cfg.pcd_file << std::endl;

    if (cfg.filter_outliers)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr clean(new pcl::PointCloud<pcl::PointXYZ>());
        for (const auto & pt : cloud->points)
            if (std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z))
                clean->push_back(pt);

        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(clean);
        sor.setMeanK(cfg.sor_neighbors);
        sor.setStddevMulThresh(cfg.sor_std_ratio);
        sor.filter(*cloud);
        std::cout << "After outlier removal: " << cloud->size() << " points" << std::endl;
    }

    int num_cloud  = static_cast<int>(cloud->size());
    int num_sphere = cfg.fill_background ? cfg.depth_sphere_num_pts : 0;
    int num_points = num_cloud + num_sphere;

    Eigen::Matrix<double, 4, Eigen::Dynamic> points_h(4, num_points);
    for (int i = 0; i < num_cloud; i++)
    {
        points_h(0, i) = cloud->points[i].x;
        points_h(1, i) = cloud->points[i].y;
        points_h(2, i) = cloud->points[i].z;
        points_h(3, i) = 1.0;
    }

    if (cfg.fill_background)
    {
        // Compute centroid from cloud points
        Eigen::Vector3d centroid(0.0, 0.0, 0.0);
        for (int i = 0; i < num_cloud; i++)
            centroid += points_h.col(i).head<3>();
        centroid /= num_cloud;

        std::cout << "Adding " << num_sphere << " background sphere points (r="
                  << cfg.sphere_radius << "m) around ["
                  << centroid.transpose() << "]" << std::endl;

        // Fibonacci sphere (uniform distribution)
        for (int i = 0; i < num_sphere; i++)
        {
            double idx   = static_cast<double>(i) + 0.5;
            double phi   = std::acos(1.0 - 2.0 * idx / num_sphere);
            double theta = M_PI * (1.0 + std::sqrt(5.0)) * idx;
            int col = num_cloud + i;
            points_h(0, col) = std::cos(theta) * std::sin(phi) * cfg.sphere_radius + centroid(0);
            points_h(1, col) = std::sin(theta) * std::sin(phi) * cfg.sphere_radius + centroid(1);
            points_h(2, col) = std::cos(phi)                   * cfg.sphere_radius + centroid(2);
            points_h(3, col) = 1.0;
        }
    }

    auto image_poses = read_image_poses(data_folder + cfg.poses_file);
    if (image_poses.empty())
    {
        std::cerr << "No image poses loaded." << std::endl;
        return 1;
    }
    std::cout << "Loaded " << image_poses.size() << " image poses" << std::endl;

    std::string images_dir = data_folder + cfg.images_dir + "/";

    fs::create_directories(png_dir);
    if (save_tiff)
        fs::create_directories(tiff_dir);

    fs::create_directories(params_path.parent_path());

    std::string diagnostics_dir = data_folder + "diagnostics/depth_renders/";
    if (save_diagnostics)
        fs::create_directories(diagnostics_dir);

    // Precompute FOV tangent thresholds
    double fov_x = 2.0 * std::atan2(cfg.img_w, 2.0 * cfg.f);
    double fov_y = 2.0 * std::atan2(cfg.img_h, 2.0 * cfg.f);
    double tan_half_fov_x = std::tan(fov_x * 0.5);
    double tan_half_fov_y = std::tan(fov_y * 0.5);

    double margin_x   = cfg.img_w * 0.01;
    double margin_y   = cfg.img_h * 0.01;
    double max_proj_x = cfg.img_w - margin_x;
    double max_proj_y = cfg.img_h - margin_y;

    std::vector<DepthResult> results(image_poses.size());
    int saved = 0;

    // Each image is fully independent (read-only access to points_h), so the
    // loop is embarrassingly parallel. Each thread gets its own working buffers.
    #pragma omp parallel for schedule(dynamic) reduction(+:saved)
    for (size_t img_idx = 0; img_idx < image_poses.size(); img_idx++)
    {
        const auto & pose = image_poses[img_idx];
        auto t0 = std::chrono::high_resolution_clock::now();

        // Thread-local working buffers
        Eigen::Matrix<double, 4, Eigen::Dynamic> cam_points(4, num_points);
        std::vector<int> fov_indices;
        fov_indices.reserve(num_points / 4);
        pcl::PointCloud<pcl::PointXYZ>::Ptr visible_cloud(new pcl::PointCloud<pcl::PointXYZ>());

        // Build transform
        Eigen::Matrix4d T_world_body = quat_to_matrix(
            pose.px, pose.py, pose.pz,
            pose.qx, pose.qy, pose.qz, pose.qw);

        Eigen::Matrix4d T_cam_world = (cfg.poses_are_body_frame
                                      ? T_world_body * cfg.trans_mat
                                      : T_world_body).inverse();

        cam_points.noalias() = T_cam_world * points_h;

        // FOV filter
        for (int i = 0; i < num_points; i++)
        {
            double cz = cam_points(2, i);
            if (cz < cfg.min_depth || cz > cfg.max_depth) continue;

            double cx = cam_points(0, i);
            double cy = cam_points(1, i);
            if (std::abs(cx) < cz * tan_half_fov_x && std::abs(cy) < cz * tan_half_fov_y)
                fov_indices.push_back(i);
        }

        if (fov_indices.empty())
        {
            std::ostringstream msg;
            msg << img_idx << "/" << image_poses.size() - 1
                << " — " << pose.filename << ": no points in FOV, skipping\n";
            #pragma omp critical
            std::cout << msg.str();
            continue;
        }

        // Hidden point removal (optional)
        std::vector<int> render_indices;
        if (use_hpr)
        {
            visible_cloud->points.resize(fov_indices.size());
            visible_cloud->width  = fov_indices.size();
            visible_cloud->height = 1;
            for (size_t j = 0; j < fov_indices.size(); j++)
            {
                int idx = fov_indices[j];
                visible_cloud->points[j].x = cam_points(0, idx);
                visible_cloud->points[j].y = cam_points(1, idx);
                visible_cloud->points[j].z = cam_points(2, idx);
            }

            Eigen::Vector3d origin(0.0, 0.0, 0.0);
            std::vector<int> hpr = hidden_point_removal(visible_cloud, origin, cfg.depth_render_hpr_radius);

            render_indices.reserve(hpr.size());
            for (int hi : hpr)
                if (hi >= 0 && hi < static_cast<int>(fov_indices.size()))
                    render_indices.push_back(fov_indices[hi]);
        }
        else
        {
            render_indices = fov_indices;
        }

        if (render_indices.empty()) continue;

        // ---- Z-buffer depth render ----
        cv::Mat depth_map(cfg.img_h, cfg.img_w, CV_32FC1, cv::Scalar(0.0f));

        int projected = 0;
        for (int ri : render_indices)
        {
            double cx = cam_points(0, ri);
            double cy = cam_points(1, ri);
            double cz = cam_points(2, ri);

            double u = cfg.f * cx / cz + cfg.px;
            double v = cfg.f * cy / cz + cfg.py;

            if (u < margin_x || u >= max_proj_x || v < margin_y || v >= max_proj_y)
                continue;

            int iu = static_cast<int>(std::round(u));
            int iv = static_cast<int>(std::round(v));
            iu = std::clamp(iu, 0, cfg.img_w - 1);
            iv = std::clamp(iv, 0, cfg.img_h - 1);

            float depth = static_cast<float>(cz);
            float & pixel = depth_map.at<float>(iv, iu);
            if (pixel == 0.0f || depth < pixel)
            {
                pixel = depth;
                projected++;
            }
        }

        // Optional dense completion (colour-guided joint bilateral)
        if (dense)
        {
            cv::Mat guide = cv::imread(images_dir + pose.filename, cv::IMREAD_COLOR);
            if (!guide.empty())
            {
                depth_map = complete_depth_map_guided(depth_map, guide, 4, 3.0f, cfg.jbf_sigma_c);
            }
            else
            {
                std::ostringstream msg;
                msg << "Warning: could not load '" << images_dir + pose.filename
                    << "', falling back to morphological completion.\n";
                #pragma omp critical
                std::cerr << msg.str();
                depth_map = complete_depth_map_morphological(depth_map);
            }
        }

        // Optional: save float32 TIFF (useful for debugging or reprocessing)
        if (save_tiff)
        {
            std::string tiff_path = (tiff_dir / (stem(pose.filename) + ".tiff")).string();
            if (!cv::imwrite(tiff_path, depth_map))
            {
                std::ostringstream msg;
                msg << "Warning: failed to write '" << tiff_path << "'\n";
                #pragma omp critical
                std::cerr << msg.str();
            }
        }

        // ---- Convert to 16-bit PNG inverse-depth ----
        // pixel = round(65536 / depth_metres), clamped to [0, 65535]
        // pixel = 0 is reserved for no-data (depth == 0)
        int n_valid = 0;
        float d_min = std::numeric_limits<float>::max();
        float d_max = 0.0f;

        cv::Mat depth_u16(cfg.img_h, cfg.img_w, CV_16UC1);
        for (int r = 0; r < depth_map.rows; r++)
            for (int c = 0; c < depth_map.cols; c++)
            {
                float d = depth_map.at<float>(r, c);
                if (d <= 0.0f)
                {
                    depth_u16.at<uint16_t>(r, c) = 0;
                }
                else
                {
                    n_valid++;
                    d_min = std::min(d_min, d);
                    d_max = std::max(d_max, d);
                    double inv = 65536.0 / static_cast<double>(d);
                    depth_u16.at<uint16_t>(r, c) =
                        static_cast<uint16_t>(std::min(inv + 0.5, 65535.0));
                }
            }

        std::string out_stem = stem(pose.filename);
        fs::path png_path    = png_dir / (out_stem + ".png");

        if (!cv::imwrite(png_path.string(), depth_u16))
        {
            std::ostringstream msg;
            msg << "Warning: failed to write '" << png_path.string() << "'\n";
            #pragma omp critical
            std::cerr << msg.str();
            continue;
        }

        double coverage = 100.0 * n_valid / (cfg.img_h * cfg.img_w);
        results[img_idx] = {true, out_stem, d_min, d_max, coverage};
        saved++;

        // Optional false-colour diagnostics PNG
        if (save_diagnostics)
        {
            cv::Mat valid_mask = (depth_map > 0.0f);
            double dmin_d, dmax_d;
            cv::minMaxLoc(depth_map, &dmin_d, &dmax_d, nullptr, nullptr, valid_mask);

            cv::Mat norm(cfg.img_h, cfg.img_w, CV_8UC1, cv::Scalar(0));
            if (dmax_d > dmin_d)
                cv::normalize(depth_map, norm, 0, 255, cv::NORM_MINMAX, CV_8UC1, valid_mask);

            cv::Mat color;
            cv::applyColorMap(norm, color, cv::COLORMAP_TURBO);
            color.setTo(cv::Scalar(0, 0, 0), ~valid_mask);

            cv::imwrite(diagnostics_dir + out_stem + ".png", color);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        std::ostringstream msg;
        msg << img_idx << "/" << image_poses.size() - 1
            << " — " << pose.filename
            << ": " << projected << " pts projected"
            << ", coverage " << std::fixed << std::setprecision(1) << coverage << "%"
            << ", depth [" << d_min << ", " << d_max << "] m"
            << " (" << std::setprecision(2) << elapsed << "s)\n";
        #pragma omp critical
        std::cout << msg.str();
    }

    // ---- Write depth_params.json (sequential, sorted order) ----
    std::ofstream json(params_path);
    if (!json.is_open())
    {
        std::cerr << "\033[31m" << "Error: could not open "
                  << params_path << " for writing" << "\033[0m" << std::endl;
        return 1;
    }

    int written = 0;
    json << "{\n";
    for (size_t i = 0; i < results.size(); ++i)
    {
        if (!results[i].success) continue;
        if (written > 0) json << ",\n";
        json << "  \"" << results[i].out_stem << "\": "
             << std::fixed << std::setprecision(10)
             << "{\"scale\": 1.0, \"offset\": 0.0}";
        written++;
    }
    json << "\n}\n";
    json.close();

    auto t_end = std::chrono::high_resolution_clock::now();
    double total = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\nDone! Saved " << saved << " depth maps to '" << png_dir.string() << "'"
              << " in " << std::fixed << std::setprecision(1) << total << "s" << std::endl;
    std::cout << "Wrote " << params_path.string() << std::endl;
    std::cout << "\n3DGS training command:\n"
              << "  python train.py \\\n"
              << "      -s " << data_folder << "distorted \\\n"
              << "      -d " << png_dir.string() << " \\\n"
              << "      --depth_params " << params_path.string() << " \\\n"
              << "      --eval --densify_until_iter 10000 --opacity_reset_interval 11000 <other args>" << std::endl;

    return 0;
}
