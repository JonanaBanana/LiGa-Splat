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
#include <cstring>

#include <omp.h>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/surface/convex_hull.h>

// ========================= Configuration =========================
struct Config
{
    double f = 1000.0;
    int img_w = 1920;
    int img_h = 1080;
    double px = 960.0;
    double py = 540.0;

    double min_depth = 1.0;
    double max_depth = 400.0;
    double hpr_radius = 100000.0;

    // Camera-to-body transform (T_body_cam): T_world_cam = T_world_body * T_body_cam
    // Set to identity if poses are already in camera frame
    Eigen::Matrix4d trans_mat = Eigen::Matrix4d::Identity();

    // If true, poses in CSV are body-frame and trans_mat is applied
    // If false, poses are already camera-frame and trans_mat is skipped
    bool poses_are_body_frame = true;

    std::string reconstructed_file = "pcd/reconstructed.pcd";
    std::string poses_file = "poses.csv";
    std::string images_dir = "images";
};

Config load_config(const std::string & path)
{
    Config cfg;
    YAML::Node node;

    try {
        node = YAML::LoadFile(path);
    } catch (const YAML::Exception & e) {
        std::cerr << "\033[31m" <<"Error reading config: " << e.what() << "\033[0m" <<std::endl;
        std::cerr << "\033[31m" << "Using default config values." << "\033[0m" << std::endl;
        cfg.trans_mat << 0,0,1,0, -1,0,0,0, 0,-1,0,0, 0,0,0,1;
        return cfg;
    }

    cfg.f     = node["focal_length"].as<double>(cfg.f);
    cfg.img_w = node["image_width"].as<int>(cfg.img_w);
    cfg.img_h = node["image_height"].as<int>(cfg.img_h);
    cfg.px    = node["principal_x"].as<double>(cfg.px);
    cfg.py    = node["principal_y"].as<double>(cfg.py);
    cfg.min_depth = node["min_depth"].as<double>(cfg.min_depth);
    cfg.max_depth = node["max_depth"].as<double>(cfg.max_depth);
    cfg.hpr_radius = node["hpr_radius"].as<double>(cfg.hpr_radius);

    cfg.poses_are_body_frame = node["poses_are_body_frame"].as<bool>(cfg.poses_are_body_frame);

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

    cfg.reconstructed_file = node["reconstructed_file"].as<std::string>(cfg.reconstructed_file);
    cfg.poses_file         = node["poses_file"].as<std::string>(cfg.poses_file);
    cfg.images_dir         = node["images_dir"].as<std::string>(cfg.images_dir);

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

struct Point2DObs
{
    double x, y;
    int64_t point3d_id;
};

// ========================= Binary Writers =========================

template<typename T>
void write_bin(std::ofstream & f, T val)
{
    f.write(reinterpret_cast<const char*>(&val), sizeof(T));
}

void write_cameras_bin(const std::string & path,
                       int camera_id, int model_id,
                       int width, int height,
                       const std::vector<double> & params)
{
    std::ofstream f(path, std::ios::binary);
    uint64_t num_cameras = 1;
    write_bin(f, num_cameras);
    write_bin(f, camera_id);
    write_bin(f, model_id);
    write_bin(f, static_cast<uint64_t>(width));
    write_bin(f, static_cast<uint64_t>(height));
    for (double p : params)
        write_bin(f, p);
}

void write_cameras_txt(const std::string & path,
                       int camera_id, const std::string & model_name,
                       int width, int height,
                       const std::vector<double> & params)
{
    std::ofstream f(path);
    f << "# Camera list with one line of data per camera:\n";
    f << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
    f << "# Number of cameras: 1\n";
    f << camera_id << " " << model_name << " " << width << " " << height;
    f << std::fixed << std::setprecision(6);
    for (double p : params)
        f << " " << p;
    f << "\n";
}

void write_images_bin(const std::string & path,
                      const std::vector<int> & image_ids,
                      const std::vector<Eigen::Vector4d> & qvecs,
                      const std::vector<Eigen::Vector3d> & tvecs,
                      const std::vector<int> & camera_ids,
                      const std::vector<std::string> & names,
                      const std::vector<std::vector<Point2DObs>> & all_obs)
{
    std::ofstream f(path, std::ios::binary);
    uint64_t num_images = image_ids.size();
    write_bin(f, num_images);

    for (size_t i = 0; i < num_images; i++)
    {
        write_bin(f, image_ids[i]);
        write_bin(f, qvecs[i](0));
        write_bin(f, qvecs[i](1));
        write_bin(f, qvecs[i](2));
        write_bin(f, qvecs[i](3));
        write_bin(f, tvecs[i](0));
        write_bin(f, tvecs[i](1));
        write_bin(f, tvecs[i](2));
        write_bin(f, camera_ids[i]);

        for (char c : names[i])
            write_bin(f, c);
        write_bin(f, '\0');

        uint64_t num_points = all_obs[i].size();
        write_bin(f, num_points);
        for (const auto & obs : all_obs[i])
        {
            write_bin(f, obs.x);
            write_bin(f, obs.y);
            write_bin(f, obs.point3d_id);
        }
    }
}

void write_images_txt(const std::string & path,
                      const std::vector<int> & image_ids,
                      const std::vector<Eigen::Vector4d> & qvecs,
                      const std::vector<Eigen::Vector3d> & tvecs,
                      const std::vector<int> & camera_ids,
                      const std::vector<std::string> & names,
                      const std::vector<std::vector<Point2DObs>> & all_obs)
{
    std::ofstream f(path);
    f << "# Image list with two lines of data per image:\n";
    f << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n";
    f << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n";
    f << "# Number of images: " << image_ids.size() << "\n";

    f << std::fixed << std::setprecision(12);
    for (size_t i = 0; i < image_ids.size(); i++)
    {
        f << image_ids[i] << " "
          << qvecs[i](0) << " " << qvecs[i](1) << " "
          << qvecs[i](2) << " " << qvecs[i](3) << " "
          << tvecs[i](0) << " " << tvecs[i](1) << " " << tvecs[i](2) << " "
          << camera_ids[i] << " " << names[i] << "\n";

        for (size_t j = 0; j < all_obs[i].size(); j++)
        {
            if (j > 0) f << " ";
            f << all_obs[i][j].x << " " << all_obs[i][j].y << " " << all_obs[i][j].point3d_id;
        }
        f << "\n";
    }
}

void write_points3D_bin(const std::string & path,
                        const pcl::PointCloud<pcl::PointXYZRGB> & cloud,
                        const std::vector<std::vector<std::pair<int32_t, int32_t>>> & tracks)
{
    std::ofstream f(path, std::ios::binary);
    uint64_t num_points = cloud.size();
    write_bin(f, num_points);

    for (uint64_t i = 0; i < num_points; i++)
    {
        const auto & pt = cloud[i];
        write_bin(f, i);
        write_bin(f, static_cast<double>(pt.x));
        write_bin(f, static_cast<double>(pt.y));
        write_bin(f, static_cast<double>(pt.z));
        write_bin(f, pt.r);
        write_bin(f, pt.g);
        write_bin(f, pt.b);
        write_bin(f, 0.1);  // placeholder error value

        uint64_t track_length = tracks[i].size();
        if (track_length == 0)
        {
            // Must have at least 1 track entry for COLMAP compatibility
            track_length = 1;
            write_bin(f, track_length);
            write_bin(f, static_cast<int32_t>(1));  // dummy image_id
            write_bin(f, static_cast<int32_t>(0));  // dummy point2d_idx
        }
        else
        {
            write_bin(f, track_length);
            for (const auto & [img_id, pt2d_idx] : tracks[i])
            {
                write_bin(f, img_id);
                write_bin(f, pt2d_idx);
            }
        }
    }
}

void write_points3D_txt(const std::string & path,
                        const pcl::PointCloud<pcl::PointXYZRGB> & cloud,
                        const std::vector<std::vector<std::pair<int32_t, int32_t>>> & tracks)
{
    std::ofstream f(path);
    
    // Compute mean track length
    double mean_track = 0;
    if (!cloud.empty())
    {
        size_t total = 0;
        for (size_t i = 0; i < cloud.size(); i++)
            total += std::max(tracks[i].size(), static_cast<size_t>(1));
        mean_track = static_cast<double>(total) / cloud.size();
    }

    f << "# 3D point list with one line of data per point:\n";
    f << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n";
    f << "# Number of points: " << cloud.size()
      << ", mean track length: " << std::fixed << std::setprecision(4) << mean_track << "\n";

    f << std::fixed << std::setprecision(8);
    for (size_t i = 0; i < cloud.size(); i++)
    {
        const auto & pt = cloud[i];
        f << i << " "
          << pt.x << " " << pt.y << " " << pt.z << " "
          << static_cast<int>(pt.r) << " "
          << static_cast<int>(pt.g) << " "
          << static_cast<int>(pt.b) << " "
          << "0.1";

        if (tracks[i].empty())
        {
            f << " 1 0";
        }
        else
        {
            for (const auto & [img_id, pt2d_idx] : tracks[i])
                f << " " << img_id << " " << pt2d_idx;
        }
        f << "\n";
    }
}

// ========================= Helpers =========================

std::vector<ImagePose> read_image_poses(const std::string & path)
{
    std::vector<ImagePose> poses;
    std::ifstream file(path);
    if (!file.is_open()) return poses;

    std::string line;
    std::getline(file, line);

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

Eigen::Matrix4d quat_to_matrix(double px, double py, double pz,
                                double qx, double qy, double qz, double qw)
{
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    Eigen::Quaterniond q(qw, qx, qy, qz);
    q.normalize();
    T.block<3, 3>(0, 0) = q.toRotationMatrix();
    T(0, 3) = px; T(1, 3) = py; T(2, 3) = pz;
    return T;
}

// Converts a rotation matrix to a unit quaternion using Shepperd's method via
// the 4×4 symmetric traceless matrix K (Bar-Itzhack, 2000).  The quaternion
// is the eigenvector of K corresponding to its largest eigenvalue.
// Matches Python's rotmat2qvec exactly (COLMAP read_write_model.py)
// Python unpacks R.flat (row-major) as: Rxx=R[0,0], Ryx=R[0,1], Rzx=R[0,2],
//   Rxy=R[1,0], Ryy=R[1,1], Rzy=R[1,2], Rxz=R[2,0], Ryz=R[2,1], Rzz=R[2,2]
// Bottom row: K[3][:3] = {Ryz-Rzy, Rzx-Rxz, Rxy-Ryx}
//           = {R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]}
Eigen::Vector4d rotmat_to_qvec(const Eigen::Matrix3d & R)
{
    Eigen::Matrix4d K;
    K << R(0,0) - R(1,1) - R(2,2), 0, 0, 0,
         R(1,0) + R(0,1), R(1,1) - R(0,0) - R(2,2), 0, 0,
         R(2,0) + R(0,2), R(2,1) + R(1,2), R(2,2) - R(0,0) - R(1,1), 0,
         R(2,1) - R(1,2), R(0,2) - R(2,0), R(1,0) - R(0,1), R(0,0) + R(1,1) + R(2,2);
    K /= 3.0;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver(K);
    Eigen::Vector4d eigvals = solver.eigenvalues();
    Eigen::Matrix4d eigvecs = solver.eigenvectors();

    int max_idx;
    eigvals.maxCoeff(&max_idx);

    Eigen::Vector4d qvec;
    qvec(0) = eigvecs(3, max_idx);
    qvec(1) = eigvecs(0, max_idx);
    qvec(2) = eigvecs(1, max_idx);
    qvec(3) = eigvecs(2, max_idx);

    if (qvec(0) < 0) qvec = -qvec;
    return qvec;
}

std::vector<int> hidden_point_removal(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr & cloud,
    double radius)
{
    int n = static_cast<int>(cloud->size());

    pcl::PointCloud<pcl::PointXYZ>::Ptr flipped(new pcl::PointCloud<pcl::PointXYZ>());
    flipped->resize(n + 1);

    for (int i = 0; i < n; i++)
    {
        const double px = cloud->points[i].x;
        const double py = cloud->points[i].y;
        const double pz = cloud->points[i].z;
        const double dist = std::sqrt(px * px + py * py + pz * pz);

        if (dist < 1e-10)
        {
            (*flipped)[i] = cloud->points[i];
            continue;
        }

        const double scale = 2.0 * (radius - dist) / dist;
        (*flipped)[i].x = px + scale * px;
        (*flipped)[i].y = py + scale * py;
        (*flipped)[i].z = pz + scale * pz;
    }

    (*flipped)[n].x = 0;
    (*flipped)[n].y = 0;
    (*flipped)[n].z = 0;

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
        std::cerr << "Usage: " << argv[0] << " <data_folder>" << std::endl;
        std::cerr << "Output: <data_folder>/distorted/sparse/0/{cameras,images,points3D}.{bin,txt}" << std::endl;
        return 1;
    }

    std::string data_folder = argv[1];
    if (data_folder.back() != '/') data_folder += '/';
    std::string config_path = data_folder + "config.cfg";

    Config cfg = load_config(config_path);

    std::cout << "Poses are " << (cfg.poses_are_body_frame ? "BODY frame (trans_mat will be applied)" : "CAMERA frame (trans_mat skipped)") << std::endl;

    std::string pcd_path = data_folder + cfg.reconstructed_file;
    std::string poses_path = data_folder + cfg.poses_file;
    std::string images_dir = data_folder + cfg.images_dir + "/";
    std::string output_dir = data_folder + "distorted/sparse/0/";

    std::filesystem::create_directories(output_dir);

    auto t_start = std::chrono::high_resolution_clock::now();

    // ---- Load reconstructed point cloud ----
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(pcd_path, *cloud) == -1)
    {
        std::cerr << "\033[31m" << "Error: Could not load PCD file '" << pcd_path << "'" << "\033[0m" << std::endl;
        return 1;
    }
    int total_points = static_cast<int>(cloud->size());
    std::cout << "Loaded " << total_points << " points from " << pcd_path << std::endl;

    // ---- Load image poses ----
    auto image_poses = read_image_poses(poses_path);
    if (image_poses.empty())
    {
        std::cerr << "No image poses loaded." << std::endl;
        return 1;
    }
    int N = static_cast<int>(image_poses.size());
    std::cout << "Loaded " << N << " image poses" << std::endl;

    // ==================================================================
    // cameras.bin / cameras.txt
    // ==================================================================
    {
        int camera_id = 1;
        int model_id = 0; // SIMPLE_PINHOLE
        std::vector<double> params = {cfg.f, cfg.px, cfg.py};

        write_cameras_bin(output_dir + "cameras.bin", camera_id, model_id,
                          cfg.img_w, cfg.img_h, params);
        write_cameras_txt(output_dir + "cameras.txt", camera_id, "SIMPLE_PINHOLE",
                          cfg.img_w, cfg.img_h, params);
        std::cout << "cameras.bin and cameras.txt created!" << std::endl;
    }

    // ==================================================================
    // images.bin / images.txt
    // ==================================================================
    double fov_x = 2.0 * std::atan2(cfg.img_w, 2.0 * cfg.f);
    double fov_y = 2.0 * std::atan2(cfg.img_h, 2.0 * cfg.f);
    double tan_half_fov_x = std::tan(fov_x * 0.5);
    double tan_half_fov_y = std::tan(fov_y * 0.5);

    // Build homogeneous points [4 x R]
    Eigen::Matrix<double, 4, Eigen::Dynamic> points_h(4, total_points);
    for (int i = 0; i < total_points; i++)
    {
        points_h(0, i) = cloud->points[i].x;
        points_h(1, i) = cloud->points[i].y;
        points_h(2, i) = cloud->points[i].z;
        points_h(3, i) = 1.0;
    }

    // Output arrays — pre-sized so each thread writes to its own slot
    std::vector<int> image_ids(N);
    std::vector<Eigen::Vector4d> qvecs(N);
    std::vector<Eigen::Vector3d> tvecs(N);
    std::vector<int> camera_ids(N, 1);
    std::vector<std::string> image_names(N);
    std::vector<std::vector<Point2DObs>> all_observations(N);

    std::cout << "Processing images for COLMAP export..." << std::endl;

    #pragma omp parallel for schedule(dynamic)
    for (int img_idx = 0; img_idx < N; img_idx++)
    {
        const auto & pose = image_poses[img_idx];

        // Thread-local buffers
        Eigen::Matrix<double, 4, Eigen::Dynamic> cam_points(4, total_points);
        std::vector<int> fov_indices;
        fov_indices.reserve(total_points / 4);
        pcl::PointCloud<pcl::PointXYZ>::Ptr visible_cloud(new pcl::PointCloud<pcl::PointXYZ>());

        // ----------------------------------------------------------------
        // Step 1: Build T_world_cam from pose
        //   - If poses are body-frame: T_world_cam = T_world_body * trans_mat
        //   - If poses are camera-frame: T_world_cam = T_world_pose (as-is)
        // ----------------------------------------------------------------
        Eigen::Matrix4d T_pose = quat_to_matrix(
            pose.px, pose.py, pose.pz,
            pose.qx, pose.qy, pose.qz, pose.qw);

        Eigen::Matrix4d T_world_cam;
        if (cfg.poses_are_body_frame)
            T_world_cam = T_pose * cfg.trans_mat;
        else
            T_world_cam = T_pose;

        // ----------------------------------------------------------------
        // Step 2: Compute T_cam_world = inv(T_world_cam)
        //   This is what COLMAP stores: "projection from world to camera"
        //   Matches Python: q_transform = np.linalg.inv(transform[i] @ trans_mat)
        // ----------------------------------------------------------------
        Eigen::Matrix4d T_cam_world = T_world_cam.inverse();

        // ----------------------------------------------------------------
        // Step 3: Extract COLMAP qvec and tvec from T_cam_world
        //   qvec = rotmat2qvec(T_cam_world[:3,:3])
        //   tvec = T_cam_world[:3, 3]
        //   Camera center can be recovered as: C = -R^T * t
        // ----------------------------------------------------------------
        Eigen::Matrix3d R_cam_world = T_cam_world.block<3, 3>(0, 0);
        Eigen::Vector3d t_cam_world = T_cam_world.block<3, 1>(0, 3);
        Eigen::Vector4d qvec = rotmat_to_qvec(R_cam_world);

        // ----------------------------------------------------------------
        // Step 4: Transform points to camera frame for FOV/HPR/projection
        //   points_h is read-only — safe to share across threads
        // ----------------------------------------------------------------
        cam_points.noalias() = T_cam_world * points_h;

        // FOV + depth filtering
        for (int i = 0; i < total_points; i++)
        {
            double cz = cam_points(2, i);
            if (cz < cfg.min_depth || cz > cfg.max_depth) continue;

            double cx = cam_points(0, i);
            double cy = cam_points(1, i);

            if (std::abs(cx) < cz * tan_half_fov_x && std::abs(cy) < cz * tan_half_fov_y)
                fov_indices.push_back(i);
        }

        image_ids[img_idx]   = img_idx + 1;
        qvecs[img_idx]       = qvec;
        tvecs[img_idx]       = t_cam_world;
        image_names[img_idx] = pose.filename;

        if (fov_indices.empty())
        {
            all_observations[img_idx] = {};
            continue;
        }

        // Hidden point removal
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

        std::vector<int> hpr_indices = hidden_point_removal(visible_cloud, cfg.hpr_radius);

        // Project visible points
        std::vector<Point2DObs> obs;
        obs.reserve(hpr_indices.size());

        for (int hi : hpr_indices)
        {
            if (hi < 0 || hi >= static_cast<int>(fov_indices.size())) continue;
            int orig_idx = fov_indices[hi];

            double cx = cam_points(0, orig_idx);
            double cy = cam_points(1, orig_idx);
            double cz = cam_points(2, orig_idx);

            double u = cfg.f * cx / cz + cfg.px;
            double v = cfg.f * cy / cz + cfg.py;

            if (u < 0 || u >= cfg.img_w || v < 0 || v >= cfg.img_h)
                continue;

            obs.push_back({u, v, static_cast<int64_t>(orig_idx)});
        }

        all_observations[img_idx] = std::move(obs);

        std::ostringstream msg;
        msg << img_idx << "/" << N - 1
            << " — " << pose.filename
            << ": " << all_observations[img_idx].size() << " 2D observations\n";
        #pragma omp critical
        std::cout << msg.str();
    }

    write_images_bin(output_dir + "images.bin",
                     image_ids, qvecs, tvecs, camera_ids, image_names, all_observations);
    write_images_txt(output_dir + "images.txt",
                     image_ids, qvecs, tvecs, camera_ids, image_names, all_observations);
    std::cout << "images.bin and images.txt created!" << std::endl;

    // ==================================================================
    // points3D.bin / points3D.txt
    // ==================================================================
    std::cout << "Writing points3D..." << std::endl;
    // A COLMAP "track" records every 2D observation of a 3D point across images:
    // track[point_id] = list of (image_id, index_into_that_image's_obs_list).
    // We invert the all_observations table (image → point) into this point → images form.
    std::vector<std::vector<std::pair<int32_t, int32_t>>> tracks(total_points);

    for (size_t i = 0; i < all_observations.size(); i++)
    {
        int32_t img_id = image_ids[i];
        for (int32_t j = 0; j < static_cast<int32_t>(all_observations[i].size()); j++)
        {
            int pt_id = static_cast<int>(all_observations[i][j].point3d_id);
            if (pt_id >= 0 && pt_id < total_points)
                tracks[pt_id].push_back({img_id, j});
        }
    }

    write_points3D_bin(output_dir + "points3D.bin", *cloud, tracks);
    write_points3D_txt(output_dir + "points3D.txt", *cloud, tracks);
    std::cout << "points3D.bin and points3D.txt created!" << std::endl;

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\nCOLMAP export complete in " << std::fixed << std::setprecision(1)
              << elapsed << "s" << std::endl;
    std::cout << "Output: " << output_dir << std::endl;

    return 0;
}