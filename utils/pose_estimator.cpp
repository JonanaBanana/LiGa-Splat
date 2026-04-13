#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <filesystem>
#include <yaml-cpp/yaml.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/PolygonMesh.h>
#include <pcl/conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <vtkObject.h>

// Interpolates a camera pose for every saved image by matching each image
// timestamp against the odometry stream.  Position is linearly interpolated;
// orientation is SLERP'd between the two bracketing odometry samples.
//
// A configurable time_delay shifts image timestamps before matching, which
// compensates for a constant trigger-to-exposure offset between the camera
// and the odometry sensor.
//
// Output: poses.csv  —  one row per image with interpolated px/py/pz/qx/qy/qz/qw

// ========================= Configuration =========================
struct Config
{
    float time_delay = 0.0f;

    // Paths (relative to data_folder)
    std::string poses_file = "poses.csv";
    std::string pose_timestamps_file = "timestamps/odom.csv";
    std::string image_timestamps_file = "timestamps/image.csv";
};

// ========================= Config Parser =========================

Config load_config(const std::string & path)
{
    Config cfg;
    YAML::Node node;

    try {
        node = YAML::LoadFile(path);
    } catch (const YAML::Exception & e) {
        std::cerr << "\033[31m" <<"Error reading config: " << e.what() << "\033[0m" <<std::endl;
        std::cerr << "\033[31m" << "Using default config values." << "\033[0m" << std::endl;
        return cfg;
    }
    cfg.time_delay       = node["time_delay"].as<float>(cfg.time_delay);
    cfg.poses_file         = node["poses_file"].as<std::string>(cfg.poses_file);
    cfg.pose_timestamps_file = node["pose_timestamps_file"].as<std::string>(cfg.pose_timestamps_file);
    cfg.image_timestamps_file = node["image_timestamps_file"].as<std::string>(cfg.image_timestamps_file);
    return cfg;
}

void print_config(const Config & cfg)
{
    std::cout << "=== Configuration ===" << std::endl;
    std::cout << "  Time delay:      " << cfg.time_delay << std::endl;
    std::cout << "  Pose timestamps: " << cfg.pose_timestamps_file << std::endl;
    std::cout << "  Image timestamps: " << cfg.image_timestamps_file << std::endl;
    std::cout << "  Output poses:      " << cfg.poses_file << std::endl;
    std::cout << "=====================" << std::endl;
}

struct Pose
{
    double timestamp; // combined seconds + nanoseconds
    double px, py, pz;
    double qx, qy, qz, qw;
};

struct ImageEntry
{
    int index;
    std::string filename;
    double timestamp;
};

double combine_timestamp(int64_t sec, uint32_t nsec)
{
    return static_cast<double>(sec) + static_cast<double>(nsec) * 1e-9;
}

// Quaternion SLERP
void quat_slerp(double t,
                 double ax, double ay, double az, double aw,
                 double bx, double by, double bz, double bw,
                 double &rx, double &ry, double &rz, double &rw)
{
    // Compute dot product
    double dot = ax * bx + ay * by + az * bz + aw * bw;

    // If dot is negative, negate one quaternion to take the shorter path
    if (dot < 0.0)
    {
        bx = -bx; by = -by; bz = -bz; bw = -bw;
        dot = -dot;
    }

    // If quaternions are very close, use linear interpolation
    if (dot > 0.9995)
    {
        rx = ax + t * (bx - ax);
        ry = ay + t * (by - ay);
        rz = az + t * (bz - az);
        rw = aw + t * (bw - aw);
    }
    else
    {
        double theta = std::acos(dot);
        double sin_theta = std::sin(theta);
        double wa = std::sin((1.0 - t) * theta) / sin_theta;
        double wb = std::sin(t * theta) / sin_theta;

        rx = wa * ax + wb * bx;
        ry = wa * ay + wb * by;
        rz = wa * az + wb * bz;
        rw = wa * aw + wb * bw;
    }

    // Normalize
    double norm = std::sqrt(rx * rx + ry * ry + rz * rz + rw * rw);
    rx /= norm; ry /= norm; rz /= norm; rw /= norm;
}

std::vector<Pose> read_odometry_csv(const std::string & path)
{
    std::vector<Pose> poses;
    std::ifstream file(path);
    if (!file.is_open())
    {
        std::cerr << "\033[31m" <<"Error: Could not open odometry file '" << path << "'" << "\033[0m" << std::endl;
        return poses;
    }

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

std::vector<ImageEntry> read_image_csv(const std::string & path)
{
    std::vector<ImageEntry> entries;
    std::ifstream file(path);
    if (!file.is_open())
    {
        std::cerr << "\033[31m" <<"Error: Could not open image timestamp file '" << path << "'" << "\033[0m" << std::endl;
        return entries;
    }

    std::string line;
    std::getline(file, line); // skip header

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string token;
        ImageEntry e;

        int64_t sec;
        uint32_t nsec;

        std::getline(ss, token, ','); e.index = std::stoi(token);
        std::getline(ss, token, ','); e.filename = token;
        std::getline(ss, token, ','); sec = std::stoll(token);
        std::getline(ss, token, ','); nsec = std::stoul(token);

        e.timestamp = combine_timestamp(sec, nsec);
        entries.push_back(e);
    }

    return entries;
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <data_folder> " << std::endl;
        return 1;
    }
    

    std::string data_folder = argv[1];
    if (data_folder.back() != '/') data_folder += '/';
    std::string config_path = data_folder + "config.cfg";
    Config cfg = load_config(config_path);
    print_config(cfg);

    std::string image_csv_path = data_folder + cfg.image_timestamps_file;
    std::string odom_csv_path = data_folder + cfg.pose_timestamps_file;
    std::string output_path = data_folder + cfg.poses_file;
    float time_delay = cfg.time_delay;

    // Read input data
    auto images = read_image_csv(image_csv_path);
    auto odom = read_odometry_csv(odom_csv_path);

    if (images.empty())
    {
        std::cerr << "No image entries loaded." << std::endl;
        return 1;
    }
    if (odom.size() < 2)
    {
        std::cerr << "Need at least 2 odometry entries for interpolation." << std::endl;
        return 1;
    }

    std::cout << "Loaded " << images.size() << " image timestamps" << std::endl;
    std::cout << "Loaded " << odom.size() << " odometry poses" << std::endl;
    std::cout << "Odom time range: [" << std::fixed << std::setprecision(6)
              << odom.front().timestamp << ", " << odom.back().timestamp << "]" << std::endl;

    // Ensure odometry is sorted by timestamp
    std::sort(odom.begin(), odom.end(),
        [](const Pose & a, const Pose & b) { return a.timestamp < b.timestamp; });

    // Interpolate pose for each image
    std::ofstream out(output_path);
    if (!out.is_open())
    {
        std::cerr << "\033[31m" <<"Error: Could not open output file '" << output_path << "'" << "\033[0m" << std::endl;
        return 1;
    }

    out << "index,filename,timestamp,px,py,pz,qx,qy,qz,qw\n";
    out << std::fixed << std::setprecision(12);

    int matched = 0;
    int skipped = 0;
    std::vector<Pose> interpolated_poses;

    for (const auto & img : images)
    {
        
        // lower_bound returns the first odom entry with timestamp >= image_ts.
        // The two bracketing samples are then [prev(it), it).
        auto it = std::lower_bound(odom.begin(), odom.end(), img.timestamp + time_delay,
            [](const Pose & p, double t) { return p.timestamp < t; });

        // Handle edge cases
        if (it == odom.begin())
        {
            // Image is before first odom — use first pose (no extrapolation)
            std::cerr << "Warning: Image '" << img.filename
                      << "' is before odometry range, using first pose." << std::endl;
            const auto & p = odom.front();
            out << img.index << "," << img.filename << "," << img.timestamp << ","
                << p.px << "," << p.py << "," << p.pz << ","
                << p.qx << "," << p.qy << "," << p.qz << "," << p.qw << "\n";
            skipped++;
            continue;
        }
        if (it == odom.end())
        {
            // Image is after last odom — use last pose
            std::cerr << "Warning: Image '" << img.filename
                      << "' is after odometry range, using last pose." << std::endl;
            const auto & p = odom.back();
            out << img.index << "," << img.filename << "," << img.timestamp + time_delay << ","
                << p.px << "," << p.py << "," << p.pz << ","
                << p.qx << "," << p.qy << "," << p.qz << "," << p.qw << "\n";
            skipped++;
            continue;
        }

        // We have two bounding odom entries
        const Pose & after = *it;
        const Pose & before = *std::prev(it);

        // Compute interpolation factor
        double dt = after.timestamp - before.timestamp;
        double t = (dt > 1e-12) ? (img.timestamp + time_delay - before.timestamp) / dt : 0.0;

        // Linearly interpolate position
        double px = before.px + t * (after.px - before.px);
        double py = before.py + t * (after.py - before.py);
        double pz = before.pz + t * (after.pz - before.pz);

        // SLERP quaternion orientation
        double qx, qy, qz, qw;
        quat_slerp(t,
                    before.qx, before.qy, before.qz, before.qw,
                    after.qx, after.qy, after.qz, after.qw,
                    qx, qy, qz, qw);

        out << img.index << "," << img.filename << "," << img.timestamp + time_delay << ","
            << px << "," << py << "," << pz << ","
            << qx << "," << qy << "," << qz << "," << qw << "\n";

        interpolated_poses.push_back({img.timestamp + time_delay, px, py, pz, qx, qy, qz, qw});
        matched++;
        
    }

    out.close();

    std::cout << "\nResults: " << matched << " interpolated, " << skipped
              << " at boundary (clamped)" << std::endl;
    std::cout << "Saved to '" << output_path << "'" << std::endl;
    return 0;
}