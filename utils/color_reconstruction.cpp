#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <cmath>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

struct ColorObs
{
    int point_id;
    float r, g, b;
};

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <data_folder> [--ascii]" << std::endl;
        std::cerr << "Reads:" << std::endl;
        std::cerr << "  <data_folder>/pcd/downsampled_point_cloud.pcd" << std::endl;
        std::cerr << "  <data_folder>/point_cloud_color_information.csv" << std::endl;
        std::cerr << "Writes:" << std::endl;
        std::cerr << "  <data_folder>/pcd/reconstructed.pcd" << std::endl;
        return 1;
    }

    std::string data_folder = argv[1];
    if (data_folder.back() != '/') data_folder += '/';

    bool save_ascii = false;
    for (int i = 2; i < argc; i++)
    {
        if (std::string(argv[i]) == "--ascii") save_ascii = true;
    }

    std::string pcd_path = data_folder + "pcd/downsampled_point_cloud.pcd";
    std::string csv_path = data_folder + "point_cloud_color_information.csv";
    std::string out_path = data_folder + "pcd/reconstructed.pcd";

    auto t_start = std::chrono::high_resolution_clock::now();

    // ---- Load point cloud ----
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_path, *cloud) == -1)
    {
        std::cerr << "Error: Could not load PCD file '" << pcd_path << "'" << std::endl;
        return 1;
    }
    int total_points = static_cast<int>(cloud->size());
    std::cout << "Loaded " << total_points << " points from " << pcd_path << std::endl;

    // ---- Read color observations ----
    std::vector<ColorObs> observations;
    {
        std::ifstream file(csv_path);
        if (!file.is_open())
        {
            std::cerr << "Error: Could not open CSV file '" << csv_path << "'" << std::endl;
            return 1;
        }

        std::string line;
        while (std::getline(file, line))
        {
            if (line.empty()) continue;

            std::stringstream ss(line);
            std::string token;
            ColorObs obs;

            std::getline(ss, token, ','); obs.point_id = std::stoi(token);
            std::getline(ss, token, ','); obs.r = std::stof(token);
            std::getline(ss, token, ','); obs.g = std::stof(token);
            std::getline(ss, token, ','); obs.b = std::stof(token);

            observations.push_back(obs);
        }
    }
    std::cout << "Loaded " << observations.size() << " color observations" << std::endl;

    if (observations.empty())
    {
        std::cerr << "No color observations found." << std::endl;
        return 1;
    }

    // ---- Sort by point_id ----
    std::sort(observations.begin(), observations.end(),
        [](const ColorObs & a, const ColorObs & b) { return a.point_id < b.point_id; });

    // ---- Compute median color per point ----
    // Group observations by point_id and compute channel-wise median
    pcl::PointCloud<pcl::PointXYZRGB> output_cloud;
    output_cloud.reserve(total_points);

    std::vector<float> rs, gs, bs;
    size_t i = 0;
    int colored_count = 0;

    while (i < observations.size())
    {
        int current_id = observations[i].point_id;

        // Collect all observations for this point
        rs.clear();
        gs.clear();
        bs.clear();

        while (i < observations.size() && observations[i].point_id == current_id)
        {
            rs.push_back(observations[i].r);
            gs.push_back(observations[i].g);
            bs.push_back(observations[i].b);
            i++;
        }

        // Skip if point_id is out of range
        if (current_id < 0 || current_id >= total_points) continue;

        // Sort each channel independently (matches Python's np.sort + np.median)
        std::sort(rs.begin(), rs.end());
        std::sort(gs.begin(), gs.end());
        std::sort(bs.begin(), bs.end());

        // Compute median
        size_t n = rs.size();
        float med_r, med_g, med_b;
        if (n % 2 == 1)
        {
            med_r = rs[n / 2];
            med_g = gs[n / 2];
            med_b = bs[n / 2];
        }
        else
        {
            med_r = (rs[n / 2 - 1] + rs[n / 2]) * 0.5f;
            med_g = (gs[n / 2 - 1] + gs[n / 2]) * 0.5f;
            med_b = (bs[n / 2 - 1] + bs[n / 2]) * 0.5f;
        }

        // Add colored point to output
        pcl::PointXYZRGB pt;
        pt.x = cloud->points[current_id].x;
        pt.y = cloud->points[current_id].y;
        pt.z = cloud->points[current_id].z;
        pt.r = static_cast<uint8_t>(std::clamp(med_r * 255.0f, 0.0f, 255.0f));
        pt.g = static_cast<uint8_t>(std::clamp(med_g * 255.0f, 0.0f, 255.0f));
        pt.b = static_cast<uint8_t>(std::clamp(med_b * 255.0f, 0.0f, 255.0f));
        output_cloud.push_back(pt);
        colored_count++;

        if (colored_count % 50000 == 0)
            std::cout << "Processed " << colored_count << " unique points..." << std::endl;
    }

    output_cloud.width = output_cloud.size();
    output_cloud.height = 1;
    output_cloud.is_dense = true;

    std::cout << "\nTotal points in cloud: " << total_points << std::endl;
    std::cout << "Points with color: " << colored_count << std::endl;
    std::cout << "Points removed (no color): " << total_points - colored_count << std::endl;

    // ---- Save ----
    if (save_ascii)
        pcl::io::savePCDFileASCII(out_path, output_cloud);
    else
        pcl::io::savePCDFileBinary(out_path, output_cloud);

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "Saved " << colored_count << " colored points to '" << out_path
              << "' in " << std::fixed << std::setprecision(1) << elapsed << "s" << std::endl;

    return 0;
}