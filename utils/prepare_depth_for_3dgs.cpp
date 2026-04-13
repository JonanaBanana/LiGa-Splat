#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <sstream>

#include <omp.h>
#include <yaml-cpp/yaml.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// Converts float32 TIFF depth renders (metres, 0 = no data) produced by
// depth_renderer into 16-bit PNG inverse-depth maps for 3DGS depth
// regularisation, and writes depth_params.json.
//
// Usage:
//   prepare_depth_for_3dgs <data_folder>
//
// Output:
//   <data_folder>/distorted/depth/          — 16-bit PNG inverse-depth maps
//   <data_folder>/distorted/sparse/0/depth_params.json

namespace fs = std::filesystem;
static const double SCALE = 1.0;

// Strip file extension from a filename string
static std::string stem(const std::string & filename)
{
    auto pos = filename.rfind('.');
    return (pos == std::string::npos) ? filename : filename.substr(0, pos);
}

struct DepthResult
{
    bool        success  = false;
    std::string out_stem;
    float       d_min    = 0.0f;
    float       d_max    = 0.0f;
    double      coverage = 0.0;
};

int main(int argc, char ** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <data_folder>" << std::endl;
        return 1;
    }

    fs::path data_folder = argv[1];

    std::string depth_dir_name = "depth_renders";
    try {
        YAML::Node cfg = YAML::LoadFile((data_folder / "config.cfg").string());
        if (cfg["depth_dir"])
            depth_dir_name = cfg["depth_dir"].as<std::string>();
    } catch (const YAML::Exception &) {}

    fs::path depth_dir   = data_folder / depth_dir_name;
    fs::path out_dir     = data_folder / "distorted/depth";
    fs::path params_path = data_folder / "distorted/sparse/0/depth_params.json";

    if (!fs::exists(depth_dir))
    {
        std::cerr << "\033[31m" << "Error: '" << depth_dir << "' not found" << "\033[0m" << std::endl;
        return 1;
    }

    // Collect .tiff files, sorted for deterministic output
    std::vector<fs::path> tiff_files;
    for (const auto & entry : fs::directory_iterator(depth_dir))
        if (entry.path().extension() == ".tiff")
            tiff_files.push_back(entry.path());

    if (tiff_files.empty())
    {
        std::cerr << "\033[31m" << "Error: no .tiff files found in " << depth_dir << "\033[0m" << std::endl;
        return 1;
    }
    std::sort(tiff_files.begin(), tiff_files.end());

    fs::create_directories(out_dir);

    std::ofstream json(params_path);
    if (!json.is_open())
    {
        std::cerr << "\033[31m" << "Error: could not open " << params_path << " for writing" << "\033[0m" << std::endl;
        return 1;
    }

    std::vector<DepthResult> results(tiff_files.size());

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < tiff_files.size(); ++i)
    {
        const fs::path & tiff_path = tiff_files[i];

        cv::Mat depth_m = cv::imread(tiff_path.string(),
                                     cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
        if (depth_m.empty())
        {
            std::ostringstream msg;
            msg << "  Warning: could not read " << tiff_path.filename().string() << ", skipping.\n";
            #pragma omp critical
            std::cerr << msg.str();
            continue;
        }

        // Count valid pixels and compute depth range for diagnostics
        int n_valid = 0;
        float d_min = std::numeric_limits<float>::max();
        float d_max = 0.0f;
        for (int r = 0; r < depth_m.rows; r++)
            for (int c = 0; c < depth_m.cols; c++)
            {
                float d = depth_m.at<float>(r, c);
                if (d > 0.0f)
                {
                    n_valid++;
                    d_min = std::min(d_min, d);
                    d_max = std::max(d_max, d);
                }
            }

        if (n_valid == 0)
        {
            std::ostringstream msg;
            msg << "  Warning: " << tiff_path.filename().string()
                << " has no valid depth pixels, skipping.\n";
            #pragma omp critical
            std::cerr << msg.str();
            continue;
        }

        double coverage = 100.0 * n_valid / (depth_m.rows * depth_m.cols);

        // Convert to uint16 inverse depth:
        //   pixel = round(1/depth_m * 65536), clamped to [0, 65535]
        //   pixel = 0 reserved for no-data (depth_m == 0)
        cv::Mat depth_u16(depth_m.rows, depth_m.cols, CV_16UC1);
        for (int r = 0; r < depth_m.rows; r++)
            for (int c = 0; c < depth_m.cols; c++)
            {
                float d = depth_m.at<float>(r, c);
                if (d <= 0.0f)
                {
                    depth_u16.at<uint16_t>(r, c) = 0;
                }
                else
                {
                    double inv = 65536.0 / static_cast<double>(d);
                    depth_u16.at<uint16_t>(r, c) =
                        static_cast<uint16_t>(std::min(inv + 0.5, 65535.0));
                }
            }

        // Save as 16-bit PNG
        std::string out_stem = stem(tiff_path.filename().string());
        fs::path out_path    = out_dir / (out_stem + ".png");

        if (!cv::imwrite(out_path.string(), depth_u16))
        {
            std::ostringstream msg;
            msg << "  Warning: failed to write " << out_path.string() << "\n";
            #pragma omp critical
            std::cerr << msg.str();
            continue;
        }

        results[i] = {true, out_stem, d_min, d_max, coverage};
    }

    // Sequential pass: write JSON entries in sorted order and print progress
    int converted = 0;
    json << "{\n";
    for (size_t i = 0; i < tiff_files.size(); ++i)
    {
        const auto & res = results[i];
        if (!res.success) continue;

        if (converted > 0) json << ",\n";
        json << "  \"" << res.out_stem << "\": "
             << std::fixed << std::setprecision(10)
             << "{\"scale\": " << SCALE << ", \"offset\": 0.0}";

        std::cout << "  " << tiff_files[i].filename().string()
                  << "  ->  " << res.out_stem << ".png"
                  << "  |  coverage " << std::fixed << std::setprecision(1)
                  << res.coverage << "%"
                  << "  |  depth [" << res.d_min << ", " << res.d_max << "] m"
                  << std::endl;

        converted++;
    }
    json << "\n}\n";
    json.close();

    if (converted == 0)
    {
        std::cerr << "No depth maps were converted." << std::endl;
        return 1;
    }

    std::cout << "\nConverted " << converted << " depth maps  ->  " << out_dir << std::endl;
    std::cout << "Wrote " << params_path << std::endl;
    std::cout << "\n3DGS training command:\n"
              << "  python train.py \\\n"
              << "      -s <colmap_dir> \\\n"
              << "      -d " << out_dir.string() << " \\\n"
              << "      --depth_params " << params_path.string() << " \\\n"
              << "      <other args>" << std::endl;

    return 0;
}
