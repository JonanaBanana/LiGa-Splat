#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <iomanip>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// Converts float32 TIFF depth renders (metres, 0 = no data) produced by
// depth_renderer into 16-bit PNG inverse-depth maps for 3DGS depth
// regularisation, and writes depth_params.json.
//
// Why inverse depth?
//   camera_utils.py loads the PNG as:  raw = pixel / 65536
//   cameras.py then applies:           invdepth = raw * scale + offset
//   The renderer compares invdepth against its own rendered inverse depth.
//   So the stored value must be inverse depth (1/metres), not direct depth.
//
// Encoding: uint16 inverse depth
//   pixel = round((1 / depth_m) * 65536)   →  range [1, 65535]
//   pixel = 0                               →  no-data sentinel
//   scale = 1.0, offset = 0.0              →  invdepth = pixel / 65536
//   Min representable depth ≈ 1 m (pixel = 65535 → invdepth ≈ 1)
//   Max representable depth = any (pixel → 0 as depth → ∞)
//
// med_scale is NOT written — 3DGS computes and injects it at runtime
// (dataset_readers.py:166-170) from the median of all per-image scales.
//
// Usage:
//   prepare_depth_for_3dgs <data_folder>
//
// Output:
//   <data_folder>/distorted/depth/          — 16-bit PNG inverse-depth maps
//   <data_folder>/distorted/sparse/0/depth_params.json

namespace fs = std::filesystem;

// 3DGS loads depth as:  raw = pixel / 65536   (camera_utils.py)
// then applies:         invdepth = raw * scale + offset   (cameras.py)
// We store pixel = 65536 / depth_m, so raw = 1 / depth_m already.
// scale = 1.0 leaves it unchanged → invdepth = 1 / depth_m  ✓
static const double SCALE = 1.0;

// Strip file extension from a filename string
static std::string stem(const std::string & filename)
{
    auto pos = filename.rfind('.');
    return (pos == std::string::npos) ? filename : filename.substr(0, pos);
}

int main(int argc, char ** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <data_folder>" << std::endl;
        return 1;
    }

    fs::path data_folder = argv[1];
    fs::path depth_dir   = data_folder / "depth_renders";
    fs::path out_dir     = data_folder / "distorted/depth";
    fs::path params_path = data_folder / "distorted/sparse/0/depth_params.json";

    if (!fs::exists(depth_dir))
    {
        std::cerr << "Error: depth_renders/ not found in " << data_folder << std::endl;
        return 1;
    }

    // Collect .tiff files, sorted for deterministic output
    std::vector<fs::path> tiff_files;
    for (const auto & entry : fs::directory_iterator(depth_dir))
        if (entry.path().extension() == ".tiff")
            tiff_files.push_back(entry.path());

    if (tiff_files.empty())
    {
        std::cerr << "Error: no .tiff files found in " << depth_dir << std::endl;
        return 1;
    }
    std::sort(tiff_files.begin(), tiff_files.end());

    fs::create_directories(out_dir);

    std::ofstream json(params_path);
    if (!json.is_open())
    {
        std::cerr << "Error: could not open " << params_path << " for writing" << std::endl;
        return 1;
    }

    int converted = 0;
    json << "{\n";

    for (size_t i = 0; i < tiff_files.size(); ++i)
    {
        const fs::path & tiff_path = tiff_files[i];

        // Read float32 depth map (metres, 0 = no data)
        cv::Mat depth_m = cv::imread(tiff_path.string(),
                                     cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
        if (depth_m.empty())
        {
            std::cerr << "  Warning: could not read " << tiff_path.filename()
                      << ", skipping." << std::endl;
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
            std::cerr << "  Warning: " << tiff_path.filename()
                      << " has no valid depth pixels, skipping." << std::endl;
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
            std::cerr << "  Warning: failed to write " << out_path << std::endl;
            continue;
        }

        // JSON entry: scale = 1/65536 so that (pixel * scale) = invdepth in 1/m
        if (converted > 0) json << ",\n";
        json << "  \"" << out_stem << "\": "
             << std::fixed << std::setprecision(10)
             << "{\"scale\": " << SCALE << ", \"offset\": 0.0}";

        std::cout << "  " << tiff_path.filename().string()
                  << "  ->  " << out_stem << ".png"
                  << "  |  coverage " << std::fixed << std::setprecision(1)
                  << coverage << "%"
                  << "  |  depth [" << d_min << ", " << d_max << "] m"
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
