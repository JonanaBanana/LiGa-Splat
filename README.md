<p align="center">
  <img src="image_samples/scene_rgb.png"    width="32%"/>
  <img src="image_samples/registration.png" width="32%"/>
  <img src="image_samples/depth_render.png" width="32%"/>
</p>

# LiGa Splat - LiDAR-Guided 3D Gaussian Splatting Pipeline
A ROS2 pipeline for capturing LiDAR + Odometry + Camera data and converting it into the COLMAP format required to train a [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) model.
This pipeline has the advantage of not relying on visual features for pose extraction and sparse point cloud generation, and can thus work in environments not suitable for the regular COLMAP + 3DGS pipeline. Such environments could be sparse environments like wind turbine scans, or low texture homogenous environments like freighter hulls or tunnels.

With accurate input poses, point clouds, and camera intrinsics, the reconstruction results are very good, both photometrically and geometrically.

As an additional enhancement for sparse scenes with a lot of free space, it is possible to add a background sphere to initialize free space (floating) points on a background. This will further improve sparse environment reconstruction, by providing a clear separation between foreground and background.

This work is a successor to [MESSER for 3DGS](https://github.com/JonanaBanana/MESSER_for_3DGS). It converts the code to C++ and improves upon the pipeline by adding more robust behavior and faster computation with multi-thread processing.

This package has been tested with Ubuntu 22.04 and ROS2 Humble. The package includes a small sample dataset that can be used for testing (~22MB compressed, ~270MB extracted).

## Dependencies

- ROS2 (Humble or later)
- PCL, OpenCV, Eigen3, yaml-cpp
- OpenMP (for parallelised utilities)

```bash
sudo apt install ros-$ROS_DISTRO-pcl-conversions ros-$ROS_DISTRO-cv-bridge \
     libyaml-cpp-dev libopencv-dev libeigen3-dev
```
If you only need the utility code and dont want to install ROS2, you can skip installing ROS2 and ROS2 packages, and simply run the scripts in the utils/ folder as standalone scripts. They do not depend on ROS2.

---

## Build

```bash
cd ~/ros2_ws
colcon build --packages-select liga_splat
source install/setup.bash
```

Again, skip this part if you do not want to use ROS2.
---

## Test (No ROS2)
To quickly test if you installation works, you can try to run the full phase 2 pipeline (see phase 2 below). Simply extract the example_dataset.tar.xz file and run the full_pipeline.sh file with the dataset_path as argument.
```bash
cd ~/ros2_ws/src/liga_splat
tar -xvf example_dataset.tar.xz
./shell_scripts/full_pipeline.sh example_dataset/
```
If everything successfully runs, your installation works, and you can even directly train a 3DGS reconstruction on the results using any 3DGS reconstructor, since the structure is mostly identical to a COLMAP output.

## Overview

The pipeline has two phases:

**Phase 1 — Data capture (ROS2 nodes)**
Subscribe to live sensor topics, accumulate and downsample the point cloud in real time, and save images, odometry, and the final point cloud to disk.

**Phase 2 — Offline processing (standalone utilities)**
Process the saved data into the COLMAP format that 3DGS training expects.

```
Sensor topics
    │
    ▼
[data_saver_launch]          ← Phase 1 (ROS2)
    │  pcd/input.pcd
    │  distorted/images/
    │  timestamps/image.csv
    │  timestamps/odom.csv
    ▼
[pose_estimator]             ← Phase 2 (offline utilities)
    │  poses.csv
    ▼
[registration]
    │  pcd/downsampled.pcd
    │  color_registration.csv
    ▼
[reconstruction]
    │  pcd/reconstructed.pcd
    ▼
[export_colmap]
    │  distorted/sparse/0/
    │    cameras.{bin,txt}
    │    images.{bin,txt}
    │    points3D.{bin,txt}
    ▼
[depth_renderer]
    │  distorted/depth/*.png
    │  distorted/sparse/0/depth_params.json
    ▼
3DGS training
```

---

## Configuration

The pipeline uses two configuration files.

---

### `config/launch_config.cfg` — Phase 1 (ROS2 launch)

Edit this before running the ROS2 launch files. It controls topic names, output paths, and real-time point cloud management.

| Key | Default | Description |
|-----|---------|-------------|
| `output_dir` | — | Root folder where captured data (images, point cloud, timestamps) is saved |
| `bag_output_dir` | — | Folder for rosbag recordings (used by `record_bag_launch`) |
| `lidar_topic` | `/isaacsim/lidar` | ROS topic for incoming `PointCloud2` messages |
| `odom_topic` | `/isaacsim/odom` | ROS topic for incoming `Odometry` messages |
| `image_topic` | `/isaacsim/rgb` | ROS topic for incoming `Image` messages |
| `image_save_interval` | `10` | Save every Nth image frame to disk |
| `image_prefix` | `frame` | Filename prefix for saved images (e.g. `frame_0001.png`) |
| `odom_save_interval` | `1` | Save every Nth odometry message to `timestamps/odom.csv` |
| `leaf_size` | `0.03` | Voxel grid leaf size in metres used by both the accumulator and global processor |
| `max_path_length` | `10000` | Maximum number of odometry poses kept in the visualised path |
| `accumulator_max_points` | `500000` | The `PointCloudAccumulator` voxel-filters and forwards its buffer to the `GlobalProcessor` when the accumulated point count reaches this threshold |
| `accumulator_publish_interval` | `100` | Also forward the buffer every N LiDAR frames regardless of size (set to `0` to disable) |
| `global_downsample_interval` | `5` | The `GlobalProcessor` runs a periodic voxel downsample on the global map every N received batches |
| `global_max_points` | `5000000` | Hard cap on the global cloud; if exceeded after a periodic downsample a second pass at `2× leaf_size` is applied to bring it back under budget |
| `frame_id` | `World` | TF frame ID stamped on all published point cloud messages |

---

### `config.cfg` — Phase 2 (per-dataset, offline utilities)

This file lives inside each **data folder** (i.e. `<output_dir>/config.cfg`). It is read by every offline utility in Phase 2. Copy or create it manually for each dataset before running the utilities.

#### Pose estimation

| Key | Default | Description |
|-----|---------|-------------|
| `time_delay` | `0.0` | Seconds added to each image timestamp before matching against odometry. Use a positive value to compensate for camera trigger latency (delays the image timestamp so it aligns with a later odometry sample); use a negative value to advance it. |

#### Camera intrinsics

| Key | Default | Description |
|-----|---------|-------------|
| `focal_length` | `1000.0` | Camera focal length in pixels (assumes equal horizontal and vertical focal length, i.e. square pixels) |
| `image_width` | `1920` | Image width in pixels |
| `image_height` | `1080` | Image height in pixels |
| `principal_x` | `960.0` | Horizontal principal point (optical centre) in pixels |
| `principal_y` | `540.0` | Vertical principal point (optical centre) in pixels |

#### Point cloud filtering (registration step)

| Key | Default | Description |
|-----|---------|-------------|
| `voxel_size` | `0.1` | Voxel grid leaf size in metres used to downsample `pcd/input.pcd` before colour registration |
| `min_depth` | `1.0` | Minimum camera-frame depth (metres) for a point to be projected onto an image |
| `max_depth` | `200.0` | Maximum camera-frame depth (metres); points beyond this are ignored |
| `filter_outliers` | `true` | Apply Statistical Outlier Removal (SOR) before voxel downsampling. Also applied by `depth_renderer` before rendering. |
| `sor_neighbors` | `10` | Number of nearest neighbours used by the SOR filter |
| `sor_std_ratio` | `2.0` | Standard-deviation multiplier threshold for SOR; points further than `sor_std_ratio × σ` from the mean neighbour distance are removed |

#### Hidden point removal

| Key | Default | Description |
|-----|---------|-------------|
| `hpr_radius` | `20000.0` | Radius parameter for the Hidden Point Removal (HPR) convex-hull algorithm used during colour registration. The empirically good range for outdoor scenes is **15000–30000**: too small makes the convex hull numerically unstable; too large causes all flipped points to cluster at the same distance so depth discrimination is lost and occluded points bleed through. Scale with `max_depth` if you change it. |
| `depth_render_hpr_radius` | `40000.0` | HPR radius used by the `depth_renderer` utility. Intentionally higher than `hpr_radius` to keep more points in the depth map (denser coverage). |

#### Edge point preservation

| Key | Default | Description |
|-----|---------|-------------|
| `preserve_edge_points` | `true` | Detects Canny edges in each image, links them to 3D points from the pre-downsample cloud (via HPR), re-adds those points after voxel downsampling, then colour-registers the combined cloud. Prevents voxel downsampling from erasing high-frequency geometric detail at object boundaries. |
| `edge_canny_low` | `50.0` | Lower Canny hysteresis threshold. Decrease to detect more (weaker) edges. |
| `edge_canny_high` | `150.0` | Upper Canny hysteresis threshold. Decrease to detect more (weaker) edges. |
| `edge_dilation_px` | `2` | Dilate the edge mask by N pixels before matching 3D points. Widens the capture zone around each detected edge. |
| `edge_voxel_size` | `0.02` | Deduplication grid size for edge points in metres. Keep well below `voxel_size` so edges are preserved at finer resolution than the main cloud. |

#### Background sphere

| Key | Default | Description |
|-----|---------|-------------|
| `fill_background` | `true` | When `true`, a uniformly-sampled sphere of points is added around the scene centroid. Used by both `registration` (via `sphere_num_points`) and `depth_renderer` (via `depth_sphere_num_points`) — each with its own density. Useful for sparse scenes where the 3DGS optimiser would otherwise place large "background blobs" at arbitrary distances. |
| `sphere_radius` | `50.0` | Radius of the background sphere in metres (shared by registration and depth renderer) |
| `sphere_num_points` | `50000` | Number of sphere points added during colour registration |
| `depth_sphere_num_points` | `500000` | Number of sphere points added by `depth_renderer`. Much denser than for registration so that sky and void regions have depth coverage in every rendered depth map. |

#### Dense depth completion

| Key | Default | Description |
|-----|---------|-------------|
| `jbf_sigma_c` | `15.0` | Colour standard deviation (0–255 scale) for the colour-guided joint bilateral filter used by `depth_renderer --dense`. Lower values produce sharper depth edges but leave more unfilled voids in textureless or sky regions. Increase toward 30–50 if the dense map has too many holes; decrease if depth bleeds across colour boundaries. |

#### Camera–body transform

| Key | Default | Description |
|-----|---------|-------------|
| `trans_mat` | (see below) | Row-major 4×4 homogeneous transform matrix **from camera frame to body frame** (T_body_cam). Provide 16 space- or comma-separated values. The default converts from a z-forward / x-right / y-down camera frame to an x-forward / y-left / z-up body frame. |
| `poses_are_body_frame` | `true` | Set to `true` if the saved odometry poses are expressed in the body/robot frame (most common). Set to `false` if poses are already in the camera frame, in which case `trans_mat` is ignored. |

Default `trans_mat` (row-major):
```
[0,  0,  1, 0,
-1,  0,  0, 0,
 0, -1,  0, 0,
 0,  0,  0, 1]
```

#### File paths (relative to `data_folder`)

These rarely need to change unless you have reorganised the folder layout.

| Key | Default | Description |
|-----|---------|-------------|
| `pcd_file` | `pcd/input.pcd` | Raw captured point cloud written by Phase 1 |
| `downsampled_file` | `pcd/downsampled.pcd` | Voxel-downsampled + edge-preserved + optionally sphere-padded cloud written by `registration` |
| `reconstructed_file` | `pcd/reconstructed.pcd` | Coloured point cloud written by `reconstruction` |
| `poses_file` | `poses.csv` | Interpolated camera poses written by `pose_estimator` |
| `registration_file` | `color_registration.csv` | Per-point colour observations written by `registration` |
| `pose_timestamps_file` | `timestamps/odom.csv` | Odometry timestamps written by Phase 1 |
| `image_timestamps_file` | `timestamps/image.csv` | Image timestamps written by Phase 1 |
| `images_dir` | `distorted/images` | Directory containing the saved camera frames |
| `depth_dir` | `depth_renders` | Directory where `depth_renderer` writes float32 TIFF depth maps |

---

## Phase 1 — Data Capture

### Option A — Record a rosbag first, play back later

```bash
ros2 launch liga_splat record_bag_launch.py
```

Records LiDAR, camera, odometry, and TF to a timestamped bag in `bag_output_dir`.  
Split size is 1 GB per bag file.

### Option B — Save data directly from live topics

```bash
ros2 launch liga_splat data_saver_launch.py
```

Starts five composable nodes in a single container:

| Node | Role |
|------|------|
| `PointCloudAccumulator` | Buffers incoming LiDAR scans; voxel-filters and forwards when the point count exceeds `accumulator_max_points` (or every `accumulator_publish_interval` frames) |
| `GlobalProcessor` | Maintains a persistent global cloud; periodically downsamples it and saves it as `pcd/input.pcd` |
| `PathPublisher` | Publishes the odometry path for RViz visualisation |
| `ImageSaver` | Saves images and writes `timestamps/image.csv` |
| `OdomSaver` | Saves odometry poses to `timestamps/odom.csv` |

RViz opens automatically with a pre-configured layout.

When capture is complete, kill the launch (Ctrl-C). The global cloud is saved automatically to `<output_dir>/pcd/input.pcd`.

**Output layout after capture:**
```
<output_dir>/
  pcd/input.pcd
  distorted/images/frame_0001.png ...
  timestamps/image.csv
  timestamps/odom.csv
```

---

## Phase 2 — Offline Processing

### QUICK START
If you want to quickly run all offline postprocessing scripts in one sequence, you can use the shell script `shell_scripts/full_pipeline.sh` which also saves debug images.
```bash
/path/to/shell_scripts/full_pipeline.sh <data_folder>
``` 

### MANUAL PROCESSING
All utilities are run as:

```bash
ros2 run liga_splat <utility> <data_folder> [options]
```

where `<data_folder>` is the `output_dir` you set in `launch_config.cfg`.

Each utility reads its parameters from `<data_folder>/config.cfg`.

---
> [!WARNING]
> **`config.cfg` must be present in your data folder before running any Phase 2 utility.**
>
> Copy the template and edit it to match your camera and scene:
> ```bash
> cp path/to/liga_splat/config/config.cfg <data_folder>/config.cfg
> ```
> At minimum, set the camera intrinsics (`focal_length`, `image_width`, `image_height`, `principal_x`, `principal_y`) and the camera-to-body transform (`trans_mat`).
> Without this file the utilities will fall back to built-in defaults and produce **incorrect results**.
---

---

### 1. pose_estimator

Matches each saved image to an interpolated odometry pose (linear position + SLERP orientation). A `time_delay` parameter compensates for camera trigger latency.

```bash
ros2 run liga_splat pose_estimator <data_folder>
```

**Output:** `<data_folder>/poses.csv`

---

### 2. registration

Projects the point cloud into each camera image to assign an RGB colour to every visible point. Runs in two passes: first detects edge pixels and links them to pre-downsample 3D points, then colour-registers the combined (downsampled + edge-preserved) cloud so all points receive stable multi-view median colours.

```bash
ros2 run liga_splat registration <data_folder> [--diag]
```

| Option | Description |
|--------|-------------|
| `--diag` | Save diagnostic images to `diagnostics/`: Canny edge masks in `edge_detection/` and a combined overlay (depth-coloured registered points + semi-transparent green edge points) in `registered_points/` |

**Output:**
- `pcd/downsampled.pcd` — geometry used for all downstream steps
- `color_registration.csv` — per-point colour observations

---

### 3. reconstruction

Fuses the multi-view colour observations into a single coloured point cloud. Each point's final colour is the per-channel median across all views, which suppresses shadows, highlights, and moving objects.

```bash
ros2 run liga_splat reconstruction <data_folder> [--ascii]
```

**Output:** `pcd/reconstructed.pcd`

---

### 4. export_colmap

Exports poses and point cloud in the COLMAP binary/text format required by 3DGS. Runs hidden-point removal per camera to build realistic 2D observation tracks.

```bash
ros2 run liga_splat export_colmap <data_folder>
```

**Output:** `distorted/sparse/0/{cameras,images,points3D}.{bin,txt}`

---

### 5. depth_renderer

Renders a depth map for each camera pose using the raw input point cloud (`pcd/input.pcd`). Applies SOR outlier removal and adds a dense background sphere (controlled by `depth_sphere_num_points`) before rendering, giving more complete depth coverage than the downsampled cloud. Converts results to 16-bit PNG inverse-depth maps and writes `depth_params.json` — the exact format expected by the depth-regularised 3DGS training script. Supports optional hidden-point removal and colour-guided dense completion.

```bash
ros2 run liga_splat depth_renderer <data_folder> [--no-hpr] [--dense] [--diag] [--save-tiff]
```

| Option | Description |
|--------|-------------|
| `--no-hpr` | Skip hidden-point removal (faster, less accurate) |
| `--dense` | Fill gaps using colour-guided joint bilateral filtering |
| `--diag` | Save false-colour visualisation PNGs to `diagnostics/depth_renders/` |
| `--save-tiff` | Also write intermediate float32 TIFF depth maps to `<depth_dir>/` (useful for debugging) |

**Output:**
- `distorted/depth/*.png` — 16-bit inverse-depth maps
- `distorted/sparse/0/depth_params.json`

The command also prints the recommended `train.py` invocation at the end.

---

### Visualisation utilities

```bash
# View any PCD file (color modes: rgb, z)
ros2 run liga_splat pcd_viewer <file.pcd> [rgb|z]

# View odometry path + camera frustums overlaid on the point cloud
ros2 run liga_splat pose_viewer <data_folder> [--no-pcd] [--frustum-scale <s>]
```

---

## Complete example
Here is a complete example including training with vanilla [3DGS](https://github.com/graphdeco-inria/gaussian-splatting)

```bash
# 1. Edit config
nano install/liga_splat/share/liga_splat/config/launch_config.cfg

# 2. Capture data
ros2 launch liga_splat data_saver_launch.py

# --- stop capture when done ---

DATA=~/dataset/my_scene

# 3. Place a config.cfg in your data folder (copy from config/config.cfg and edit)
cp install/liga_splat/share/liga_splat/config/config.cfg $DATA/config.cfg

# 4. Estimate poses
ros2 run liga_splat pose_estimator $DATA

# 5. Colour the point cloud
ros2 run liga_splat registration $DATA

# 6. Fuse colours
ros2 run liga_splat reconstruction $DATA

# 7. Export COLMAP
ros2 run liga_splat export_colmap $DATA

# 8. Render depth maps and prepare for 3DGS
ros2 run liga_splat depth_renderer $DATA --dense

# 9. Train 3DGS (example — adjust paths as needed)
python train.py \
    -s $DATA/distorted \
    -d $DATA/distorted/depth \
    -r 1
```

---

## Data folder layout (complete)

```
<data_folder>/
  config.cfg                          ← per-dataset parameters
  poses.csv                           ← interpolated camera poses
  color_registration.csv              ← per-point colour observations
  pcd/
    input.pcd                         ← raw captured cloud
    downsampled.pcd                   ← filtered + sphere-padded
    reconstructed.pcd                 ← coloured cloud
  distorted/
    images/                           ← saved camera frames
    depth/                            ← 16-bit PNG inverse-depth maps
    sparse/0/
      cameras.{bin,txt}
      images.{bin,txt}
      points3D.{bin,txt}
      depth_params.json
  timestamps/
    image.csv
    odom.csv
  depth_renders/                      ← float32 TIFF depth maps (only with --save-tiff)
  diagnostics/                        ← optional debug images (only with --diag)
```
