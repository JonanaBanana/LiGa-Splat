# LiGa Splat

A ROS2 pipeline for capturing LiDAR + camera data and converting it into the COLMAP format required to train a [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) model.

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
    │  depth_renders/*.tiff
    ▼
[prepare_depth_for_3dgs]
    │  distorted/depth/*.png
    │  distorted/sparse/0/depth_params.json
    ▼
3DGS training
```

---

## Dependencies

- ROS2 (Humble or later)
- PCL, OpenCV, Eigen3, yaml-cpp
- OpenMP (for parallelised utilities)

```bash
sudo apt install ros-$ROS_DISTRO-pcl-conversions ros-$ROS_DISTRO-cv-bridge \
     libyaml-cpp-dev libopencv-dev libeigen3-dev
```

---

## Build

```bash
cd ~/ros2_ws
colcon build --packages-select liga_splat
source install/setup.bash
```

---

## Configuration

All user-specific paths and tuning parameters live in one file:

```
config/launch_config.cfg
```

Edit this before running anything. Key fields:

| Key | Description |
|-----|-------------|
| `output_dir` | Root folder where captured data is saved |
| `bag_output_dir` | Folder for rosbag recordings |
| `lidar_topic` | ROS topic for incoming `PointCloud2` messages |
| `image_topic` | ROS topic for incoming `Image` messages |
| `odom_topic` | ROS topic for incoming `Odometry` messages |
| `leaf_size` | Voxel grid leaf size in metres (e.g. `0.03`) |
| `image_save_interval` | Save every Nth image frame |
| `accumulator_max_points` | Publish accumulated cloud after this many points |
| `global_max_points` | Hard cap on the global cloud before aggressive downsampling |

A `config.cfg` file also lives in each **data folder** (copied or created manually) and holds the per-dataset parameters used by the offline utilities (camera intrinsics, transform matrix, depth range, etc.).

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

All utilities are run as:

```bash
ros2 run liga_splat <utility> <data_folder> [options]
```

where `<data_folder>` is the `output_dir` you set in `launch_config.cfg`.

Each utility reads its parameters from `<data_folder>/config.cfg`.

---

### 1. pose_estimator

Matches each saved image to an interpolated odometry pose (linear position + SLERP orientation). A `time_delay` parameter compensates for camera trigger latency.

```bash
ros2 run liga_splat pose_estimator <data_folder>
```

**Output:** `<data_folder>/poses.csv`

---

### 2. registration

Projects the point cloud into each camera image to assign an RGB colour to every visible point. Outputs both a downsampled point cloud (with an optional background sphere added) and a per-point colour observation table.

```bash
ros2 run liga_splat registration <data_folder> [--diag]
```

| Option | Description |
|--------|-------------|
| `--diag` | Save false-colour depth overlay images to `diagnostics/color_registration/` |

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

Renders a float32 depth map (metres) for each camera pose using the downsampled point cloud. Supports optional hidden-point removal and colour-guided dense completion.

```bash
ros2 run liga_splat depth_renderer <data_folder> [--no-hpr] [--dense] [--diag]
```

| Option | Description |
|--------|-------------|
| `--no-hpr` | Skip hidden-point removal (faster, less accurate) |
| `--dense` | Fill gaps using colour-guided joint bilateral filtering |
| `--diag` | Save false-colour visualisation PNGs to `diagnostics/depth_renders/` |

**Output:** `<depth_dir>/*.tiff`  (default: `depth_renders/`)

---

### 6. prepare_depth_for_3dgs

Converts the float32 TIFF depth maps into 16-bit PNG inverse-depth maps and writes `depth_params.json` — the exact format expected by the depth-regularised 3DGS training script.

```bash
ros2 run liga_splat prepare_depth_for_3dgs <data_folder>
```

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

```bash
# 1. Edit config
nano install/liga_splat/share/liga_splat/config/launch_config.cfg

# 2. Capture data
ros2 launch liga_splat data_saver_launch.py

# --- stop capture when done ---

DATA=~/dataset/my_scene

# 3. Estimate poses
ros2 run liga_splat pose_estimator $DATA

# 4. Colour the point cloud
ros2 run liga_splat registration $DATA

# 5. Fuse colours
ros2 run liga_splat reconstruction $DATA

# 6. Export COLMAP
ros2 run liga_splat export_colmap $DATA

# 7. Render depth maps
ros2 run liga_splat depth_renderer $DATA --dense

# 8. Convert for 3DGS
ros2 run liga_splat prepare_depth_for_3dgs $DATA

# 9. Train 3DGS (example — adjust paths as needed)
CUDA_VISIBLE_DEVICES=0 python train.py \
    -s $DATA/distorted \
    --data_factor 1 \
    -d $DATA/distorted/depth \
    --depth_params $DATA/distorted/sparse/0/depth_params.json
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
  depth_renders/                      ← float32 TIFF depth maps
  diagnostics/                        ← optional debug images
```
