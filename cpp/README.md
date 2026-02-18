# Hailo Inference Pipeline — C++ Implementation

This directory contains a C++ port of the [Python inference pipeline](../README.md).
The feature set is equivalent unless noted below.

## Differences from the Python Implementation

| Feature                        | Python | C++  |
| ------------------------------ | ------ | ---- |
| Synchronous inference          | Yes    | Yes  |
| Asynchronous inference         | Yes    | Yes  |
| Callback mode                  | Yes    | Yes  |
| Performance profiling (console)| Yes    | Yes  |
| Perfetto trace export          | Yes    | Yes  |
| Matplotlib charts              | Yes    | No   |

## Requirements

### Software Dependencies

- **C++20** or later
- **CMake** 3.16 or later
- **OpenCV** (system package)
- **HailoRT** CMake package (installed with the Hailo SDK)
- **nlohmann/json** — fetched automatically via CMake `FetchContent` if not found on the system

### Install system dependencies

```bash
sudo apt install cmake libopencv-dev
```

Install HailoRT following the official Hailo documentation, then verify the device is accessible:

```bash
hailortcli fw-control identify
```

## Build

```bash
cd cpp
cmake -B build
cmake --build build -j$(nproc)
```

The binary is placed at `build/hailo_inference_pipeline`.

## Usage

The command-line interface mirrors the Python version exactly.

### Basic Usage

#### Image Classification

```bash
./build/hailo_inference_pipeline image.jpg \
    --net ./hefs/resnet_v1_50.hef \
    --postprocess classification
```

#### Object Detection (Video)

```bash
./build/hailo_inference_pipeline video.mp4 \
    --net ./hefs/yolov8n.hef \
    --postprocess nms_on_host \
    --config ./configs/yolov8.json
```

#### Asynchronous Inference with Profiling

```bash
./build/hailo_inference_pipeline video.mp4 \
    --net ./hefs/resnet_v1_50.hef \
    --postprocess classification \
    --profile
```

#### Performance Profiling with Perfetto Trace Export

```bash
./build/hailo_inference_pipeline video.mp4 \
    --net ./hefs/resnet_v1_50.hef \
    --postprocess classification \
    --profile \
    --trace trace.json
```

#### Synchronous Inference

```bash
./build/hailo_inference_pipeline image.jpg \
    --net ./hefs/resnet_v1_50.hef \
    --postprocess classification \
    --synchronous
```

### Command Line Arguments

| Argument        | Short | Type       | Default                   | Description                                                             |
| --------------- | ----- | ---------- | ------------------------- | ----------------------------------------------------------------------- |
| `input`         | -     | positional | required                  | Path to image or video file to process                                  |
| `--net`         | `-n`  | string     | `./hefs/resnet_v1_50.hef` | Path to HEF model file                                                  |
| `--postprocess` | `-p`  | choice     | `classification`          | Post-processing type: `classification`, `nms_on_host`, `palm_detection` |
| `--config`      | `-c`  | string     | auto-detected             | Path to custom JSON configuration file for post-processing              |
| `--synchronous` | `-s`  | flag       | false                     | Use synchronous inference                                               |
| `--callback`    | -     | flag       | false                     | Use callback mode with async inference                                  |
| `--batch-size`  | `-b`  | integer    | 1                         | Number of images per batch                                              |
| `--profile`     | -     | flag       | false                     | Enable performance profiling                                            |
| `--trace`       | -     | string     | none                      | Export profiling data to Perfetto trace JSON file (requires `--profile`)|
