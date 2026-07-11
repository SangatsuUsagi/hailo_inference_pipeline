# Hailo Inference Pipeline

![Apache 2.0 License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

![](./images/out_27928552_s.png)![](./images/out_6908_1280x720.gif)

![](./images/out_yolov11.png)![](./images/out_yolov11.gif)

A Python pipeline for running deep learning inference on Hailo AI accelerators, with synchronous and asynchronous inference modes, multi-threaded video capture/display, and built-in performance profiling.

It handles both single images and video streams, and ships with three post-processing pipelines: image classification (Top-N with ImageNet-style labels), YOLOv8-style object detection (NMS on host), and MediaPipe-style palm detection.

## Features

- **Inference modes**: synchronous and asynchronous (default), plus an optional callback mode for the async path
- **Hailo-specific exception handling**: dedicated exception types per failure mode, each with a defined recovery action (see [Exception Handling](#exception-handling))
- **Performance profiling**: per-stage timing statistics, matplotlib charts, and Perfetto trace export across up to three threads (main, capture, display)
- **Multi-threaded video I/O**: frame capture and display run on separate threads from inference, for higher throughput on video streams
- **Context manager support**: `InferPipeline` handles Hailo device setup/teardown automatically

## Requirements

### Hardware

- Hailo AI accelerator (Hailo-8, Hailo-8L, or compatible device)
- PCIe interface connection

### Software

- Python 3.10 or higher
- HailoRT and the `hailo_platform` Python package - installed manually from the Hailo-provided wheel for your HailoRT version (not available on PyPI)
- `numpy`, `opencv-python` (pinned to `3.4.18.65`), `matplotlib` - see `requirements.txt` / `pyproject.toml`

## Installation

1. Install HailoRT and the `hailo_platform` wheel following the official Hailo documentation.
2. Install the remaining Python dependencies:

```bash
pip install -r requirements.txt
# or, with uv:
uv sync
```

   `numpy`'s version must be compatible with your installed HailoRT build, which is why `opencv-python` is pinned to the OpenCV 3.x line (`3.4.18.65`) - it's the last major version built against the older numpy versions HailoRT expects. If `import hailo_platform` fails with a numpy-related error after this step, check the numpy version HailoRT was built against and pin `numpy` to match.

3. Confirm the Hailo device is connected and recognized:

```bash
hailortcli fw-control identify
```

4. Get a `.hef` model file for your target model. This repository doesn't ship any (`hefs/` is empty by default) - convert your own model with the Hailo Dataflow Compiler, or see [About Jupyter Notebooks](#about-jupyter-notebooks) for building one from a MediaPipe TFLite palm-detection model.

## Usage

### Basic Usage

#### Image Classification

```bash
python inference.py image.jpg \
    --net ./hefs/resnet_v1_50.hef \
    --postprocess classification
```

#### Object Detection (Video)

```bash
python inference.py video.mp4 \
    --net ./hefs/yolov8n.hef \
    --postprocess nms_on_host \
    --config ./configs/yolov8.json
```

#### Asynchronous Inference with Profiling

```bash
python inference.py video.mp4 \
    --net ./hefs/resnet_v1_50.hef \
    --postprocess classification \
    --profile
```

#### Performance Profiling with Perfetto Trace Export

```bash
python inference.py video.mp4 \
    --net ./hefs/resnet_v1_50.hef \
    --postprocess classification \
    --profile \
    --trace trace.json
```

#### Synchronous Inference

```bash
python inference.py image.jpg \
    --net ./hefs/resnet_v1_50.hef \
    --postprocess classification \
    --synchronous
```

### Command Line Arguments

| Argument        | Short | Type       | Default                   | Description                                                              |
| --------------- | ----- | ---------- | ------------------------- | ------------------------------------------------------------------------ |
| `images`        | -     | positional | required                  | Path to image or video file to process                                   |
| `--net`         | `-n`  | string     | `./hefs/resnet_v1_50.hef` | Path to HEF model file                                                   |
| `--postprocess` | `-p`  | choice     | `classification`          | Post-processing type: `classification`, `nms_on_host`, `palm_detection`  |
| `--config`      | `-c`  | string     | auto-detected             | Path to custom JSON configuration file for post-processing               |
| `--synchronous` | `-s`  | flag       | false                     | Use synchronous inference on HRT 4.X (default: asynchronous)             |
| `--callback`    | -     | flag       | false                     | Use callback mode with async inference                                   |
| `--batch-size`  | `-b`  | integer    | 1                         | Number of images per batch                                               |
| `--profile`     | -     | flag       | false                     | Enable performance profiling with visualization                          |
| `--trace`       | -     | string     | none                      | Export profiling data to Perfetto trace JSON file (requires `--profile`) |

### Configuration Files

The pipeline uses JSON configuration files for post-processing, and each post-processor expects a different structure:

- **Classification**: `./configs/class_names_imagenet.json` - a flat `{"<class_id>": "<label>"}` map
- **YOLOv8 Detection**: `./configs/yolov8.json` - preprocessing params plus a class-id/label map
- **Palm Detection**: `./configs/palm_detection_full.json` - MediaPipe anchor-generation and decoding parameters

For example, `yolov8.json` looks like this (trimmed):

```json
[
  {
    "preprocessing": {
      "network_type": "detection",
      "input_shape": [640, 640, 3]
    }
  },
  {
    "0": "Person",
    "1": "Bicycle",
    "2": "Car",
    "...": "...",
    "79": "Toothbrush"
  }
]
```

## Synchronous vs Asynchronous Inference

### Synchronous Inference (`--synchronous`)

**Characteristics:**

- Blocking operations: each inference waits for completion before proceeding
- Simpler execution flow with sequential processing
- Lower latency for single-frame processing
- Easier to debug and understand

**Best For:**

- Single image processing
- Batch processing where order matters
- Development and debugging
- Applications where simplicity is preferred over throughput

**Performance:**

```
Frame → Preprocess → [Inference] → Postprocess → Display
         (wait)                       (wait)
```

### Asynchronous Inference (default)

**Characteristics:**

- Non-blocking operations: submit inference and continue processing
- Pipelined execution with overlapped operations
- Higher throughput for video streams
- More complex error handling

**Best For:**

- Real-time video processing
- High-throughput applications
- Production deployments
- Applications requiring maximum FPS

**Performance:**

```
Frame 1 → Preprocess → Submit Inference
Frame 2 → Preprocess → Submit Inference ────────────┐
Frame 3 → Preprocess → Submit Inference ──────┐     │
                                              │     │
                                Postprocess Frame 1 & Wait Frame 2 inference
                                              │
                                Postprocess Frame 2 & Wait Frame 3 inference
```

### Example performance measurements

The following table shows measured performance on a 3019-frame video input for different models, comparing asynchronous and synchronous inference modes. (Measured with RaspberryPi5 + Hailo AI-HAT/Hailo-8)

| Model               | Asynchronous FPS | Synchronous FPS | Post-processing time (ms) |
| ------------------- | ---------------- | --------------- | ------------------------- |
| Palm_detection_full | 51.66 fps        | 62.02 fps       | 2.72 ms                   |
| Yolov8n             | 74.33 fps        | 65.96 fps       | 0.560 ms                  |
| resnet_v1_50        | 72.17 fps        | 68.81 fps       | 0.497 ms                  |

### Callback Mode (`--callback`)

Available only in asynchronous mode, callback mode processes results immediately upon completion:

```bash
python inference.py video.mp4 \
    --net ./hefs/yolov8n.hef \
    --postprocess nms_on_host \
    --callback
```

**Benefits:**

- Reduced latency by processing results immediately
- Better suited for real-time applications
- Automatic result handling

## Performance Profiling

Enable profiling with `--profile` to get detailed timing statistics:

```bash
python inference.py video.mp4 --net ./hefs/resnet_v1_50.hef --profile
```

### Profiling Output

**Console Statistics:**

```
======================================================================================
PERFORMANCE PROFILING RESULTS
======================================================================================
Checkpoint                     Count      Min(ms)      Max(ms)     Mean(ms)   Var(ms²)
--------------------------------------------------------------------------------
1_frame_read                     300        0.125        2.456        0.234      0.012
2_preprocessing                  300        1.234        3.456        1.567      0.045
3_inference_submit               300        0.089        0.234        0.123      0.003
4_inference_wait                 300       15.234       18.456       16.234      1.234
5_postprocessing                 300        2.345        4.567        2.789      0.234
6_display_queue                  300        0.012        0.456        0.034      0.001
total_frame_time                 300       19.567       25.678       21.234      3.456
======================================================================================

Average Frame Processing Time: 21.234 ms
Average FPS (from frame time): 47.09
======================================================================================
```

**Visual Charts:**

- **Stacked Time Chart**: Shows cumulative timing for each pipeline stage across frames

![](./images/Pipeline_Stage_Timing_Stacked_Time_Chart.png)

- **Detailed Timing Chart**: Individual timing plots for each stage with mean lines

![](./images/Detailed_Pipeline_Timing_Analysis.png)

### Perfetto Trace Export

Export profiling data to Perfetto's trace event format for advanced timeline visualization:

![](./images/Perfetto_UI.png)

```bash
python inference.py video.mp4 \
    --net ./hefs/resnet_v1_50.hef \
    --profile \
    --trace trace.json
```

**Output:**

```
Perfetto trace exported to: trace.json
Total frames: 300
Total events: 1803
Threads: Main Thread, Display Thread

Visualize the trace at:
  - Chrome: chrome://tracing
  - Perfetto UI: https://ui.perfetto.dev
```

**Viewing the Trace:**

1. **Chrome Tracing Tool**:
   - Open Chrome browser
   - Navigate to `chrome://tracing`
   - Click "Load" and select your trace file
   - Use WASD keys to navigate the timeline

2. **Perfetto UI** (recommended):
   - Visit https://ui.perfetto.dev
   - Click "Open trace file" and select your trace file
   - Explore the interactive timeline with advanced features

**Multi-Thread Visualization:**

The trace shows events across **up to three separate thread rows** (depending on configuration):

**Main Thread (tid: 1)**

- Frame processing pipeline: `frame_read`, `preprocessing`, `inference_submit`, `inference_wait`, `postprocessing`
- Frame markers showing processing boundaries
- Queue operations: `display_queue` (sending frames to display thread)

**Capture Thread (tid: 3)** (video mode only)

- Asynchronous frame capture running in parallel
- `read`: Time to read frame from video source (`cv2.VideoCapture.read`)
- `queue_put`: Time to queue frame for main thread processing

**Display Thread (tid: 2)** (video mode only)

- Asynchronous display operations running in parallel
- `queue_wait`: Time waiting for frames from main thread
- `display`: OpenCV rendering time (`cv2.imshow`)
- `key_check`: Keyboard input polling time

This separation lets you see true parallelism across threads, spot I/O bottlenecks (slow capture-thread reads), find processing bottlenecks (main or display thread lagging), and check whether threads are starved or backing up on their queues.

**What You'll See:**

```
┌─────────────────────────────────────────────────────┐
│ Hailo Inference Pipeline                            │
├─────────────────────────────────────────────────────┤
│ Main Thread                                         │
│ ┌─┬────┬────┬───┬────────┬────┬──┐                  │
│ │●│read│prep│inf│inf_wait│post│q │ Frame 1          │
│ └─┴────┴────┴───┴────────┴────┴──┘                  │
│   ┌─┬────┬────┬───┬────────┬────┬──┐                │
│   │●│read│prep│inf│inf_wait│post│q │ Frame 2        │
│   └─┴────┴────┴───┴────────┴────┴──┘                │
├─────────────────────────────────────────────────────┤
│ Capture Thread                                      │
│ ┌────┬──┐                                           │
│ │read│qp│ Frame 1                                   │
│ └────┴──┘                                           │
│     ┌────┬──┐                                       │
│     │read│qp│ Frame 2                               │
│     └────┴──┘                                       │
├─────────────────────────────────────────────────────┤
│ Display Thread                                      │
│   ┌────┬───────┬────┐                               │
│   │wait│display│ key│ Frame 1                       │
│   └────┴───────┴────┘                               │
│        ┌────┬───────┬────┐                          │
│        │wait│display│ key│ Frame 2                  │
│        └────┴───────┴────┘                          │
└─────────────────────────────────────────────────────┘
```

Each checkpoint appears as a duration bar on the timeline, with frame markers separating each frame's events - useful for spotting timing variance and bottlenecks across frames and threads.

## Exception Handling

The pipeline defines Hailo-specific exception types, each with a suggested recovery action (see [Limitations](#limitations) for a caveat on this):

### Exception Types

| Exception                | Description                    | Recovery Action      |
| ------------------------ | ------------------------------ | -------------------- |
| `InferenceTimeoutError`  | Inference operation timed out  | Retry frame or skip  |
| `InferenceSubmitError`   | Failed to submit inference job | Break (device error) |
| `InferenceWaitError`     | Failed to retrieve results     | Retry or skip frame  |
| `InferencePipelineError` | Synchronous inference failed   | Break (fatal)        |

### Error Recovery Example

```python
try:
    outputs = infer.inference(dataset)
    results = infer.wait_and_get_output()
except InferenceTimeoutError as e:
    # Timeout is recoverable - retry or skip frame
    print(f"Timeout on frame {frame_count}: {e}")
    continue
except InferenceSubmitError as e:
    # Device error - stop processing
    print(f"Device error: {e}")
    break
```

## Video Controls

When processing video files:

- **Press 'q'**: Quit the application
- **Close window**: Stop processing

## Output

### Classification Output

```
Top 3 predictions:
1. Egyptian cat (0.8756)
2. Tabby cat (0.0234)
3. Tiger cat (0.0123)
```

### Detection Output

- Bounding boxes drawn on frame
- Class labels with confidence scores
- Real-time FPS counter

### Performance Summary (video mode only)

```
================================================================================
BASIC PERFORMANCE SUMMARY
================================================================================
Total execution time: 12.345678 seconds
Total frames processed: 300
Overall throughput: 24.31 FPS
================================================================================
```

## Troubleshooting

### Common Issues

**1. "No input streams found in the model"**

- Verify HEF file path is correct
- Ensure HEF file is compatible with your Hailo device

**2. "Inference device not ready: timeout"**

- Check Hailo device connection: `hailortcli fw-control identify`
- Ensure no other processes are using the device

**3. "Failed to initialize InferPipeline"**

- Install Hailo SDK properly
- Check device permissions: `sudo chmod 666 /dev/hailo0`
- Verify PCIe connection

**4. Low FPS performance**

- Use asynchronous mode (default)
- Disable profiling for production
- Increase `max_queue_size` for `DisplayThread`/`FrameReaderThread` (hardcoded in `inference.py`, not a CLI flag - requires editing the source)
- Use hardware-accelerated video decoding

### Debug Mode

For detailed debugging, the pipeline prints comprehensive error messages:

```python
# Each exception includes:
# - Error type and description
# - Frame number where error occurred
# - Suggested recovery action
# - Original exception chain
```

## Architecture

### Pipeline Flow

**Single Image Mode:**

```
Input (Image)
    ↓
Frame Reading (synchronous)
    ↓
Preprocessing (resize, pad, color conversion)
    ↓
Inference (Hailo device)
    ↓
Postprocessing (NMS, classification, etc.)
    ↓
Display (synchronous)
```

**Video Mode (Multi-threaded):**

```
Capture Thread:          Main Thread:              Display Thread:
Frame Reading      →     Preprocessing       →     Frame Display
(FrameReaderThread)      ↓                         (DisplayThread)
     ↓                   Inference                       ↓
  Queue Buffer           (Hailo device)             Queue Buffer
                         ↓
                    Postprocessing
                         ↓
                    Performance Profiling
                       (optional)
```

**Key Features:**

- **Capture Thread**: Asynchronously reads frames from video file
- **Main Thread**: Processes frames through inference pipeline
- **Display Thread**: Asynchronously displays results
- **All three threads run in parallel for maximum throughput**

### Class Structure

- **`InferPipeline`**: Main inference manager
  - Handles device initialization
  - Manages synchronous/asynchronous inference
  - Implements Hailo-specific exception handling

- **`FrameReaderThread`**: Asynchronous frame capture (video mode)
  - Non-blocking video frame reading
  - Queue buffering (default: 4 frames)
  - Parallel I/O with main processing
  - Thread-specific performance profiling (when enabled)

- **`DisplayThread`**: Asynchronous frame display (video mode)
  - Non-blocking video output
  - Queue management with frame dropping
  - User interaction handling
  - Thread-specific performance profiling (when enabled)

- **`PerformanceProfiler`**: Timing and statistics
  - Per-stage timing measurement across multiple threads
  - Statistical analysis with variance calculations
  - Matplotlib visualization (stacked and detailed charts)
  - Perfetto trace export with multi-thread support

### Model Attribution

- The palm detection model used in this project is from Google MediaPipe: [MediaPipe Models](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/models.md)
- The post-processing code for the MediaPipe model is adapted from [blaze_app_python](https://github.com/AlbertaBeef/blaze_app_python/tree/main)

## C++ Port

A C++ port of this pipeline is available under the [`cpp/`](./cpp/) directory. It provides an equivalent command-line interface and feature set, compiled with CMake.

See **[cpp/README.md](./cpp/README.md)** for build instructions, requirements, and usage details.

| Feature                         | Python | C++  |
| ------------------------------- | ------ | ---- |
| Synchronous inference           | Yes    | Yes  |
| Asynchronous inference          | Yes    | Yes  |
| Callback mode                   | Yes    | Yes  |
| Performance profiling (console) | Yes    | Yes  |
| Perfetto trace export           | Yes    | Yes  |
| Matplotlib charts               | Yes    | No   |

## Additional Documentation

For more detailed information about specific features:

- **[FRAMEREADER_INTEGRATION.md](./doc/FRAMEREADER_INTEGRATION.md)**: FrameReaderThread integration details
  - How asynchronous frame capture works
  - Performance benefits and comparison
  - Queue configuration and tuning
  - Thread lifecycle and error handling

- **[DISPLAY_THREAD_TIMING.md](./doc/DISPLAY_THREAD_TIMING.md)**: Technical details about multi-thread profiling implementation
  - Display Thread and Capture Thread profiling
  - Thread-safe timing data collection
  - How timing data flows between threads
  - Implementation notes and performance impact

- **[PERFETTO_VISUALIZATION_GUIDE.md](./doc/PERFETTO_VISUALIZATION_GUIDE.md)**: Complete guide to using Perfetto traces
  - Visual examples of multi-thread timeline view (Main, Capture, Display)
  - How to interpret the trace data
  - Common patterns and what they mean
  - Analysis tips and real-world examples
  - Interactive features and navigation

## About Jupyter Notebooks

This project includes a reference Jupyter notebook that demonstrates:

- How to build a HEF (Hailo Executable Format) file from a MediaPipe TFLite model

This makes it easier to convert and optimize MediaPipe models for use with Hailo hardware acceleration.

To use the notebook:

1. Open it in Jupyter Lab or Jupyter Notebook
2. Follow the step-by-step instructions of DFC notebook to convert your TFLite model to HEF format
3. Use the generated HEF file with the main inference script

## Limitations

- Exception handling has not been validated against real hardware failures.
- Some Hailo devices don't support the synchronous inference API (`--synchronous`) - Hailo-10 is one example. On those devices, use the default asynchronous mode instead.

Found a bug or have a question? Open an issue on this repository's GitHub Issues page.

## License

This project is provided as-is for use with Hailo hardware accelerators.
Please refer to the Hailo SDK license for terms and conditions regarding the use of Hailo software and hardware.

This software is licensed under the Apache License 2.0 - see the LICENSE file for details.

### Third-Party Licenses

- OpenCV: Apache 2.0 License
- NumPy: BSD License
- Matplotlib: PSF-based License

---

**Version**: 1.5.1
**Last Updated**: 2026-07-11
**Hailo SDK Compatibility**: 4.0+

**What's New in 1.5.1:**

- Refactoring pass across `inference.py`/`inference_utils.py`/`postprocess/`:
  - Replaced `Optional[Any]` typing on Hailo SDK handles in `InferPipeline` with minimal `Protocol` types, resolving 19 `mypy --strict` errors
  - Split the 313-line `main()` function (cyclomatic complexity F) into 7 focused helper functions
  - Deduplicated repeated Hailo exception classification logic across `infer_async`/`wait_and_get_output`/`infer_pipeline` into a single helper
  - Formalized the postprocessor interface via `typing.Protocol` instead of a hand-maintained `Union`
  - Deduplicated the three near-identical event-building loops in `PerformanceProfiler.export_perfetto_trace`
- Bug fixes surfaced during the refactor:
  - Fixed a type mismatch in NMS postprocessing where `ImagePostprocessorNmsOnHost.postprocess` was declared against the wrong output shape
  - Fixed `FrameReaderThread`: a momentary read stall no longer terminates the entire video pipeline (previously any single frame-queue timeout was treated as end of stream)
- Added `pyproject.toml` for centralized dependency and dev-tooling metadata; `numpy` is now an explicit dependency in `requirements.txt`

**What's New in 1.5.0:**

- C++ port of the inference pipeline under `cpp/` directory
- Equivalent command-line interface and feature set in C++
- CMake build system with CLI11 and nlohmann_json dependencies
- Synchronous, asynchronous, and callback inference modes in C++
- Performance profiling with console output and Perfetto trace export in C++

**What's New in 1.4.0:**

- Multi-thread profiling with Perfetto trace export
- FrameReaderThread enabled for asynchronous frame capture in video mode
- Display thread timing measurements
- Capture thread (FrameReaderThread) timing measurements
- Up to three separate thread views in Perfetto UI (Main, Capture, Display)
- True three-way parallelism: Capture → Process → Display
- Enhanced performance analysis capabilities
