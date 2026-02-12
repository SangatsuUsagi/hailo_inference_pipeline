# Display Thread & Capture Thread Timing Enhancement

## Overview

The DisplayThread and FrameReaderThread (Capture Thread) classes have been enhanced to capture detailed timing measurements for Perfetto trace visualization. This allows you to see what's happening inside both the display and capture threads on the timeline.

## Changes Made

### 1. DisplayThread Class (`inference_utils.py`)

#### New Constructor Parameter
```python
def __init__(
    self, 
    window_name: str = "Output", 
    max_queue_size: int = 2, 
    profiler: Optional['PerformanceProfiler'] = None
)
```

- Added optional `profiler` parameter to enable timing measurements
- Added `timing_queue` for thread-safe communication of timing data

#### Timing Measurements in Display Loop

The display thread now measures three distinct operations:

1. **`display_queue_wait`**: Time spent waiting for a frame from the queue
2. **`display_display`**: Time spent actually displaying the frame with `cv2.imshow()`
3. **`display_key_check`**: Time spent checking for keyboard input with `cv2.waitKey()`

These measurements happen in the separate display thread and are queued for collection by the main thread.

#### New Method: `collect_timing_data()`

```python
def collect_timing_data(self) -> None:
    """
    Collect timing data from display thread and add to profiler.
    Should be called from main thread after frame processing.
    """
```

This method should be called from the main thread to safely transfer timing data from the display thread to the profiler.

### 2. FrameReaderThread Class (`inference_utils.py`)

#### New Constructor Parameter
```python
def __init__(
    self,
    video_source: cv2.VideoCapture,
    max_queue_size: int = 4,
    profiler: Optional['PerformanceProfiler'] = None
)
```

- Added optional `profiler` parameter to enable timing measurements
- Added `timing_queue` for thread-safe communication of timing data

#### Timing Measurements in Capture Loop

The capture thread now measures two distinct operations:

1. **`capture_read`**: Time spent reading a frame from the video source (`cv2.VideoCapture.read()`)
2. **`capture_queue_put`**: Time spent waiting to put the frame into the queue

These measurements happen in the separate capture thread and are queued for collection by the main thread.

#### New Method: `collect_timing_data()`

```python
def collect_timing_data(self) -> None:
    """
    Collect timing data from frame reader thread and add to profiler.
    Should be called from main thread after getting a frame.
    """
```

This method should be called from the main thread to safely transfer timing data from the capture thread to the profiler.

### 2. Inference Pipeline (`inference.py`)

#### DisplayThread Initialization
```python
display_thread = DisplayThread(
    window_name="Output", 
    max_queue_size=2,
    profiler=profiler if profiling_enabled else None
)
```

The profiler is now passed to the DisplayThread when profiling is enabled.

#### Timing Data Collection
```python
display_thread.display(out_frame)
if profiling_enabled:
    profiler.checkpoint("7_display_queue")
    # Collect timing data from display thread
    display_thread.collect_timing_data()
```

After queuing each frame, timing data is collected from the display thread.

## How It Works

### Thread-Safe Communication

```
Capture Thread                  Main Thread                     Display Thread
      |                              |                                 |
      |-- read frame from video      |                                 |
      |-- queue frame -------------→ |                                 |
      |                              |                                 |
      |                              |-- process frame                 |
      |                              |-- queue frame ----------------→ |
      |                              |                                 |
      |                              |                                 |-- wait for frame
      |                              |                                 |-- cv2.imshow()
      |                              |                                 |-- cv2.waitKey()
      |                              |                                 |-- store timing data
      |                              |                                 |
      |<- collect_timing_data() -----                                  |
      |   (retrieves timing)         |                                 |
      |                              |<- collect_timing_data() --------|
      |                              |   (retrieves timing)            |
      |                              |                                 |
      |-- profiler stores data       |-- profiler stores data          |
```

### Data Flow

1. Capture thread measures frame read and queue operations
2. Main thread processes frames and measures pipeline stages
3. Display thread measures display and input operations
4. All threads place timing data in thread-safe queues
5. Main thread calls `collect_timing_data()` to retrieve data from both threads
6. Profiler stores the timing data with appropriate checkpoint names

### Perfetto Trace Output

In the Perfetto trace, you'll see additional checkpoints on **separate thread views**:

**Main Thread (tid: 1)**
- Pipeline stages: `frame_read`, `preprocessing`, `inference_submit`, `inference_wait`, `postprocessing`
- `display_queue`: queuing frame to display thread

**Capture Thread (tid: 3)** (when using FrameReaderThread)
- `read`: Time to read frame from video source
- `queue_put`: Time to put frame into queue

**Display Thread (tid: 2)** (video mode only)
- `queue_wait`: Time waiting for frames
- `display`: Frame rendering time
- `key_check`: Keyboard input checking time

These appear as separate thread rows in Perfetto, making it easy to see:
- Parallelism between all three threads
- When capture thread is waiting vs. reading
- When display thread is waiting vs. busy
- True end-to-end pipeline timing including I/O

## Benefits

### Before (Without Thread Timing)
```
Perfetto Timeline:
├─ Main Thread
│  ├─ 7_display_queue (main thread queues frame)
│  └─ [Other thread activity is invisible]
```

### After (With Multi-Thread Timing)
```
Perfetto Timeline:
├─ Main Thread (tid: 1)
│  ├─ 1_frame_read
│  ├─ 2_preprocessing
│  ├─ 3_inference_submit
│  ├─ 4_inference_wait
│  ├─ 5_postprocessing
│  └─ 7_display_queue
│
├─ Capture Thread (tid: 3)
│  ├─ read
│  └─ queue_put
│
└─ Display Thread (tid: 2)
   ├─ queue_wait
   ├─ display
   └─ key_check
```

**Key Improvements:**
- **Separate Thread Rows**: Main, Capture, and Display threads appear as distinct rows in Perfetto
- **True Parallelism**: See when all threads run in parallel vs. serially
- **Complete Picture**: Understand the full pipeline including I/O and async operations
- **Bottleneck Analysis**: Easily identify if capture, processing, or display is the bottleneck

## Usage Example

```bash
# Run with profiling and trace export
python inference.py video.mp4 \
    --net ./hefs/yolov8n.hef \
    --postprocess nms_on_host \
    --profile \
    --trace trace.json

# Open in Perfetto UI
# Visit: https://ui.perfetto.dev
# Load: trace.json
```

### What You'll See in Perfetto

The trace will show **up to three separate thread rows** (depending on implementation):

**Process: Hailo Inference Pipeline**

**Thread 1: Main Thread**
- Frame markers (instant events)
- Pipeline stages: frame_read, preprocessing, inference_submit, inference_wait, postprocessing
- display_queue: queuing frame to display thread

**Thread 3: Capture Thread** (when using FrameReaderThread)
- read: cv2.VideoCapture.read() time
- queue_put: time to queue frame for main thread

**Thread 2: Display Thread** (video mode only)
- queue_wait: time waiting for frames from main thread
- display: OpenCV imshow() rendering time  
- key_check: keyboard input polling time

### Benefits of Multi-Thread View

1. **See Complete Parallelism**: Visualize when capture, main, and display threads all run concurrently
2. **Identify I/O Bottlenecks**: See if video reading is slow
3. **Understand Queue Dynamics**: See how frames flow between threads
4. **Find True Bottlenecks**: Determine which thread limits overall performance
5. **Validate Pipeline**: Ensure threads aren't blocking each other unnecessarily

## Performance Impact

The timing measurement overhead is minimal:
- Uses `time.time()` which is very fast
- Thread-safe queue operations are non-blocking
- No impact when profiling is disabled
- Timing data is dropped if collection falls behind (no blocking)

## Implementation Notes

### Thread Safety
- Timing data is communicated via `queue.Queue` (thread-safe)
- `collect_timing_data()` uses `get_nowait()` to avoid blocking
- Data is dropped if the queue is full (graceful degradation)

### Checkpoint Naming
Thread checkpoints are prefixed internally to distinguish them:
- **Main thread** (tid: 1): `1_frame_read`, `2_preprocessing`, etc.
- **Capture thread** (tid: 3): `capture_read`, `capture_queue_put` → shown as `read`, `queue_put` in Perfetto
- **Display thread** (tid: 2): `display_queue_wait`, `display_display`, `display_key_check` → shown as `queue_wait`, `display`, `key_check` in Perfetto

Prefixes are removed in Perfetto for clarity.

### Thread IDs in Perfetto
- **Main Thread**: tid = 1
- **Display Thread**: tid = 2
- **Capture Thread**: tid = 3
- All threads share the same process ID (pid = 1)

This separation allows Perfetto to show events on separate thread rows, making parallelism and timing relationships clearly visible.

### Perfetto Integration
All thread timing data is automatically included in the Perfetto trace export when using `--trace`, providing complete visibility into the entire pipeline with proper multi-thread visualization.
