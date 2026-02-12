# Display Thread Timing Enhancement

## Overview

The DisplayThread class has been enhanced to capture detailed timing data for Perfetto trace visualization. This allows you to see what's happening inside the display thread on the timeline.

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
Main Thread                          Display Thread
    |                                      |
    |-- queue frame -------------------->  |
    |                                      |-- wait for frame
    |                                      |-- cv2.imshow()
    |                                      |-- cv2.waitKey()
    |                                      |-- store timing data
    |                                      |
    |<- collect_timing_data() -----------  |
    |   (retrieves timing from queue)      |
    |                                      |
    |-- profiler stores data              |
```

### Data Flow

1. Display thread measures operation times
2. Timing data is placed in a thread-safe queue
3. Main thread calls `collect_timing_data()` to retrieve the data
4. Profiler stores the timing data with appropriate checkpoint names

### Perfetto Trace Output

In the Perfetto trace, you'll see additional checkpoints on a **separate thread view**:

**Main Thread (tid: 1)**
- `1_frame_read`, `2_preprocessing`, `3_inference_submit`, etc.
- `7_display_queue` - queuing frame to display thread

**Display Thread (tid: 2)**
- `queue_wait` - How long waiting for frames
- `display` - Actual frame rendering time
- `key_check` - Keyboard input checking time

These appear as separate thread rows in Perfetto, making it easy to see:
- Parallelism between main and display threads
- When the display thread is waiting vs. busy
- True end-to-end pipeline timing

## Benefits

### Before (Without Display Thread Timing)
```
Perfetto Timeline:
├─ Main Thread
│  ├─ 7_display_queue (main thread queues frame)
│  └─ [Display thread activity is invisible]
```

### After (With Display Thread Timing - Multi-Thread View)
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
└─ Display Thread (tid: 2)
   ├─ queue_wait
   ├─ display
   └─ key_check
```

**Key Improvements:**
- **Separate Thread Rows**: Main and Display threads appear as distinct rows in Perfetto
- **True Parallelism**: See when threads run in parallel vs. serially
- **Complete Picture**: Understand the full pipeline including async display operations
- **Bottleneck Analysis**: Easily identify if display thread is waiting or busy

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

The trace will show **two separate thread rows**:

**Process: Hailo Inference Pipeline**

**Thread 1: Main Thread**
- Frame markers (instant events)
- Pipeline stages: frame_read, preprocessing, inference_submit, inference_wait, postprocessing
- display_queue: queuing frame to display thread

**Thread 2: Display Thread** (if video mode)
- queue_wait: time waiting for frames from main thread
- display: OpenCV imshow() rendering time  
- key_check: keyboard input polling time

### Benefits of Multi-Thread View

1. **See Parallelism**: Visually see when main thread and display thread run concurrently
2. **Identify Blocking**: See if display thread is starved (long queue_wait) or if main thread is blocked
3. **Understand Queue Behavior**: See the relationship between main thread queuing and display thread consuming
4. **Find Bottlenecks**: Quickly identify if rendering or input checking is slow

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
Display thread checkpoints are prefixed with `display_` internally to distinguish them from main thread checkpoints. In the Perfetto trace:
- **Main thread**: `1_frame_read`, `2_preprocessing`, etc.
- **Display thread** (shown on tid: 2): `queue_wait`, `display`, `key_check` (prefix removed for clarity)

### Thread IDs in Perfetto
- **Main Thread**: tid = 1
- **Display Thread**: tid = 2
- Both threads share the same process ID (pid = 1)

This separation allows Perfetto to show events on separate thread rows, making parallelism and timing relationships clearly visible.

### Perfetto Integration
The display thread timing data is automatically included in the Perfetto trace export when using `--trace`, providing complete visibility into the entire pipeline with proper multi-thread visualization.
