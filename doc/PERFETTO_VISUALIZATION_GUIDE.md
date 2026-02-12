# Perfetto Multi-Thread View Example

## Visual Representation

When you open the trace in Perfetto UI, you'll see something like this:

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Hailo Inference Pipeline (Process)                                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│ 🧵 Main Thread (tid: 1)                                                             │
│ ┌─────────────────────────────────────────────────────────────────────────────┐     │
│ │ Frame 1 ●                                                                   │     │
│ │ ┌──────────┐┌──────────┐┌────┐┌───────────┐┌──────────┐┌───┐                │     │
│ │ │frame_read││preprocess││inf.││inf_wait   ││postproc  ││q  │                │     │
│ │ │          ││          ││sub.││           ││          ││ue │                │     │
│ │ └──────────┘└──────────┘└────┘└───────────┘└──────────┘└───┘                │     │
│ │                                                                             │     │
│ │ Frame 2 ●                                                                   │     │
│ │      ┌──────────┐┌──────────┐┌────┐┌───────────┐┌──────────┐┌───┐           │     │
│ │      │frame_read││preprocess││inf.││inf_wait   ││postproc  ││q  │           │     │
│ │      │          ││          ││sub.││           ││          ││ue │           │     │
│ │      └──────────┘└──────────┘└────┘└───────────┘└──────────┘└───┘           │     │
│ └─────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                     │
│ 🧵 Capture Thread (tid: 3) [when using FrameReaderThread]                           │
│ ┌──────────────────────────────────────────────────────────────────────────────┐    │
│ │                                                                              │    │
│ │ ┌──────┐┌──┐                                                                 │    │
│ │ │ read ││qt│                                                                 │    │
│ │ │      ││pt│                                                                 │    │
│ │ └──────┘└──┘                                                                 │    │
│ │         ┌──────┐┌──┐                                                         │    │
│ │         │ read ││qt│                                                         │    │
│ │         │      ││pt│                                                         │    │
│ │         └──────┘└──┘                                                         │    │
│ └──────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                     │
│ 🧵 Display Thread (tid: 2)                                                          │
│ ┌──────────────────────────────────────────────────────────────────────────────┐    │
│ │                                                                              │    │
│ │      ┌────┐┌──────┐┌───┐                                                     │    │
│ │      │wait││render││key│                                                     │    │
│ │      │    ││      ││chk│                                                     │    │
│ │      └────┘└──────┘└───┘                                                     │    │
│ │                       ┌────┐┌──────┐┌───┐                                    │    │
│ │                       │wait││render││key│                                    │    │
│ │                       │    ││      ││chk│                                    │    │
│ │                       └────┘└──────┘└───┘                                    │    │
│ └──────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘

Time ──────────────────────────────────────────────────────────────────────────────►
```

## Key Observations

### 1. Parallel Execution

Notice how the Capture Thread reads Frame 2 **while** the Main Thread processes Frame 1, and the Display Thread shows Frame 1. This is true three-way parallelism!

### 2. Queue Interactions

- **Capture → Main**: Capture Thread's `queue_put` feeds Main Thread's `frame_read`
- **Main → Display**: Main Thread's `queue` feeds Display Thread's `wait`

### 3. Frame Markers

The `●` symbols on the Main Thread timeline mark the start of each frame processing cycle.

### 4. Thread Activity

- **Capture Thread**: Continuously reads frames from video source
- **Main Thread**: Processes frames through the inference pipeline
- **Display Thread**: Renders processed frames to screen

### 4. Color Coding (in actual Perfetto)

- Main Thread events: Various colors based on category ("pipeline")
- Capture Thread events: Different color palette (category "capture")
- Display Thread events: Different color palette (category "display")
- Frame markers: Distinct color (usually red/pink)

## Interactive Features in Perfetto

### Navigation

- **Zoom**: Use mouse wheel or W/S keys
- **Pan**: Click and drag, or A/D keys
- **Select**: Click on any event to see details

### Event Details

When you click on an event, you'll see:

```
Event: postprocessing
Duration: 2.789 ms
Thread: Main Thread
Frame: 1
Category: pipeline
```

### Thread Filtering

- Click thread name to collapse/expand
- Right-click for more options
- Use search to find specific events

### Time Range Selection

- Select a time range to see statistics
- Export selected range
- Compare multiple ranges

## Typical Patterns

### Good Pipeline (Well-Balanced)

```
Main Thread:    ████████████████████████████████████████
Display Thread:       ███      ███      ███      ███
                      ↑        ↑        ↑        ↑
                   Processing frames with minimal wait
```

### Display Bottleneck

```
Main Thread:    ████████████████████████████████████████
Display Thread: ███████████████████████████████████████
                ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
                Display is constantly busy (rendering too slow)
```

### Starved Display Thread

```
Main Thread:    ████████████████████████████████████████
Display Thread: ───█──────█──────█──────█──────█────
                   ↑      ↑      ↑      ↑      ↑
                   Long waits between frames (main thread too slow)
```

## Real-World Example Timing (YOLOv8n @ 74 FPS)

### Main Thread

- Frame read: ~0.2 ms
- Preprocessing: ~1.6 ms
- Inference submit: ~0.1 ms
- Inference wait: ~11.5 ms
- Postprocessing: ~0.6 ms
- Display queue: ~0.03 ms

**Total: ~14 ms per frame**

### Display Thread (running in parallel)

- Queue wait: ~0.1 ms (good - frames available immediately)
- Display: ~2.7 ms (OpenCV rendering)
- Key check: ~0.01 ms

**Total: ~2.8 ms per frame**

Notice that while the main thread takes 14ms, the display thread only needs 2.8ms. This means the display thread spends most of its time waiting for new frames, which is expected and good - it means the main thread is the bottleneck (as it should be for inference-heavy workloads).

## Analysis Tips

### 1. Check Thread Utilization

Look at the density of events on each thread:

- Dense bars = thread is busy
- Gaps = thread is idle/waiting

### 2. Identify Bottlenecks

The slowest operation on the critical path (usually Main Thread) determines overall FPS.

### 3. Validate Parallelism

Ensure Display Thread events overlap with Main Thread events. If they're strictly sequential, there may be a synchronization issue.

### 4. Watch for Anomalies

- Sudden spikes in duration
- Gaps in the timeline
- Missing events
- Threads blocking each other

### 5. Compare Across Frames

Use Perfetto's selection tools to compare timing across multiple frames:

- Is there variance in inference time?
- Do certain frames take longer?
- Are there periodic patterns?
