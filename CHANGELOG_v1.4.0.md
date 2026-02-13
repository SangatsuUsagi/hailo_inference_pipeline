# Version 1.3.0 Release Notes

## Multi-Thread Profiling & Perfetto Trace Enhancement

**Release Date**: February 12, 2026

### Overview

This release adds comprehensive multi-thread profiling capabilities with Perfetto trace export, providing complete visibility into the main inference pipeline, asynchronous frame capture (when using FrameReaderThread), and asynchronous display operations.

### New Features

#### 1. Display Thread Profiling

**What's New:**
- Display thread now captures detailed timing measurements
- Three new profiling checkpoints:
  - `queue_wait`: Time waiting for frames from main thread
  - `display`: OpenCV rendering time (`cv2.imshow`)
  - `key_check`: Keyboard input polling time

**Implementation:**
- Thread-safe timing data collection using `queue.Queue`
- Non-blocking communication between display thread and profiler
- Automatic checkpoint collection when profiling is enabled

**Files Modified:**
- `inference_utils.py`: Enhanced `DisplayThread` class with profiling support
- `inference.py`: Added profiler integration and timing data collection

#### 2. Capture Thread Profiling

**What's New:**
- FrameReaderThread (Capture Thread) now captures detailed timing measurements
- **Enabled by default in video mode** for asynchronous frame capture
- Two new profiling checkpoints:
  - `read`: Time to read frame from video source (`cv2.VideoCapture.read`)
  - `queue_put`: Time to queue frame for main thread processing

**Implementation:**
- Thread-safe timing data collection using `queue.Queue`
- Non-blocking communication between capture thread and profiler
- Automatic checkpoint collection when profiling is enabled
- Enabled automatically for video files, disabled for single images

**Benefits:**
- True three-way parallelism: Capture → Process → Display
- Reduced frame processing latency (I/O happens in parallel)
- Better CPU/accelerator utilization
- Complete visibility of I/O performance

**Files Modified:**
- `inference_utils.py`: Enhanced `FrameReaderThread` class with profiling support
- `inference.py`: Integrated FrameReaderThread into main processing loop

#### 3. Multi-Thread Perfetto Trace Export

**What's New:**
- Perfetto traces now show events across **up to three separate thread rows**:
  - **Main Thread (tid: 1)**: Pipeline stages and frame processing
  - **Capture Thread (tid: 3)**: Video frame capture operations
  - **Display Thread (tid: 2)**: Asynchronous display operations

**Benefits:**
- Visualize true parallelism between all threads
- Identify I/O bottlenecks (capture thread)
- Identify processing bottlenecks (main thread)
- Identify display bottlenecks (display thread)
- Understand queue behavior and synchronization
- Industry-standard trace format

**Files Modified:**
- `inference_utils.py`: Updated `export_perfetto_trace()` method

#### 3. Enhanced Command-Line Interface

**New Option:**
```bash
--trace FILENAME    Export profiling data to Perfetto trace JSON file
```

**Requirements:**
- Must be used with `--profile` flag
- Outputs trace in Perfetto/Chrome trace event format

**Example Usage:**
```bash
python inference.py video.mp4 \
    --net ./hefs/yolov8n.hef \
    --postprocess nms_on_host \
    --profile \
    --trace trace.json
```

**Files Modified:**
- `inference.py`: Added `--trace` argument and validation

#### 4. Comprehensive Documentation

**New Documentation Files:**

1. **DISPLAY_THREAD_TIMING.md**
   - Technical implementation details
   - Thread-safe communication patterns
   - Performance impact analysis
   - Code examples

2. **PERFETTO_VISUALIZATION_GUIDE.md**
   - Visual timeline examples
   - How to use Perfetto UI
   - Common patterns and analysis tips
   - Real-world performance examples
   - Interactive feature guide

**Updated Files:**
- `README.md`: Enhanced with multi-thread profiling documentation

### Technical Details

#### Thread Separation Logic

The Perfetto export automatically separates checkpoints:
```python
# Main Thread (tid: 1)
- 1_frame_read
- 2_preprocessing
- 3_inference_submit
- 4_inference_wait
- 5_postprocessing
- 7_display_queue

# Capture Thread (tid: 3)
- read (from capture_read)
- queue_put (from capture_queue_put)

# Display Thread (tid: 2)
- queue_wait (from display_queue_wait)
- display (from display_display)
- key_check (from display_key_check)
```

#### Performance Impact

**Profiling Overhead:**
- Minimal: Uses fast `time.time()` calls
- Non-blocking: Queue operations don't block threads
- Graceful degradation: Timing data dropped if collection falls behind
- Zero overhead when profiling disabled

**Memory Usage:**
- Timing queue: Small fixed size (default queue size)
- Checkpoint data: ~24 bytes per checkpoint per frame
- Example: 300 frames × 9 checkpoints × 24 bytes = ~65 KB

### Visualization Examples

#### Before (v1.2.0)
```
Perfetto Timeline:
└─ Main Thread
   └─ All events mixed together
```

#### After (v1.3.0)
```
Perfetto Timeline:
├─ Main Thread (tid: 1)
│  ├─ Frame processing pipeline
│  └─ Queue operations
│
├─ Capture Thread (tid: 3)
│  ├─ Video frame reading
│  └─ Queue operations
│
└─ Display Thread (tid: 2)
   ├─ Queue waiting
   ├─ Frame rendering
   └─ Input handling
```

### Compatibility

- **Backward Compatible**: All existing functionality preserved
- **Optional Feature**: Profiling disabled by default
- **Hailo SDK**: 4.0+
- **Python**: 3.8+
- **No new dependencies**: Uses existing libraries

### Migration Guide

**For Existing Users:**

No changes required! The new features are opt-in:

1. **Continue using without profiling**: No changes needed
   ```bash
   python inference.py video.mp4 --net model.hef
   ```

2. **Use profiling without trace**: Works as before
   ```bash
   python inference.py video.mp4 --net model.hef --profile
   ```

3. **Add trace export**: New capability
   ```bash
   python inference.py video.mp4 --net model.hef --profile --trace trace.json
   ```

### Known Limitations

1. **Video Mode Only**: Display thread profiling only works in video mode (not single images)
2. **Trace File Size**: Large traces (1000+ frames) can be several MB
3. **Chrome Tracing**: Chrome has a 100MB file size limit for traces

### Future Enhancements

Potential improvements for future releases:

- [ ] GPU memory usage tracking
- [ ] Network I/O profiling for camera streams
- [ ] Flame graph generation
- [ ] CSV export for external analysis tools
- [ ] Real-time profiling dashboard

### Acknowledgments

This feature was developed based on best practices from:
- Chrome Trace Event Format specification
- Perfetto project documentation
- Real-world profiling of Hailo inference pipelines

### Files Changed

**Modified:**
- `inference_utils.py`: DisplayThread enhancements, FrameReaderThread enhancements, Perfetto export improvements
- `inference.py`: CLI argument, profiler integration, FrameReaderThread integration in main loop
- `README.md`: Updated documentation

**Added:**
- `FRAMEREADER_INTEGRATION.md`: Technical documentation for FrameReaderThread integration
- `DISPLAY_THREAD_TIMING.md`: Technical documentation for multi-thread profiling
- `PERFETTO_VISUALIZATION_GUIDE.md`: User guide for Perfetto traces
- `CHANGELOG_v1.3.0.md`: This file

### Testing Recommendations

After upgrading to v1.3.0, test the new features:

1. **Basic profiling** (existing feature):
   ```bash
   python inference.py video.mp4 --net model.hef --profile
   ```
   Expected: Statistics printed, charts displayed

2. **Trace export**:
   ```bash
   python inference.py video.mp4 --net model.hef --profile --trace test.json
   ```
   Expected: trace.json file created

3. **View in Perfetto**:
   - Visit https://ui.perfetto.dev
   - Load test.json
   - Verify **three thread rows** appear (Main, Capture, Display)
   - Verify events are properly categorized

4. **Verify FrameReaderThread**:
   - Check console output for "Frame reader thread started"
   - In Perfetto, verify Capture Thread (tid: 3) appears
   - Check `1_frame_read` time in Main Thread (should be <1ms)
   - Check `read` time in Capture Thread (actual I/O time)

5. **Validation error**:
   ```bash
   python inference.py video.mp4 --trace test.json
   ```
   Expected: Error message about requiring --profile

6. **Image mode (no threading)**:
   ```bash
   python inference.py image.jpg --net model.hef --profile --trace image.json
   ```
   Expected: No Capture or Display threads (single image mode)

### Support

For issues or questions:
- Check `PERFETTO_VISUALIZATION_GUIDE.md` for usage help
- Check `DISPLAY_THREAD_TIMING.md` for technical details
- Review examples in documentation

---

**Version**: 1.3.0  
**Release Date**: 2026-02-12  
**Hailo SDK Compatibility**: 4.0+
