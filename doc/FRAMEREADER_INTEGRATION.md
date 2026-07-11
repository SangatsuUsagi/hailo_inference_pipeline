# FrameReaderThread Integration

## Overview

The inference pipeline now uses `FrameReaderThread` for asynchronous frame capture in video mode. This enables true three-way parallelism: **Capture → Process → Display**.

## Implementation Details

### When FrameReaderThread is Used

**Video Mode Only:**
- Automatically enabled when processing video files
- Single image mode continues to use synchronous `cap.read()`

**Thread Lifecycle:**
1. Started after determining video mode: `is_image = False`
2. Runs in parallel with main processing loop
3. Stopped in finally block before display thread

### Frame Reading Flow

#### Image Mode (No Threading)
```python
ret, frame = cap.read()
# Process frame immediately
```

#### Video Mode (With FrameReaderThread)
```python
# Frame reader thread runs in background, continuously reading
frame = frame_reader_thread.get_frame()
# Main thread gets pre-read frame from queue
```

### Code Changes

#### 1. Variable Initialization
```python
frame_reader_thread: Optional[FrameReaderThread] = None
```

#### 2. Thread Startup (Video Mode)
```python
if not is_image:
    frame_reader_thread = FrameReaderThread(
        video_source=cap,
        max_queue_size=4,
        profiler=profiler if profiling_enabled else None
    )
    frame_reader_thread.start()
```

#### 3. Frame Reading in Main Loop
```python
if is_image:
    # Direct read for single images
    ret, frame = cap.read()
    if profiling_enabled:
        profiler.checkpoint("1_frame_read")
else:
    # Threaded read for video
    frame = frame_reader_thread.get_frame()
    if profiling_enabled:
        profiler.checkpoint("1_frame_read")
        # Collect timing data from capture thread
        frame_reader_thread.collect_timing_data()

    if frame is None:
        if frame_reader_thread.has_error():
            # Reader thread has genuinely stopped (end of stream or a
            # read error - cv2.VideoCapture can't reliably tell the two apart)
            break
        # Otherwise get_frame() just timed out waiting on an empty queue;
        # the reader thread is presumably still running, so retry instead
        # of treating a transient stall as end of stream. Bounded so an
        # unresponsive reader thread still can't hang the loop forever.
        consecutive_empty_reads += 1
        if consecutive_empty_reads >= MAX_CONSECUTIVE_EMPTY_READS:
            break
        continue

    consecutive_empty_reads = 0
    ret = True
```

#### 4. Thread Cleanup
```python
finally:
    # Stop frame reader thread first
    if frame_reader_thread is not None:
        print("\nStopping frame reader thread...")
        frame_reader_thread.stop()
    
    # Then stop display thread
    if display_thread is not None:
        print("Stopping display thread...")
        display_thread.stop()
```

## Benefits

### 1. **Parallel I/O**
Frame reading happens in a separate thread while main thread processes:
```
Frame N:   [Capture] ──────────────────►
Frame N-1:           [Process] ─────────►
Frame N-2:                     [Display] ►
```

### 2. **Reduced Latency**
- Next frame is pre-buffered and ready
- Main thread doesn't wait for disk/network I/O
- Smoother pipeline flow

### 3. **Better Performance**
- Overlaps I/O with computation
- Maximizes CPU/accelerator utilization
- Particularly beneficial for high-resolution video

### 4. **Queue Buffering**
- Default queue size: 4 frames
- Provides buffer against I/O jitter
- Prevents stalls from variable read times

## Profiling Integration

When profiling is enabled (`--profile`), FrameReaderThread captures:

### Capture Thread Metrics
- **`capture_read`**: Time to read frame from video source
- **`capture_queue_put`**: Time to put frame in queue

These appear on **Thread 3** in Perfetto traces.

### Main Thread Metrics
- **`1_frame_read`**: Time to get frame from queue (should be fast!)

The difference between old synchronous read time and new queue get time shows the I/O parallelization benefit.

## Performance Comparison

### Before (Synchronous Read)
```
Total Frame Time: 20ms
├─ frame_read: 5ms      (blocking I/O)
├─ preprocessing: 2ms
├─ inference: 10ms
└─ postprocessing: 3ms
```

### After (Asynchronous Read with FrameReaderThread)
```
Main Thread Time: 15ms
├─ frame_read: <1ms     (queue get - fast!)
├─ preprocessing: 2ms
├─ inference: 10ms
└─ postprocessing: 3ms

Capture Thread (parallel):
├─ read: 5ms            (happens in background)
└─ queue_put: <1ms
```

**Result**: 5ms saved per frame = 25% faster!

## Queue Size Configuration

The default queue size is 4 frames. This can be adjusted:

```python
frame_reader_thread = FrameReaderThread(
    video_source=cap,
    max_queue_size=8,  # Larger buffer for more unstable I/O
    profiler=profiler if profiling_enabled else None
)
```

**Trade-offs:**
- **Larger queue**: More buffering, higher latency, more memory
- **Smaller queue**: Less buffering, lower latency, risk of stalls

## Error Handling

`get_frame()` returns `None` in two different situations, and the main loop
distinguishes between them via `has_error()` before deciding whether to stop:

### The Reader Thread Has Genuinely Stopped
Once `cv2.VideoCapture.read()` returns `False` - either normal end of video
or a genuine read error (OpenCV doesn't reliably distinguish the two) - the
reader thread sets an internal `read_error` flag *before* queuing a `None`
sentinel, then exits its loop. A caller that receives this `None` can check
`has_error()` to confirm the thread has actually stopped:

```python
frame = frame_reader_thread.get_frame()
if frame is None:
    if frame_reader_thread.has_error():
        break  # Reader thread stopped: end of video or a read error
```

### A Transient Empty-Queue Timeout
`get_frame()` also returns `None` if its 1-second `queue.get()` call times
out - which can happen even while the reader thread is alive and running
(e.g. a momentary I/O stall). Treating this identically to "the reader
stopped" would silently end the whole pipeline on a single slow read, so the
main loop retries instead, up to `MAX_CONSECUTIVE_EMPTY_READS` (30, ~30s at
the default 1.0s timeout) consecutive empty reads:

```python
if frame is None:
    if frame_reader_thread.has_error():
        break
    consecutive_empty_reads += 1
    if consecutive_empty_reads >= MAX_CONSECUTIVE_EMPTY_READS:
        break  # Reader thread appears unresponsive
    continue
```

The retry counter resets to `0` every time a real frame is received, so it
only trips on *consecutive* empty reads, not on occasional ones scattered
across a long-running stream.

## Thread Safety

### Frame Queue
- `queue.Queue` is thread-safe
- No explicit locking needed
- Handles concurrent access automatically

### Timing Queue
- Separate `queue.Queue` for profiling data
- Non-blocking operations (`put_nowait`, `get_nowait`)
- Dropped data on overflow (no blocking)

## Startup Sequence

```
1. Create VideoCapture
2. Determine if image or video
3. If video:
   a. Create FrameReaderThread with profiler
   b. Start frame reader thread
   c. Create DisplayThread with profiler
   d. Start display thread
4. Enter main processing loop
```

## Shutdown Sequence

```
1. Exit main loop
2. Stop FrameReaderThread (stops reading)
3. Stop DisplayThread (stops displaying)
4. Close InferPipeline (release Hailo device)
5. Release VideoCapture
6. Destroy OpenCV windows
```

## Perfetto Trace Visualization

With FrameReaderThread enabled, Perfetto shows:

```
┌─────────────────────────────────────────────┐
│ Main Thread (tid: 1)                        │
│ ├─ 1_frame_read (fast - queue get)         │
│ ├─ 2_preprocessing                          │
│ ├─ 3_inference_submit                       │
│ └─ ...                                      │
├─────────────────────────────────────────────┤
│ Capture Thread (tid: 3)                     │
│ ├─ read (actual I/O time)                   │
│ └─ queue_put                                │
├─────────────────────────────────────────────┤
│ Display Thread (tid: 2)                     │
│ ├─ queue_wait                               │
│ ├─ display                                  │
│ └─ key_check                                │
└─────────────────────────────────────────────┘
```

**Key Observations:**
- Capture thread's `read` overlaps with main thread processing
- Main thread's `1_frame_read` is much faster (queue get vs disk I/O)
- Three threads show true parallelism

## Backward Compatibility

### Image Mode
- Single images continue to work as before
- No threading overhead for single frame processing
- Direct `cap.read()` used

### Existing Code
- No changes needed to existing call sites
- Thread management is automatic
- Cleanup is handled in finally block

## Debug Output

When running in video mode with frame reader thread:

```
Frame reader thread started
Display thread started

Starting inference loop...
Profiling enabled.
Press 'q' to quit (video mode only)

[Processing frames...]

Stopping frame reader thread...
Stopping display thread...
```

## Common Issues and Solutions

### Issue: Frames arrive out of order
**Cause**: Queue doesn't guarantee order (but in practice it does)
**Solution**: Frame reader thread reads sequentially, so order is preserved

### Issue: Queue fills up
**Cause**: Main thread processing slower than capture
**Solution**: Normal! Queue provides buffering. Old frames may be dropped.

### Issue: High memory usage
**Cause**: Large queue size with high-resolution frames
**Solution**: Reduce `max_queue_size` or process lower resolution

### Issue: Choppy video playback
**Cause**: Display thread can't keep up
**Solution**: Check display thread timing in Perfetto trace

### Issue: Pipeline stops after a brief camera/stream hiccup
**Cause**: Before the retry logic described in [Error Handling](#error-handling)
was added, any `get_frame()` timeout (even a transient one, with the reader
thread still alive) was treated as end of stream and broke the main loop.
**Solution**: Already fixed - the main loop now checks `has_error()` and only
stops on a genuine reader-thread stop, retrying (bounded to
`MAX_CONSECUTIVE_EMPTY_READS`) on transient empty-queue timeouts.

## Future Enhancements

Potential improvements:
- [ ] Configurable queue size via command line
- [ ] Frame skipping strategy when queue is full
- [ ] Timestamp preservation for frame synchronization
- [ ] Multiple video source support
- [ ] Camera source optimization

## Testing

To verify FrameReaderThread is working:

1. **Run with profiling:**
   ```bash
   python inference.py video.mp4 --profile --trace trace.json
   ```

2. **Check console output:**
   ```
   Frame reader thread started
   Display thread started
   ```

3. **View Perfetto trace:**
   - Open trace.json in Perfetto UI
   - Verify "Capture Thread (tid: 3)" appears
   - Check `read` and `queue_put` events
   - Confirm parallelism with main thread

4. **Compare performance:**
   - Note `1_frame_read` time (should be <1ms)
   - Note `capture_read` time (actual I/O time)
   - Overall FPS should improve

## Summary

FrameReaderThread integration provides:
✅ True three-way parallelism (Capture → Process → Display)
✅ Reduced frame processing latency
✅ Better CPU/accelerator utilization
✅ Complete profiling visibility
✅ Automatic enablement for video mode
✅ Zero impact on image mode
✅ Clean thread lifecycle management
