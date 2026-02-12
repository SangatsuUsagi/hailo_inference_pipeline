"""
Hailo Inference Pipeline Utility Classes
"""
import queue
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np


class DisplayThread:
    """Manages asynchronous display of frames in a separate thread."""

    def __init__(self, window_name: str = "Output", max_queue_size: int = 2) -> None:
        """
        Initialize the display thread.

        Args:
            window_name: Name of the OpenCV window
            max_queue_size: Maximum number of frames to buffer
        """
        self.window_name = window_name
        self.frame_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(
            maxsize=max_queue_size
        )
        self.should_stop = False
        self.thread: Optional[threading.Thread] = None
        self.quit_requested = False

    def start(self) -> None:
        """Start the display thread."""
        self.should_stop = False
        self.quit_requested = False
        self.thread = threading.Thread(target=self._display_loop, daemon=True)
        self.thread.start()

    def _display_loop(self) -> None:
        """Main loop for the display thread."""
        while not self.should_stop:
            try:
                frame = self.frame_queue.get(timeout=0.1)

                if frame is None:
                    break

                cv2.imshow(self.window_name, frame)

                # Check for 'q' key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.quit_requested = True

            except queue.Empty:
                # Check for window close event
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    self.quit_requested = True
                    break
                continue

    def display(self, frame: np.ndarray) -> bool:
        """
        Queue a frame for display.

        Args:
            frame: Frame to display

        Returns:
            True if frame was queued, False if queue is full
        """
        try:
            self.frame_queue.put_nowait(frame)
            return True
        except queue.Full:
            # Drop frame if queue is full to avoid blocking
            return False

    def is_quit_requested(self) -> bool:
        """Check if user requested to quit."""
        return self.quit_requested

    def stop(self) -> None:
        """Stop the display thread and clean up."""
        self.should_stop = True

        # Send sentinel value to unblock the queue
        try:
            self.frame_queue.put(None, timeout=0.5)
        except queue.Full:
            pass

        # Wait for thread to finish
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        cv2.destroyAllWindows()


class FrameReaderThread:
    """Asynchronously reads frames from video source in a separate thread."""

    def __init__(self, video_source: cv2.VideoCapture, max_queue_size: int = 4) -> None:
        """
        Initialize the frame reader thread.

        Args:
            video_source: OpenCV VideoCapture object
            max_queue_size: Maximum number of frames to buffer
        """
        self.video_source = video_source
        self.frame_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(
            maxsize=max_queue_size
        )
        self.should_stop = False
        self.thread: Optional[threading.Thread] = None
        self.read_error = False

    def start(self) -> None:
        """Start the frame reading thread."""
        self.should_stop = False
        self.read_error = False
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self) -> None:
        """Main loop for reading frames."""
        while not self.should_stop:
            ret, frame = self.video_source.read()

            if not ret:
                # Signal end of video
                try:
                    self.frame_queue.put(None, timeout=1.0)
                except queue.Full:
                    pass
                self.read_error = not ret
                break

            try:
                # Block if queue is full (backpressure mechanism)
                self.frame_queue.put(frame, timeout=1.0)
            except queue.Full:
                # Drop frame if queue is consistently full
                continue

    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get the next frame from the queue.

        Args:
            timeout: Maximum time to wait for a frame

        Returns:
            Frame as numpy array, or None if video ended or timeout occurred
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def has_error(self) -> bool:
        """Check if a read error occurred."""
        return self.read_error

    def stop(self) -> None:
        """Stop the frame reading thread."""
        self.should_stop = True

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)


class PerformanceProfiler:
    """Profiles execution times across different pipeline stages."""

    def __init__(self) -> None:
        self.checkpoints: Dict[str, List[float]] = defaultdict(list)
        self.last_time: Optional[float] = None
        self.frame_start_time: Optional[float] = None
        self.frame_order: List[str] = []  # Track the order of checkpoints

    def start_frame(self) -> None:
        """Mark the beginning of a frame processing cycle."""
        self.frame_start_time = time.time()
        self.last_time = self.frame_start_time

    def checkpoint(self, name: str) -> None:
        """
        Record a checkpoint with time elapsed since the last checkpoint.

        Args:
            name: Identifier for this checkpoint
        """
        current_time = time.time()
        if self.last_time is not None:
            elapsed = current_time - self.last_time
            self.checkpoints[name].append(elapsed)

            # Track order of checkpoints (only once)
            if name not in self.frame_order:
                self.frame_order.append(name)

        self.last_time = current_time

    def end_frame(self) -> None:
        """Mark the end of frame processing and record total time."""
        if self.frame_start_time is not None:
            total_time = time.time() - self.frame_start_time
            self.checkpoints["total_frame_time"].append(total_time)

    def print_statistics(self) -> None:
        """Print comprehensive statistics for all recorded checkpoints."""
        print("\n" + "=" * 92)
        print("PERFORMANCE PROFILING RESULTS")
        print("=" * 92)

        if not self.checkpoints:
            print("No profiling data collected.")
            return

        print(
            f"{'Checkpoint':<30} {'Count':>8} {'Min(ms)':>12} {'Max(ms)':>12} "
            f"{'Mean(ms)':>12} {'Var(ms²)':>12}"
        )
        print("-" * 92)

        for name, times in sorted(self.checkpoints.items()):
            if len(times) > 0:
                times_ms = np.array(times) * 1000
                min_time = np.min(times_ms)
                max_time = np.max(times_ms)
                mean_time = np.mean(times_ms)
                var_time = np.var(times_ms)

                print(
                    f"{name:<30} {len(times):>8} {min_time:>12.3f} {max_time:>12.3f} "
                    f"{mean_time:>12.3f} {var_time:>12.3f}"
                )

        print("=" * 92)

        if "total_frame_time" in self.checkpoints:
            total_times = self.checkpoints["total_frame_time"]
            avg_frame_time = np.mean(total_times)
            avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            print(f"\nAverage Frame Processing Time: {avg_frame_time * 1000:.3f} ms")
            print(f"Average FPS (from frame time): {avg_fps:.2f}")

        print("=" * 92 + "\n")

    def draw_stacked_time_chart(self) -> None:
        """
        Draw a stacked time chart showing timing breakdown for each frame.

        Vertical axis shows profiling stages from bottom to top.
        Horizontal axis shows time (frame number).
        Each stage is color-coded and stacked on top of previous stages.
        """
        if not self.checkpoints:
            print("No profiling data to visualize.")
            return

        # Exclude 'total_frame_time' from the stacked chart
        checkpoint_names = [
            name for name in self.frame_order if name != "total_frame_time"
        ]

        if not checkpoint_names:
            print("No checkpoint data to visualize.")
            return

        # Determine the number of frames
        num_frames = len(self.checkpoints[checkpoint_names[0]])

        # Prepare data for stacking (convert to milliseconds)
        data_matrix = np.zeros((len(checkpoint_names), num_frames))

        for i, name in enumerate(checkpoint_names):
            times = self.checkpoints[name]
            # Pad with zeros if this checkpoint has fewer entries
            if len(times) < num_frames:
                times = times + [0] * (num_frames - len(times))
            data_matrix[i, :] = np.array(times[:num_frames]) * 1000  # Convert to ms

        # Create the stacked area chart
        fig, ax = plt.subplots(figsize=(14, 8))

        # Generate colors for each stage
        colors = plt.cm.tab20(np.linspace(0, 1, len(checkpoint_names)))

        # Create frame numbers for x-axis
        frame_numbers = np.arange(1, num_frames + 1)

        # Plot stacked areas
        ax.stackplot(
            frame_numbers,
            data_matrix,
            labels=checkpoint_names,
            colors=colors,
            alpha=0.8,
        )

        # Customize the plot
        ax.set_xlabel("Frame Number", fontsize=12, fontweight="bold")
        ax.set_ylabel("Cumulative Time (ms)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Pipeline Stage Timing - Stacked Time Chart",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        # Add legend
        ax.legend(
            loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=9
        )

        # Add grid
        ax.grid(True, alpha=0.3, linestyle="--")

        # Calculate and display average total time
        if "total_frame_time" in self.checkpoints:
            total_times = np.array(self.checkpoints["total_frame_time"]) * 1000
            avg_total = np.mean(total_times)
            ax.axhline(
                y=avg_total,
                color="r",
                linestyle="--",
                linewidth=2,
                label=f"Avg Total: {avg_total:.2f}ms",
            )

            # Update legend to include the average line
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                borderaxespad=0,
                fontsize=9,
            )

        # Adjust layout to prevent label cutoff
        plt.tight_layout()

        # Display the chart
        print("\nDisplaying stacked time chart...")
        print("Close the chart window to continue.")
        plt.show()

    def draw_detailed_timing_chart(self) -> None:
        """
        Draw detailed timing charts with multiple subplots.

        Shows individual timing for each stage across frames.
        """
        if not self.checkpoints:
            print("No profiling data to visualize.")
            return

        # Exclude 'total_frame_time' temporarily
        checkpoint_names = [
            name for name in self.frame_order if name != "total_frame_time"
        ]

        if not checkpoint_names:
            print("No checkpoint data to visualize.")
            return

        # Determine grid layout
        num_plots = len(checkpoint_names) + 1  # +1 for total_frame_time
        num_cols = 2
        num_rows = (num_plots + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
        axes = axes.flatten() if num_plots > 1 else [axes]

        plot_idx = 0

        # Plot each checkpoint
        for name in checkpoint_names:
            times_ms = np.array(self.checkpoints[name]) * 1000
            frame_numbers = np.arange(1, len(times_ms) + 1)

            ax = axes[plot_idx]
            ax.plot(
                frame_numbers,
                times_ms,
                linewidth=1.5,
                marker="o",
                markersize=3,
                alpha=0.7,
            )
            ax.set_title(name, fontsize=10, fontweight="bold")
            ax.set_xlabel("Frame Number", fontsize=9)
            ax.set_ylabel("Time (ms)", fontsize=9)
            ax.grid(True, alpha=0.3)

            # Add mean line
            mean_val = np.mean(times_ms)
            ax.axhline(
                y=mean_val,
                color="r",
                linestyle="--",
                linewidth=1,
                alpha=0.6,
                label=f"Mean: {mean_val:.2f}ms",
            )
            ax.legend(fontsize=8)

            plot_idx += 1

        # Plot total frame time
        if "total_frame_time" in self.checkpoints:
            times_ms = np.array(self.checkpoints["total_frame_time"]) * 1000
            frame_numbers = np.arange(1, len(times_ms) + 1)

            ax = axes[plot_idx]
            ax.plot(
                frame_numbers,
                times_ms,
                linewidth=1.5,
                marker="o",
                markersize=3,
                alpha=0.7,
                color="purple",
            )
            ax.set_title("Total Frame Time", fontsize=10, fontweight="bold")
            ax.set_xlabel("Frame Number", fontsize=9)
            ax.set_ylabel("Time (ms)", fontsize=9)
            ax.grid(True, alpha=0.3)

            mean_val = np.mean(times_ms)
            fps = 1000.0 / mean_val if mean_val > 0 else 0
            ax.axhline(
                y=mean_val,
                color="r",
                linestyle="--",
                linewidth=1,
                alpha=0.6,
                label=f"Mean: {mean_val:.2f}ms ({fps:.1f} FPS)",
            )
            ax.legend(fontsize=8)

            plot_idx += 1

        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(
            "Detailed Pipeline Timing Analysis", fontsize=14, fontweight="bold", y=0.995
        )
        plt.tight_layout()

        print("\nDisplaying detailed timing charts...")
        print("Close the chart window to continue.")
        plt.show()

    def export_perfetto_trace(self, output_path: str = "trace.json") -> None:
        """
        Export profiling data to Perfetto/Chrome trace event format.

        The output JSON can be visualized in:
        - Chrome: chrome://tracing
        - Perfetto UI: https://ui.perfetto.dev

        Args:
            output_path: Path to save the JSON trace file
        """
        import json

        if not self.checkpoints:
            print("No profiling data to export.")
            return

        trace_events = []
        process_id = 1
        thread_id = 1

        # Exclude 'total_frame_time' from individual events
        checkpoint_names = [
            name for name in self.frame_order if name != "total_frame_time"
        ]

        if not checkpoint_names:
            print("No checkpoint data to export.")
            return

        # Determine the number of frames
        num_frames = len(self.checkpoints[checkpoint_names[0]])

        # Base timestamp in microseconds (arbitrary start time)
        base_timestamp_us = 0

        # Track cumulative time for each frame
        frame_timestamps = []
        current_time_us = base_timestamp_us

        # Build frame start timestamps
        for frame_idx in range(num_frames):
            frame_timestamps.append(current_time_us)

            # Calculate frame duration
            frame_duration_us = 0
            for checkpoint_name in checkpoint_names:
                if frame_idx < len(self.checkpoints[checkpoint_name]):
                    checkpoint_time_s = self.checkpoints[checkpoint_name][frame_idx]
                    frame_duration_us += int(checkpoint_time_s * 1_000_000)

            current_time_us += frame_duration_us

        # Generate trace events for each frame and checkpoint
        for frame_idx in range(num_frames):
            frame_start_us = frame_timestamps[frame_idx]
            checkpoint_start_us = frame_start_us

            # Add frame boundary event (instant marker)
            trace_events.append(
                {
                    "name": f"Frame {frame_idx + 1}",
                    "cat": "frame",
                    "ph": "i",  # Instant event
                    "ts": frame_start_us,
                    "pid": process_id,
                    "tid": thread_id,
                    "s": "g",  # Global scope
                }
            )

            # Add duration events for each checkpoint
            for checkpoint_name in checkpoint_names:
                if frame_idx < len(self.checkpoints[checkpoint_name]):
                    duration_s = self.checkpoints[checkpoint_name][frame_idx]
                    duration_us = int(duration_s * 1_000_000)

                    # Duration event (Begin/End pair)
                    trace_events.append(
                        {
                            "name": checkpoint_name,
                            "cat": "pipeline",
                            "ph": "X",  # Complete event (duration)
                            "ts": checkpoint_start_us,
                            "dur": duration_us,
                            "pid": process_id,
                            "tid": thread_id,
                            "args": {"frame": frame_idx + 1},
                        }
                    )

                    checkpoint_start_us += duration_us

        # Add metadata
        trace_events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": process_id,
                "args": {"name": "Hailo Inference Pipeline"},
            }
        )

        trace_events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": process_id,
                "tid": thread_id,
                "args": {"name": "Main Thread"},
            }
        )

        # Create the trace file structure
        trace_data = {"traceEvents": trace_events, "displayTimeUnit": "ms"}

        # Write to file
        with open(output_path, "w") as f:
            json.dump(trace_data, f, indent=2)

        print(f"\nPerfetto trace exported to: {output_path}")
        print(f"Total frames: {num_frames}")
        print(f"Total events: {len(trace_events)}")
        print("\nVisualize the trace at:")
        print("  - Chrome: chrome://tracing")
        print("  - Perfetto UI: https://ui.perfetto.dev")
