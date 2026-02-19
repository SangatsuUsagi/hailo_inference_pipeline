#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>

// ============================================================================
// Thread-safe bounded queue
// ============================================================================

template <typename T> class BoundedQueue {
public:
  explicit BoundedQueue(size_t max_size) : max_size_(max_size) {}

  /// Try to push without blocking. Returns false if full.
  bool try_push(T item) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.size() >= max_size_)
      return false;
    queue_.push(std::move(item));
    cond_.notify_one(); // wake consumers
    return true;
  }

  /// Push with timeout. Returns false on timeout.
  bool push(T item, std::chrono::milliseconds timeout = std::chrono::milliseconds(500)) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!cond_not_full_.wait_for(lock, timeout, [this] { return queue_.size() < max_size_; }))
      return false;
    queue_.push(std::move(item));
    cond_.notify_one();
    return true;
  }

  /// Pop with timeout. Returns false on timeout (item is left unchanged).
  bool pop(T &item, std::chrono::milliseconds timeout = std::chrono::milliseconds(1000)) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!cond_.wait_for(lock, timeout, [this] { return !queue_.empty(); }))
      return false;
    item = std::move(queue_.front());
    queue_.pop();
    cond_not_full_.notify_one();
    return true;
  }

  /// Try to pop without blocking. Returns false if empty.
  bool try_pop(T &item) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty())
      return false;
    item = std::move(queue_.front());
    queue_.pop();
    cond_not_full_.notify_one(); // wake producers waiting for space
    return true;
  }

private:
  std::queue<T> queue_;
  std::mutex mutex_;
  std::condition_variable cond_;          // signals: item available
  std::condition_variable cond_not_full_; // signals: slot available
  size_t max_size_;
};

// ============================================================================
// PerformanceProfiler
// ============================================================================

/// Profiles execution times across different pipeline stages.
class PerformanceProfiler {
public:
  PerformanceProfiler();

  /// Mark the beginning of a frame processing cycle.
  void start_frame();

  /// Record elapsed time since the last checkpoint.
  void checkpoint(const std::string &name);

  /// Mark the end of frame processing and record total time.
  void end_frame();

  /// Print comprehensive statistics for all recorded checkpoints.
  void print_statistics() const;

  /// Export profiling data to Perfetto/Chrome trace JSON format.
  void export_perfetto_trace(const std::string &output_path) const;

  // Direct access for thread-collected timing data
  std::unordered_map<std::string, std::vector<double>> checkpoints;
  std::vector<std::string> frame_order; // ordered unique checkpoint names

private:
  void add_checkpoint_entry(const std::string &name, double duration_s);

  std::optional<std::chrono::steady_clock::time_point> last_time_;
  std::optional<std::chrono::steady_clock::time_point> frame_start_time_;
};

// ============================================================================
// DisplayThread
// ============================================================================

/// Manages asynchronous display of frames in a separate thread.
class DisplayThread {
public:
  explicit DisplayThread(const std::string &window_name = "Output", int max_queue_size = 2,
                         PerformanceProfiler *profiler = nullptr);
  ~DisplayThread();

  void start();
  void stop();

  /// Queue a frame for display. Returns false if queue is full (frame dropped).
  bool display(const cv::Mat &frame);

  /// Check if user pressed 'q' or closed the window.
  bool is_quit_requested() const;

  /// Transfer timing data from display thread to profiler (call from main thread).
  void collect_timing_data();

private:
  void display_loop(std::stop_token stoken);

  std::string window_name_;
  BoundedQueue<std::optional<cv::Mat>> frame_queue_;
  std::atomic<bool> quit_requested_{false};
  std::jthread thread_;
  PerformanceProfiler *profiler_;

  struct TimingData {
    double queue_wait, display, key_check;
  };
  BoundedQueue<TimingData> timing_queue_;
};

// ============================================================================
// FrameReaderThread
// ============================================================================

/// Asynchronously reads frames from a video source in a separate thread.
class FrameReaderThread {
public:
  explicit FrameReaderThread(cv::VideoCapture &video_source, int max_queue_size = 4,
                             PerformanceProfiler *profiler = nullptr);
  ~FrameReaderThread();

  void start();
  void stop();

  /// Get next frame (blocks up to timeout_ms). Returns empty Mat if end-of-video or timeout.
  cv::Mat get_frame(int timeout_ms = 1000);

  bool has_error() const;

  /// Transfer timing data from reader thread to profiler (call from main thread).
  void collect_timing_data();

private:
  void read_loop(std::stop_token stoken);

  cv::VideoCapture &video_source_;
  BoundedQueue<std::optional<cv::Mat>> frame_queue_;
  std::atomic<bool> read_error_{false};
  std::jthread thread_;
  PerformanceProfiler *profiler_;

  struct TimingData {
    double read, queue_put;
  };
  BoundedQueue<TimingData> timing_queue_;
};
