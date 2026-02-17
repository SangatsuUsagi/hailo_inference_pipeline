#include "inference_utils.hpp"

#include <iostream>
#include <iomanip>
#include <numeric>
#include <cmath>
#include <fstream>
#include <ranges>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace std::chrono;

// ============================================================================
// PerformanceProfiler
// ============================================================================

PerformanceProfiler::PerformanceProfiler() = default;

void PerformanceProfiler::start_frame() {
    frame_start_time_ = steady_clock::now();
    last_time_ = frame_start_time_;
}

void PerformanceProfiler::checkpoint(const std::string& name) {
    auto now = steady_clock::now();
    if (last_time_) {
        double elapsed = duration<double>(now - *last_time_).count();
        add_checkpoint_entry(name, elapsed);
    }
    last_time_ = now;
}

void PerformanceProfiler::end_frame() {
    if (frame_start_time_) {
        auto now = steady_clock::now();
        double total = duration<double>(now - *frame_start_time_).count();
        add_checkpoint_entry("total_frame_time", total);
    }
}

void PerformanceProfiler::add_checkpoint_entry(const std::string& name, double dur) {
    checkpoints[name].push_back(dur);
    if (std::ranges::find(frame_order, name) == frame_order.end())
        frame_order.push_back(name);
}

void PerformanceProfiler::print_statistics() const {
    constexpr int W = 92;
    std::cout << "\n" << std::string(W, '=') << "\n";
    std::cout << "PERFORMANCE PROFILING RESULTS\n";
    std::cout << std::string(W, '=') << "\n";

    if (checkpoints.empty()) {
        std::cout << "No profiling data collected.\n";
        return;
    }

    std::cout << std::left << std::setw(30) << "Checkpoint"
              << std::right
              << std::setw(8)  << "Count"
              << std::setw(12) << "Min(ms)"
              << std::setw(12) << "Max(ms)"
              << std::setw(12) << "Mean(ms)"
              << std::setw(12) << "Var(ms²)"
              << "\n";
    std::cout << std::string(W, '-') << "\n";

    // Print in frame_order for reproducible ordering
    for (const auto& name : frame_order) {
        auto it = checkpoints.find(name);
        if (it == checkpoints.end() || it->second.empty()) continue;

        const auto& times = it->second;
        double min_t = std::ranges::min(times) * 1000.0;
        double max_t = std::ranges::max(times) * 1000.0;
        double mean  = std::accumulate(times.begin(), times.end(), 0.0) / times.size() * 1000.0;
        double var   = 0.0;
        for (double t : times) {
            double d = t * 1000.0 - mean;
            var += d * d;
        }
        var /= times.size();

        std::cout << std::left  << std::setw(30) << name
                  << std::right
                  << std::setw(8)  << times.size()
                  << std::fixed << std::setprecision(3)
                  << std::setw(12) << min_t
                  << std::setw(12) << max_t
                  << std::setw(12) << mean
                  << std::setw(12) << var
                  << "\n";
    }

    std::cout << std::string(W, '=') << "\n";

    auto it = checkpoints.find("total_frame_time");
    if (it != checkpoints.end() && !it->second.empty()) {
        double avg = std::accumulate(it->second.begin(), it->second.end(), 0.0)
                     / it->second.size();
        double fps = avg > 0 ? 1.0 / avg : 0.0;
        std::cout << "\nAverage Frame Processing Time: " << avg * 1000.0 << " ms\n";
        std::cout << "Average FPS (from frame time): " << fps << "\n";
    }

    std::cout << std::string(W, '=') << "\n\n";
}

void PerformanceProfiler::export_perfetto_trace(const std::string& output_path) const {
    if (checkpoints.empty()) {
        std::cout << "No profiling data to export.\n";
        return;
    }

    const int process_id       = 1;
    const int main_thread_id   = 1;
    const int display_thread_id = 2;
    const int capture_thread_id = 3;

    std::vector<std::string> main_cps, display_cps, capture_cps;
    for (const auto& name : frame_order) {
        if (name == "total_frame_time") continue;
        if (name.rfind("display_", 0) == 0)      display_cps.push_back(name);
        else if (name.rfind("capture_", 0) == 0) capture_cps.push_back(name);
        else                                       main_cps.push_back(name);
    }

    if (main_cps.empty() && display_cps.empty() && capture_cps.empty()) {
        std::cout << "No checkpoint data to export.\n";
        return;
    }

    // Determine frame count
    const auto& first_cp = main_cps.empty() ?
        (display_cps.empty() ? capture_cps[0] : display_cps[0]) : main_cps[0];
    size_t num_frames = checkpoints.at(first_cp).size();

    json trace_events = json::array();

    // Build per-frame timestamps from main thread durations
    std::vector<long long> frame_ts(num_frames, 0);
    long long cur_us = 0;
    for (size_t f = 0; f < num_frames; ++f) {
        frame_ts[f] = cur_us;
        long long frame_dur = 0;
        for (const auto& cp : main_cps) {
            auto it = checkpoints.find(cp);
            if (it != checkpoints.end() && f < it->second.size())
                frame_dur += static_cast<long long>(it->second[f] * 1e6);
        }
        cur_us += frame_dur;
    }

    auto add_events = [&](const std::vector<std::string>& cps, int tid,
                          const std::string& cat, const std::string& prefix)
    {
        for (size_t f = 0; f < num_frames; ++f) {
            long long start_us = frame_ts[f];

            // Frame instant marker (main thread only)
            if (tid == main_thread_id) {
                trace_events.push_back({
                    {"name", "Frame " + std::to_string(f + 1)},
                    {"cat", "frame"}, {"ph", "i"},
                    {"ts", start_us}, {"pid", process_id}, {"tid", tid}, {"s", "g"}
                });
            }

            for (const auto& cp : cps) {
                auto it = checkpoints.find(cp);
                if (it == checkpoints.end() || f >= it->second.size()) continue;
                long long dur_us = static_cast<long long>(it->second[f] * 1e6);

                std::string display_name = cp;
                if (!prefix.empty() && display_name.rfind(prefix, 0) == 0)
                    display_name = display_name.substr(prefix.size());

                trace_events.push_back({
                    {"name", display_name}, {"cat", cat}, {"ph", "X"},
                    {"ts", start_us}, {"dur", dur_us},
                    {"pid", process_id}, {"tid", tid},
                    {"args", {{"frame", f + 1}}}
                });
                start_us += dur_us;
            }
        }
    };

    add_events(main_cps,    main_thread_id,    "pipeline", "");
    add_events(capture_cps, capture_thread_id, "capture",  "capture_");
    add_events(display_cps, display_thread_id, "display",  "display_");

    // Metadata
    trace_events.push_back({
        {"name", "process_name"}, {"ph", "M"}, {"pid", process_id},
        {"args", {{"name", "Hailo Inference Pipeline"}}}
    });
    trace_events.push_back({
        {"name", "thread_name"}, {"ph", "M"}, {"pid", process_id}, {"tid", main_thread_id},
        {"args", {{"name", "Main Thread"}}}
    });
    if (!capture_cps.empty())
        trace_events.push_back({
            {"name", "thread_name"}, {"ph", "M"}, {"pid", process_id}, {"tid", capture_thread_id},
            {"args", {{"name", "Capture Thread"}}}
        });
    if (!display_cps.empty())
        trace_events.push_back({
            {"name", "thread_name"}, {"ph", "M"}, {"pid", process_id}, {"tid", display_thread_id},
            {"args", {{"name", "Display Thread"}}}
        });

    json trace = {{"traceEvents", trace_events}, {"displayTimeUnit", "ms"}};

    std::ofstream ofs(output_path);
    ofs << trace.dump(2);

    std::cout << "\nPerfetto trace exported to: " << output_path << "\n";
    std::cout << "Total frames: " << num_frames << "\n";
    std::cout << "Total events: " << trace_events.size() << "\n";
    std::cout << "\nVisualize at:\n"
              << "  - Chrome: chrome://tracing\n"
              << "  - Perfetto UI: https://ui.perfetto.dev\n";
}

// ============================================================================
// DisplayThread
// ============================================================================

DisplayThread::DisplayThread(
    const std::string& window_name,
    int max_queue_size,
    PerformanceProfiler* profiler)
    : window_name_(window_name)
    , frame_queue_(max_queue_size)
    , profiler_(profiler)
    , timing_queue_(256)
{}

DisplayThread::~DisplayThread() {
    stop();
}

void DisplayThread::start() {
    quit_requested_ = false;
    thread_ = std::jthread([this](std::stop_token st) { display_loop(st); });
}

void DisplayThread::display_loop(std::stop_token stoken) {
    while (!stoken.stop_requested()) {
        std::optional<cv::Mat> item;
        auto t0 = steady_clock::now();
        bool got = frame_queue_.pop(item, milliseconds(1000));
        double queue_wait = duration<double>(steady_clock::now() - t0).count();

        if (!got) {
            // Check window close
            if (cv::getWindowProperty(window_name_, cv::WND_PROP_VISIBLE) < 1) {
                quit_requested_ = true;
                break;
            }
            continue;
        }

        if (!item.has_value()) break; // sentinel

        auto t1 = steady_clock::now();
        cv::imshow(window_name_, *item);
        double display_t = duration<double>(steady_clock::now() - t1).count();

        auto t2 = steady_clock::now();
        int key = cv::waitKey(1) & 0xFF;
        double key_t = duration<double>(steady_clock::now() - t2).count();

        if (key == 'q') quit_requested_ = true;

        if (profiler_) {
            timing_queue_.try_push({queue_wait, display_t, key_t});
        }
    }
}

bool DisplayThread::display(const cv::Mat& frame) {
    return frame_queue_.try_push(std::make_optional(frame.clone()));
}

bool DisplayThread::is_quit_requested() const {
    return quit_requested_;
}

void DisplayThread::collect_timing_data() {
    if (!profiler_) return;
    TimingData td;
    if (!timing_queue_.try_pop(td)) return;

    auto add = [this](const std::string& name, double dur) {
        profiler_->checkpoints[name].push_back(dur);
        if (std::ranges::find(profiler_->frame_order, name) == profiler_->frame_order.end())
            profiler_->frame_order.push_back(name);
    };
    add("display_queue_wait", td.queue_wait);
    add("display_display",    td.display);
    add("display_key_check",  td.key_check);
}

void DisplayThread::stop() {
    frame_queue_.try_push(std::nullopt); // sentinel to wake blocked pop
    thread_.request_stop();
    if (thread_.joinable()) thread_.join();
    cv::destroyAllWindows();
}

// ============================================================================
// FrameReaderThread
// ============================================================================

FrameReaderThread::FrameReaderThread(
    cv::VideoCapture& video_source,
    int max_queue_size,
    PerformanceProfiler* profiler)
    : video_source_(video_source)
    , frame_queue_(max_queue_size)
    , profiler_(profiler)
    , timing_queue_(256)
{}

FrameReaderThread::~FrameReaderThread() {
    stop();
}

void FrameReaderThread::start() {
    read_error_ = false;
    thread_ = std::jthread([this](std::stop_token st) { read_loop(st); });
}

void FrameReaderThread::read_loop(std::stop_token stoken) {
    while (!stoken.stop_requested()) {
        cv::Mat frame;
        auto t0 = steady_clock::now();
        bool ok = video_source_.read(frame);
        double read_t = duration<double>(steady_clock::now() - t0).count();

        if (!ok || frame.empty()) {
            read_error_ = !ok;
            frame_queue_.push(std::nullopt, milliseconds(1000)); // end-of-video sentinel
            break;
        }

        auto t1 = steady_clock::now();
        bool pushed = frame_queue_.push(std::make_optional(frame), milliseconds(1000));
        double put_t = duration<double>(steady_clock::now() - t1).count();

        if (!pushed) continue; // drop frame if queue stayed full

        if (profiler_) {
            timing_queue_.try_push({read_t, put_t});
        }
    }
}

cv::Mat FrameReaderThread::get_frame(int timeout_ms) {
    std::optional<cv::Mat> item;
    bool got = frame_queue_.pop(item, std::chrono::milliseconds(timeout_ms));
    if (!got || !item.has_value()) return {}; // empty Mat signals end/timeout
    return *item;
}

bool FrameReaderThread::has_error() const {
    return read_error_;
}

void FrameReaderThread::collect_timing_data() {
    if (!profiler_) return;
    TimingData td;
    if (!timing_queue_.try_pop(td)) return;

    auto add = [this](const std::string& name, double dur) {
        profiler_->checkpoints[name].push_back(dur);
        if (std::ranges::find(profiler_->frame_order, name) == profiler_->frame_order.end())
            profiler_->frame_order.push_back(name);
    };
    add("capture_read",      td.read);
    add("capture_queue_put", td.queue_put);
}

void FrameReaderThread::stop() {
    thread_.request_stop();
    if (thread_.joinable()) thread_.join();
}
