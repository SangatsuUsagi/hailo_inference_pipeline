#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <ranges>
#include <string>
#include <vector>

#include <hailo/hailort.hpp>
#include <opencv2/opencv.hpp>

#include "exceptions.hpp"
#include "inference_pipeline.hpp"
#include "inference_utils.hpp"
#include "postprocess/classification.hpp"
#include "postprocess/nms_on_host.hpp"
#include "postprocess/palm_detection.hpp"
#include <CLI/CLI.hpp>

namespace fs = std::filesystem;
using clock_t_ = std::chrono::steady_clock;

struct Args {
  std::string image_path;
  std::string net_path = "./hefs/resnet_v1_50.hef";
  std::string postprocess_type = "classification";
  std::string config_path;
  std::string trace_path;
  int batch_size = 1;
  bool synchronous = false;
  bool use_callback = false;
  bool profile = false;
};

// ---------------------------------------------------------------------------
// Format and print vstream info; returns {{name, {H,W}}, ...}
// ---------------------------------------------------------------------------
struct VStreamInfo {
  std::string name;
  int H, W;
  bool is_nms;
};

static std::vector<VStreamInfo> print_vstream_infos(const std::vector<hailo_vstream_info_t> &infos,
                                                    bool is_input) {
  std::vector<VStreamInfo> result;
  for (size_t i = 0; i < infos.size(); ++i) {
    const auto &vi = infos[i];
    std::string kind = is_input ? "Input" : "Output";
    bool nms = (vi.format.order == HAILO_FORMAT_ORDER_HAILO_NMS_BY_CLASS ||
                vi.format.order == HAILO_FORMAT_ORDER_HAILO_NMS);
    std::cout << kind << " #" << i << " " << vi.name << (nms ? " [NMS]" : "") << " ("
              << vi.shape.height << "x" << vi.shape.width << "x" << vi.shape.features << ")\n";
    result.push_back(
        {vi.name, static_cast<int>(vi.shape.height), static_cast<int>(vi.shape.width), nms});
  }
  return result;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
  Args args;

  CLI::App app{"Hailo Inference Pipeline - deep learning inference on Hailo AI "
               "accelerators"};
  app.add_option("input", args.image_path, "Input image or video file")->required();
  app.add_option("-n,--net", args.net_path, "Path to HEF model file")->capture_default_str();
  app.add_option("-p,--postprocess", args.postprocess_type,
                 "Postprocess type: classification | nms_on_host | palm_detection")
      ->capture_default_str();
  app.add_option("-c,--config", args.config_path, "JSON config file path");
  app.add_option("-b,--batch-size", args.batch_size, "Batch size")->capture_default_str();
  app.add_flag("-s,--synchronous", args.synchronous, "Use synchronous inference");
  app.add_flag("--callback", args.use_callback, "Use callback mode with async inference");
  app.add_flag("--profile", args.profile, "Enable performance profiling");
  app.add_option("--trace", args.trace_path, "Export Perfetto trace JSON (requires --profile)");

  CLI11_PARSE(app, argc, argv);

  if (!args.trace_path.empty() && !args.profile) {
    std::cerr << "Error: --trace requires --profile\n";
    return 1;
  }

  PerformanceProfiler profiler;
  PerformanceProfiler *prof_ptr = args.profile ? &profiler : nullptr;

  bool is_async = !args.synchronous;
  bool is_callback = is_async && args.use_callback;
  bool is_nms = false;
  int input_H = 0, input_W = 0;

  std::unique_ptr<InferPipeline> infer;
  std::unique_ptr<cv::VideoCapture> cap;
  std::unique_ptr<DisplayThread> display_thread;
  std::unique_ptr<FrameReaderThread> reader_thread;

  bool is_image = false; // set inside try; used in post-try summary

  auto overall_start = clock_t_::now();
  int frame_count = 0;

  try {
    // --- Load HEF and inspect streams ---
    auto hef_exp = hailort::Hef::create(args.net_path);
    if (!hef_exp)
      throw std::runtime_error("Failed to load HEF: " + std::to_string(hef_exp.status()));
    auto &hef = hef_exp.value();

    auto in_infos_exp = hef.get_input_vstream_infos();
    if (!in_infos_exp)
      throw std::runtime_error("Failed to get input vstream infos");
    auto in_infos = in_infos_exp.release();

    auto out_infos_exp = hef.get_output_vstream_infos();
    if (!out_infos_exp)
      throw std::runtime_error("Failed to get output vstream infos");
    auto out_infos = out_infos_exp.release();

    if (in_infos.empty())
      throw std::runtime_error("No input streams found in model.");
    if (in_infos.size() != 1)
      throw std::runtime_error("Only single-input models are supported.");

    std::cout << "VStream infos(inputs):\n";
    auto in_meta = print_vstream_infos(in_infos, true);
    input_H = in_meta[0].H;
    input_W = in_meta[0].W;

    std::cout << "VStream infos(outputs):\n";
    auto out_meta = print_vstream_infos(out_infos, false);
    is_nms = std::ranges::any_of(out_meta, &VStreamInfo::is_nms);

    // --- Create InferPipeline ---
    infer = std::make_unique<InferPipeline>(args.net_path, args.batch_size, is_async, is_callback,
                                            is_nms, std::vector<std::string>{}, // layer_name_u8
                                            std::vector<std::string>{}          // layer_name_u16
    );

    // --- Open video/image ---
    if (!fs::exists(args.image_path))
      throw std::runtime_error("File not found: " + args.image_path);

    cap = std::make_unique<cv::VideoCapture>(args.image_path);
    if (!cap->isOpened())
      throw std::runtime_error("Cannot open: " + args.image_path);

    // Compute initial preprocessing params (scale/pad) from a dummy frame
    cv::Mat dummy(static_cast<int>(cap->get(cv::CAP_PROP_FRAME_HEIGHT)),
                  static_cast<int>(cap->get(cv::CAP_PROP_FRAME_WIDTH)), CV_8UC3,
                  cv::Scalar(0, 0, 0));
    auto [_, scale_init, pad_init] = preprocess_image_with_pad(dummy, input_H, input_W);

    using Params = std::pair<std::pair<float, float>, std::pair<int, int>>;
    Params pp_params = {scale_init, pad_init};

    // --- Select postprocessor ---
    std::unique_ptr<PostprocessBase> postprocessor;
    if (args.postprocess_type == "nms_on_host") {
      std::string cfg = args.config_path.empty() ? "./configs/yolov8.json" : args.config_path;
      postprocessor = std::make_unique<ImagePostprocessorNmsOnHost>(pp_params, cfg);
    } else if (args.postprocess_type == "palm_detection") {
      std::string cfg =
          args.config_path.empty() ? "./configs/palm_detection_full.json" : args.config_path;
      postprocessor = std::make_unique<ImagePostprocessorPalmDetection>(pp_params, cfg);
    } else { // classification (default)
      std::string cfg =
          args.config_path.empty() ? "./configs/class_names_imagenet.json" : args.config_path;
      postprocessor = std::make_unique<ImagePostprocessorClassification>(pp_params, cfg, 3);
    }

    is_image = static_cast<int>(cap->get(cv::CAP_PROP_FRAME_COUNT)) <= 1;

    // --- Start background threads (video mode only) ---
    if (!is_image) {
      reader_thread = std::make_unique<FrameReaderThread>(*cap, 4, prof_ptr);
      reader_thread->start();
      std::cout << "Frame reader thread started\n";

      display_thread = std::make_unique<DisplayThread>("Output", 2, prof_ptr);
      display_thread->start();
      std::cout << "Display thread started\n";
    }

    OutputMap last_outputs;
    bool loop = true;

    std::cout << "\nStarting inference loop...\n";
    if (args.profile)
      std::cout << "Profiling enabled.\n";
    std::cout << "Press 'q' to quit\n\n";

    while (loop) {
      if (args.profile)
        profiler.start_frame();
      ++frame_count;

      // --- Read frame ---
      cv::Mat frame;
      bool ret = false;

      if (is_image) {
        ret = cap->read(frame);
        if (args.profile)
          profiler.checkpoint("1_frame_read");
      } else {
        frame = reader_thread->get_frame();
        if (args.profile) {
          profiler.checkpoint("1_frame_read");
          reader_thread->collect_timing_data();
        }
        if (frame.empty())
          break;
        ret = true;
      }
      if (!ret || frame.empty())
        break;

      // --- Preprocess ---
      auto [input_frame, scale, pad] = preprocess_image_with_pad(frame, input_H, input_W);
      if (args.profile)
        profiler.checkpoint("2_preprocessing");

      try {
        // --- Inference ---
        auto outputs = infer->inference({input_frame});
        if (args.profile)
          profiler.checkpoint("3_inference_submit");

        if (is_image && is_async) {
          last_outputs = infer->wait_and_get_output();
          if (args.profile)
            profiler.checkpoint("4_inference_wait");
        } else if (!is_async) {
          last_outputs = outputs;
        }

        // --- Postprocess ---
        cv::Mat out_frame = frame;
        if (!last_outputs.empty())
          out_frame = postprocessor->postprocess(frame, last_outputs);
        if (args.profile)
          profiler.checkpoint("5_postprocessing");

        if (!is_image && is_async) {
          last_outputs = infer->wait_and_get_output();
          if (args.profile)
            profiler.checkpoint("6_inference_wait");
        }

        // --- Display ---
        if (is_image) {
          cv::imshow("Output", out_frame);
          if (args.profile)
            profiler.checkpoint("7_display");
          loop = false;
          cv::waitKey(0);
        } else {
          if (display_thread) {
            display_thread->display(out_frame);
            if (args.profile) {
              profiler.checkpoint("7_display_queue");
              display_thread->collect_timing_data();
            }
            if (display_thread->is_quit_requested())
              loop = false;
          }
        }

      } catch (const InferenceTimeoutError &e) {
        std::cerr << "Inference timeout (frame " << frame_count << "): " << e.what() << "\n";
        continue;
      } catch (const InferenceSubmitError &e) {
        std::cerr << "Failed to submit inference (frame " << frame_count << "): " << e.what()
                  << "\n";
        break;
      } catch (const InferenceWaitError &e) {
        std::cerr << "Failed to get results (frame " << frame_count << "): " << e.what() << "\n";
        continue;
      } catch (const InferencePipelineError &e) {
        std::cerr << "Sync inference failed (frame " << frame_count << "): " << e.what() << "\n";
        break;
      }

      if (args.profile)
        profiler.end_frame();
    }

  } catch (const InferenceError &e) {
    std::cerr << "Fatal inference error: " << e.what() << "\n";
  } catch (const std::exception &e) {
    std::cerr << "Unexpected error: " << e.what() << "\n";
    return 1;
  }

  // --- Cleanup ---
  if (reader_thread) {
    std::cout << "\nStopping frame reader thread...\n";
    reader_thread->stop();
  }
  if (display_thread) {
    std::cout << "Stopping display thread...\n";
    display_thread->stop();
  }
  if (infer)
    infer->close();
  if (cap)
    cap->release();
  cv::destroyAllWindows();

  // --- Performance summary ---
  auto overall_end = clock_t_::now();
  double overall_elapsed = std::chrono::duration<double>(overall_end - overall_start).count();

  // is_image was set before cap was released; use it directly here
  if (!is_image) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "BASIC PERFORMANCE SUMMARY\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "Total execution time: " << overall_elapsed << " seconds\n";
    std::cout << "Total frames processed: " << frame_count << "\n";
    if (overall_elapsed > 0)
      std::cout << "Overall throughput: " << frame_count / overall_elapsed << " FPS\n";
    std::cout << std::string(80, '=') << "\n";
  }

  if (args.profile) {
    profiler.print_statistics();
    if (!args.trace_path.empty())
      profiler.export_perfetto_trace(args.trace_path);
  }

  return 0;
}
