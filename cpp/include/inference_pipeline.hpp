#pragma once
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <hailo/hailort.hpp>
#include <opencv2/opencv.hpp>

#include "exceptions.hpp"
#include "inference_types.hpp"

static constexpr uint32_t TIMEOUT_MS = 10000;

/// Returns true if a hailo_status represents a timeout condition.
bool is_hailo_timeout(hailo_status status);

/// Returns true if a hailo_status is any non-success Hailo error.
bool is_hailo_error(hailo_status status);

/// Preprocess image: letterbox resize + RGB conversion.
/// Returns: (preprocessed image, (scale_h, scale_w), (pad_h, pad_w))
std::tuple<cv::Mat, std::pair<float, float>, std::pair<int, int>>
preprocess_image_with_pad(const cv::Mat &image, int target_height, int target_width);

/// Manages asynchronous and synchronous inference pipelines for Hailo models.
class InferPipeline {
public:
  InferPipeline(const std::string &net_path, int batch_size, bool is_async, bool is_callback,
                bool is_nms, const std::vector<std::string> &layer_name_u8,
                const std::vector<std::string> &layer_name_u16);
  ~InferPipeline();

  InferPipeline(const InferPipeline &) = delete;
  InferPipeline &operator=(const InferPipeline &) = delete;

  /// Submit async job or run sync inference. Returns {} for async mode.
  OutputMap inference(const std::vector<cv::Mat> &dataset);

  /// Wait for async job and return results (async mode only).
  OutputMap wait_and_get_output();

  void close();

private:
  void infer_async(const std::vector<cv::Mat> &inputs);
  OutputMap infer_pipeline(const std::vector<cv::Mat> &inputs);
  OutputMap collect_output_from_bindings();

  void parse_nms_buffer(std::span<const uint8_t> buf, const std::string &name,
                        InferenceOutput &out);
  void parse_nms_buffer_raw(const std::vector<uint8_t> &raw, const std::string &name,
                            InferenceOutput &out);

  bool is_async_;
  bool is_callback_;
  bool is_nms_;
  std::vector<std::string> layer_name_u8_;
  std::vector<std::string> layer_name_u16_;

  // Async mode
  std::unique_ptr<hailort::VDevice> vdevice_;
  std::shared_ptr<hailort::InferModel> infer_model_;
  std::optional<hailort::ConfiguredInferModel> configured_infer_model_;
  hailort::ConfiguredInferModel::Bindings bindings_;
  hailort::AsyncInferJob async_job_;
  bool has_job_ = false;

  // Per-inference output buffers (kept alive across submit/wait)
  std::unordered_map<std::string, std::vector<float>> output_buffers_f32_;
  std::unordered_map<std::string, std::vector<uint8_t>> output_buffers_u8_;
  std::unordered_map<std::string, std::vector<uint16_t>> output_buffers_u16_;

  // Callback mode results
  OutputMap callback_results_;
  bool callback_error_ = false;

  // Sync mode
  std::unique_ptr<hailort::Hef> hef_;
  std::shared_ptr<hailort::ConfiguredNetworkGroup> network_group_;
  std::map<std::string, hailo_vstream_params_t> input_vstream_params_;
  std::map<std::string, hailo_vstream_params_t> output_vstream_params_;
};
