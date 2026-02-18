#include "postprocess/classification.hpp"

#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <ranges>
#include <stdexcept>

using json = nlohmann::json;

ImagePostprocessorClassification::ImagePostprocessorClassification(
    std::pair<std::pair<float, float>, std::pair<int, int>> /*params*/, const std::string &configs,
    int top_n)
    : top_n_(top_n) {
  std::ifstream f(configs);
  if (!f.is_open())
    throw std::runtime_error("Label file not found: " + configs);

  json j;
  try {
    f >> j;
  } catch (const json::parse_error &e) {
    throw std::runtime_error("Error parsing label JSON: " + std::string(e.what()));
  }

  for (auto &[key, val] : j.items())
    labels_[key] = val.get<std::string>();
}

cv::Mat
ImagePostprocessorClassification::add_text_to_image(cv::Mat image,
                                                    const std::vector<std::string> &strings) {
  const int font = cv::FONT_HERSHEY_SIMPLEX;
  const double scale = 1.0;
  const int thickness = 2;
  cv::Scalar color(0, 0, 255); // Red (BGR)

  int max_w = 0, max_h = 0;
  for (const auto &text : strings) {
    int baseline = 0;
    cv::Size sz = cv::getTextSize(text, font, scale, thickness, &baseline);
    max_w = std::max(max_w, sz.width);
    max_h = std::max(max_h, sz.height);
  }

  int x_start = image.cols - max_w;
  int y_start = 10 + max_h / 2;

  for (size_t i = 0; i < strings.size(); ++i) {
    int y = y_start + static_cast<int>(i) * (max_h + 10);
    cv::putText(image, strings[i], cv::Point(x_start, y), font, scale, color, thickness);
  }
  return image;
}

cv::Mat ImagePostprocessorClassification::postprocess(const cv::Mat &frame,
                                                      const OutputMap &outputs) {
  if (outputs.empty())
    throw std::runtime_error("Empty outputs provided to classification postprocessor.");

  cv::Mat out = frame.clone();
  std::vector<std::string> display_strings;

  for (const auto &[key, tensor] : outputs) {
    if (tensor.is_nms || tensor.regular.data.empty())
      continue;

    const auto &data = tensor.regular.data;

    // Compute top-N indices by sorting
    std::vector<size_t> indices(data.size());
    std::iota(indices.begin(), indices.end(), 0);
    size_t n = std::min(static_cast<size_t>(top_n_), data.size());
    std::ranges::partial_sort(indices, indices.begin() + static_cast<std::ptrdiff_t>(n),
                              [&data](size_t a, size_t b) { return data[a] > data[b]; });

    display_strings.push_back("Output: " + key);
    for (size_t i = 0; i < n; ++i) {
      size_t idx = indices[i];
      std::string idx_str = std::to_string(idx);
      auto it = labels_.find(idx_str);
      if (it == labels_.end())
        throw std::runtime_error("Label not found for index " + idx_str);
      display_strings.push_back("#" + std::to_string(i + 1) + ": " + it->second + " (" + idx_str +
                                ")");
    }
  }

  return add_text_to_image(out, display_strings);
}
