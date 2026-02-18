#pragma once
#include <array>
#include <string>
#include <unordered_map>
#include <vector>

/// A single detection bounding box with confidence score.
/// Coordinates: y_min, x_min, y_max, x_max (normalized), confidence
struct DetectionBox {
  float y_min, x_min, y_max, x_max, confidence;
};

/// Per-class list of detections (used for NMS output)
using ClassDetections = std::vector<DetectionBox>;

/// NMS output: one ClassDetections entry per class index
using NmsOutput = std::vector<ClassDetections>;

/// Regular (non-NMS) tensor output: flat float32 data + shape
struct RegularOutput {
  std::vector<float> data;
  std::vector<size_t> shape;
};

/// Unified inference output - either regular float tensor or NMS per-class detections
struct InferenceOutput {
  bool is_nms = false;
  RegularOutput regular;
  NmsOutput nms;
};

/// Map from layer name to inference output
using OutputMap = std::unordered_map<std::string, InferenceOutput>;
