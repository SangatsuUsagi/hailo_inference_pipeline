#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <array>

#include <opencv2/opencv.hpp>

#include "postprocess_base.hpp"

/// Post-processor for object detection models using Hailo on-device NMS.
/// Converts normalized detection coordinates to image coordinates and draws boxes.
class ImagePostprocessorNmsOnHost : public PostprocessBase {
public:
    /// @param params   (scale_factors, pad_values) from preprocess_image_with_pad()
    /// @param configs  Path to model config JSON (yolov8.json format)
    ImagePostprocessorNmsOnHost(
        std::pair<std::pair<float,float>, std::pair<int,int>> params,
        const std::string& configs
    );

    cv::Mat postprocess(const cv::Mat& frame, const OutputMap& outputs) override;

private:
    cv::Mat draw_detections(cv::Mat frame, const OutputMap& outputs);

    // Denormalize a detection bounding box from model-space to original image pixels.
    // box: [y_min, x_min, y_max, x_max] (normalized, 0-1)
    // Returns: [y_min, x_min, y_max, x_max] in pixel coordinates
    std::array<float, 4> denormalize_box(const std::array<float, 4>& box) const;

    std::pair<float,float> pads_;   // (pad_h, pad_w) in original image pixels
    std::pair<float,float> scales_; // (scale_h * input_h, scale_w * input_w)

    std::unordered_map<std::string, std::string> labels_; // class_id_str -> name
    std::vector<cv::Scalar> palette_line_;                 // per-class box colors (BGR)
    std::vector<cv::Scalar> palette_text_;                 // per-class text colors (BGR)
};
