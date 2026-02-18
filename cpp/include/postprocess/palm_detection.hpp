#pragma once
#include <string>
#include <vector>
#include <array>

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

#include "postprocess_base.hpp"

/// Anchor box for palm detection
struct Anchor { float x_center, y_center, w, h; };

/// Post-processor for palm detection models (Google MediaPipe style).
class ImagePostprocessorPalmDetection : public PostprocessBase {
public:
    /// @param params   (scale_factors, pad_values) from preprocess_image_with_pad()
    /// @param configs  Path to palm_detection_full.json
    ImagePostprocessorPalmDetection(
        std::pair<std::pair<float,float>, std::pair<int,int>> params,
        const std::string& configs
    );

    cv::Mat postprocess(const cv::Mat& frame, const OutputMap& outputs) override;

private:
    // --- Core processing ---
    struct Detection {
        float y_min, x_min, y_max, x_max;
        std::vector<float> keypoints; // flat: kp0_x, kp0_y, kp1_x, kp1_y, ...
        float score;
    };

    std::vector<Detection> postprocess_palm_detection(const OutputMap& outputs) const;
    std::vector<Detection> tensors_to_detections(
        const std::vector<float>& raw_boxes,   // shape: [num_anchors, num_coords]
        const std::vector<float>& raw_scores,  // shape: [num_anchors, 1]
        size_t num_anchors
    ) const;
    Detection decode_box(const float* raw_box, size_t anchor_idx) const;
    std::vector<Detection> weighted_nms(std::vector<Detection> detections) const;

    // --- Coordinate utilities ---
    void denormalize_detections(std::vector<Detection>& dets) const;
    float iou(const Detection& a, const Detection& b) const;

    // --- Visualization ---
    cv::Mat draw_detections(cv::Mat image, const std::vector<Detection>& dets) const;
    cv::Mat draw_roi(cv::Mat image, const std::vector<Detection>& dets) const;

    // --- Anchor generation ---
    static std::vector<Anchor> generate_anchors(const nlohmann::json& options);
    static float calculate_scale(float min_scale, float max_scale,
                                  int stride_index, int num_strides);

    // Config values
    float scale_;
    std::pair<int,int> pad_;     // (pad_h, pad_w) in original image pixels
    std::vector<Anchor> anchors_;
    nlohmann::json model_configs_;
};
