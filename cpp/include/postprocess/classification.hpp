#pragma once
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>

#include "postprocess_base.hpp"

/// Post-processor for image classification models.
/// Displays the top-N predicted class labels on the frame.
class ImagePostprocessorClassification : public PostprocessBase {
public:
    /// @param configs  Path to JSON file with {"index": "label_name", ...} mapping.
    /// @param top_n    Number of top predictions to display.
    ImagePostprocessorClassification(
        std::pair<std::pair<float,float>, std::pair<int,int>> params,
        const std::string& configs,
        int top_n = 3
    );

    cv::Mat postprocess(const cv::Mat& frame, const OutputMap& outputs) override;

private:
    cv::Mat add_text_to_image(cv::Mat image, const std::vector<std::string>& strings);

    int top_n_;
    std::unordered_map<std::string, std::string> labels_; // index_str -> label_name
};
