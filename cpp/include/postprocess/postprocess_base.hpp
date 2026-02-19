#pragma once
#include <opencv2/opencv.hpp>
#include "inference_types.hpp"

/// Common interface for all post-processors.
class PostprocessBase {
public:
    virtual ~PostprocessBase() = default;

    /// Apply post-processing and annotate frame. Returns annotated frame.
    virtual cv::Mat postprocess(const cv::Mat& frame, const OutputMap& outputs) = 0;
};
