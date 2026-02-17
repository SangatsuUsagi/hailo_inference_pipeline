#include "postprocess/palm_detection.hpp"

#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <numeric>

// ---------------------------------------------------------------------------
// Anchor generation
// ---------------------------------------------------------------------------

float ImagePostprocessorPalmDetection::calculate_scale(
    float min_scale, float max_scale, int stride_index, int num_strides)
{
    if (num_strides == 1)
        return (max_scale + min_scale) * 0.5f;
    return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1.0f);
}

std::vector<Anchor> ImagePostprocessorPalmDetection::generate_anchors(
    const nlohmann::json& opts)
{
    int strides_size = static_cast<int>(opts["strides"].size());
    assert(opts["num_layers"].get<int>() == strides_size);

    std::vector<Anchor> anchors;
    int layer_id = 0;

    while (layer_id < strides_size) {
        std::vector<float> anchor_height, anchor_width, aspect_ratios, scales;

        int last_same = layer_id;
        while (last_same < strides_size &&
               opts["strides"][last_same].get<int>() == opts["strides"][layer_id].get<int>())
        {
            float scale = calculate_scale(
                opts["min_scale"].get<float>(), opts["max_scale"].get<float>(),
                last_same, strides_size);

            if (last_same == 0 && opts["reduce_boxes_in_lowest_layer"].get<bool>()) {
                aspect_ratios.insert(aspect_ratios.end(), {1.0f, 2.0f, 0.5f});
                scales.insert(scales.end(), {0.1f, scale, scale});
            } else {
                for (float ar : opts["aspect_ratios"]) {
                    aspect_ratios.push_back(ar);
                    scales.push_back(scale);
                }
                if (opts["interpolated_scale_aspect_ratio"].get<float>() > 0.0f) {
                    float scale_next = (last_same == strides_size - 1) ? 1.0f
                        : calculate_scale(opts["min_scale"].get<float>(),
                                          opts["max_scale"].get<float>(),
                                          last_same + 1, strides_size);
                    scales.push_back(std::sqrt(scale * scale_next));
                    aspect_ratios.push_back(
                        opts["interpolated_scale_aspect_ratio"].get<float>());
                }
            }
            ++last_same;
        }

        // Anchor dimensions
        std::vector<float> ah, aw;
        for (size_t i = 0; i < aspect_ratios.size(); ++i) {
            float sq = std::sqrt(aspect_ratios[i]);
            ah.push_back(scales[i] / sq);
            aw.push_back(scales[i] * sq);
        }

        int stride = opts["strides"][layer_id].get<int>();
        int fh = static_cast<int>(std::ceil(opts["input_size_height"].get<float>() / stride));
        int fw = static_cast<int>(std::ceil(opts["input_size_width"].get<float>()  / stride));

        for (int y = 0; y < fh; ++y) {
            for (int x = 0; x < fw; ++x) {
                for (size_t a = 0; a < ah.size(); ++a) {
                    Anchor anc;
                    anc.x_center = (x + opts["anchor_offset_x"].get<float>()) / fw;
                    anc.y_center = (y + opts["anchor_offset_y"].get<float>()) / fh;
                    if (opts["fixed_anchor_size"].get<bool>()) {
                        anc.w = 1.0f; anc.h = 1.0f;
                    } else {
                        anc.w = aw[a]; anc.h = ah[a];
                    }
                    anchors.push_back(anc);
                }
            }
        }
        layer_id = last_same;
    }
    return anchors;
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

ImagePostprocessorPalmDetection::ImagePostprocessorPalmDetection(
    std::pair<std::pair<float,float>, std::pair<int,int>> params,
    const std::string& configs)
    : scale_(params.first.first)
    , pad_(params.second)
{
    std::ifstream f(configs, std::ios::in);
    if (!f.is_open())
        throw std::runtime_error("Config file not found: " + configs);

    nlohmann::json model_info;
    try { f >> model_info; }
    catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error("Error parsing config JSON: " + std::string(e.what()));
    }

    anchors_      = generate_anchors(model_info[0]);
    model_configs_ = model_info[1];
}

// ---------------------------------------------------------------------------
// Box decoding
// ---------------------------------------------------------------------------

ImagePostprocessorPalmDetection::Detection
ImagePostprocessorPalmDetection::decode_box(
    const float* raw, size_t anchor_idx) const
{
    const Anchor& anc = anchors_[anchor_idx];
    float xs = model_configs_["x_scale"].get<float>();
    float ys = model_configs_["y_scale"].get<float>();
    float ws = model_configs_["w_scale"].get<float>();
    float hs = model_configs_["h_scale"].get<float>();
    int nk    = model_configs_["num_keypoints"].get<int>();

    float cx = raw[0] / xs * anc.w + anc.x_center;
    float cy = raw[1] / ys * anc.h + anc.y_center;
    float w  = raw[2] / ws * anc.w;
    float h  = raw[3] / hs * anc.h;

    Detection d;
    d.y_min = cy - h / 2.0f;
    d.x_min = cx - w / 2.0f;
    d.y_max = cy + h / 2.0f;
    d.x_max = cx + w / 2.0f;

    d.keypoints.resize(nk * 2);
    for (int k = 0; k < nk; ++k) {
        d.keypoints[2*k]   = raw[4 + 2*k]     / xs * anc.w + anc.x_center;
        d.keypoints[2*k+1] = raw[4 + 2*k + 1] / ys * anc.h + anc.y_center;
    }
    d.score = 0.0f;
    return d;
}

// ---------------------------------------------------------------------------
// Tensors -> detections
// ---------------------------------------------------------------------------

std::vector<ImagePostprocessorPalmDetection::Detection>
ImagePostprocessorPalmDetection::tensors_to_detections(
    const std::vector<float>& raw_boxes,
    const std::vector<float>& raw_scores,
    size_t num_anchors) const
{
    float thresh  = model_configs_["score_clipping_thresh"].get<float>();
    float min_sc  = model_configs_["min_score_thresh"].get<float>();
    int num_coords = model_configs_["num_coords"].get<int>();

    std::vector<Detection> result;
    for (size_t i = 0; i < num_anchors; ++i) {
        float raw_s = raw_scores[i];
        raw_s = std::max(-thresh, std::min(thresh, raw_s));
        float score = 1.0f / (1.0f + std::exp(-raw_s));
        if (score < min_sc) continue;

        Detection d = decode_box(raw_boxes.data() + i * num_coords, i);
        d.score = score;
        result.push_back(d);
    }
    return result;
}

// ---------------------------------------------------------------------------
// IoU
// ---------------------------------------------------------------------------

float ImagePostprocessorPalmDetection::iou(
    const Detection& a, const Detection& b) const
{
    float inter_y1 = std::max(a.y_min, b.y_min);
    float inter_x1 = std::max(a.x_min, b.x_min);
    float inter_y2 = std::min(a.y_max, b.y_max);
    float inter_x2 = std::min(a.x_max, b.x_max);

    float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float inter   = inter_h * inter_w;

    float area_a = (a.y_max - a.y_min) * (a.x_max - a.x_min);
    float area_b = (b.y_max - b.y_min) * (b.x_max - b.x_min);
    float uni    = area_a + area_b - inter;

    return (uni > 0.0f) ? inter / uni : 0.0f;
}

// ---------------------------------------------------------------------------
// Weighted NMS
// ---------------------------------------------------------------------------

std::vector<ImagePostprocessorPalmDetection::Detection>
ImagePostprocessorPalmDetection::weighted_nms(
    std::vector<Detection> dets) const
{
    if (dets.empty()) return {};

    float suppress_thresh = model_configs_["min_suppression_threshold"].get<float>();
    int num_coords        = model_configs_["num_coords"].get<int>();

    // Sort by score descending
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b){ return a.score > b.score; });

    std::vector<bool> remaining(dets.size(), true);
    std::vector<Detection> output;

    for (size_t i = 0; i < dets.size(); ++i) {
        if (!remaining[i]) continue;

        // Find overlapping detections
        std::vector<size_t> overlapping;
        overlapping.push_back(i);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (!remaining[j]) continue;
            if (iou(dets[i], dets[j]) > suppress_thresh) {
                overlapping.push_back(j);
                remaining[j] = false;
            }
        }
        remaining[i] = false;

        // Weighted average
        Detection weighted = dets[i];
        if (overlapping.size() > 1) {
            float total_score = 0.0f;
            for (size_t idx : overlapping) total_score += dets[idx].score;

            // Zero out coords
            weighted.y_min = weighted.x_min = weighted.y_max = weighted.x_max = 0.0f;
            std::fill(weighted.keypoints.begin(), weighted.keypoints.end(), 0.0f);

            for (size_t idx : overlapping) {
                float w = dets[idx].score;
                weighted.y_min += w * dets[idx].y_min;
                weighted.x_min += w * dets[idx].x_min;
                weighted.y_max += w * dets[idx].y_max;
                weighted.x_max += w * dets[idx].x_max;
                for (size_t k = 0; k < dets[idx].keypoints.size(); ++k)
                    weighted.keypoints[k] += w * dets[idx].keypoints[k];
            }
            weighted.y_min /= total_score;
            weighted.x_min /= total_score;
            weighted.y_max /= total_score;
            weighted.x_max /= total_score;
            for (float& kp : weighted.keypoints) kp /= total_score;
            weighted.score = total_score / overlapping.size();
        }
        output.push_back(weighted);
    }
    return output;
}

// ---------------------------------------------------------------------------
// Denormalize
// ---------------------------------------------------------------------------

void ImagePostprocessorPalmDetection::denormalize_detections(
    std::vector<Detection>& dets) const
{
    float xs = model_configs_["x_scale"].get<float>();
    float factor = scale_ * xs;

    for (auto& d : dets) {
        d.y_min = d.y_min * factor - pad_.first;
        d.x_min = d.x_min * factor - pad_.second;
        d.y_max = d.y_max * factor - pad_.first;
        d.x_max = d.x_max * factor - pad_.second;

        for (size_t k = 0; k < d.keypoints.size(); k += 2) {
            d.keypoints[k]     = d.keypoints[k]     * factor - pad_.second; // x
            d.keypoints[k + 1] = d.keypoints[k + 1] * factor - pad_.first;  // y
        }
    }
}

// ---------------------------------------------------------------------------
// postprocess_palm_detection
// ---------------------------------------------------------------------------

std::vector<ImagePostprocessorPalmDetection::Detection>
ImagePostprocessorPalmDetection::postprocess_palm_detection(
    const OutputMap& outputs) const
{
    const std::array<std::string, 4> required_keys = {
        "palm_detection_full/conv29",
        "palm_detection_full/conv34",
        "palm_detection_full/conv30",
        "palm_detection_full/conv35",
    };
    for (const auto& k : required_keys) {
        if (outputs.find(k) == outputs.end())
            throw std::runtime_error("Missing expected output tensor: " + k);
    }

    // conv29: (864, 1) scores part 1
    // conv34: (1152, 1) scores part 2
    // conv30: (864, 18) boxes part 1
    // conv35: (1152, 18) boxes part 2
    const auto& s1 = outputs.at("palm_detection_full/conv29").regular.data; // 864
    const auto& s2 = outputs.at("palm_detection_full/conv34").regular.data; // 1152
    const auto& b1 = outputs.at("palm_detection_full/conv30").regular.data; // 864*18
    const auto& b2 = outputs.at("palm_detection_full/conv35").regular.data; // 1152*18

    // Concatenate: scores = [s2 | s1], boxes = [b2 | b1] (matching Python order)
    std::vector<float> scores;
    scores.reserve(s2.size() + s1.size());
    scores.insert(scores.end(), s2.begin(), s2.end());
    scores.insert(scores.end(), s1.begin(), s1.end());

    std::vector<float> boxes;
    boxes.reserve(b2.size() + b1.size());
    boxes.insert(boxes.end(), b2.begin(), b2.end());
    boxes.insert(boxes.end(), b1.begin(), b1.end());

    size_t num_anchors = scores.size();
    auto dets = tensors_to_detections(boxes, scores, num_anchors);
    return weighted_nms(std::move(dets));
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

cv::Mat ImagePostprocessorPalmDetection::draw_detections(
    cv::Mat image, const std::vector<Detection>& dets) const
{
    for (const auto& d : dets) {
        cv::Point tl(static_cast<int>(d.x_min), static_cast<int>(d.y_min));
        cv::Point br(static_cast<int>(d.x_max), static_cast<int>(d.y_max));
        cv::rectangle(image, tl, br, cv::Scalar(0, 0, 255), 4);

        size_t n_kp = d.keypoints.size() / 2;
        for (size_t k = 0; k < n_kp; ++k) {
            int kx = static_cast<int>(d.keypoints[2*k]);
            int ky = static_cast<int>(d.keypoints[2*k + 1]);
            cv::circle(image, cv::Point(kx, ky), 10, cv::Scalar(0, 0, 255), 2);
        }
    }
    return image;
}

cv::Mat ImagePostprocessorPalmDetection::draw_roi(
    cv::Mat image, const std::vector<Detection>& dets) const
{
    int kp1_idx = model_configs_["kp1"].get<int>();
    int kp2_idx = model_configs_["kp2"].get<int>();
    float dy     = model_configs_["dy"].get<float>();
    float dscale = model_configs_["dscale"].get<float>();
    float theta0 = model_configs_["theta0"].get<float>();

    for (const auto& d : dets) {
        float xc = (d.x_min + d.x_max) / 2.0f;
        float yc = (d.y_min + d.y_max) / 2.0f;
        float sc = d.x_max - d.x_min;

        yc += dy * sc;
        sc *= dscale;

        float x0 = d.keypoints[2 * kp1_idx];
        float y0 = d.keypoints[2 * kp1_idx + 1];
        float x1 = d.keypoints[2 * kp2_idx];
        float y1 = d.keypoints[2 * kp2_idx + 1];
        float theta = std::atan2(y0 - y1, x0 - x1) - theta0;

        float cos_t = std::cos(theta);
        float sin_t = std::sin(theta);
        float hs = sc / 2.0f;

        // 4 corners of the ROI (rotated rectangle)
        std::array<cv::Point, 4> pts;
        float cx_arr[4] = {-1,  1, -1,  1};
        float cy_arr[4] = {-1, -1,  1,  1};
        for (int i = 0; i < 4; ++i) {
            float px = hs * cx_arr[i];
            float py = hs * cy_arr[i];
            pts[i] = cv::Point(
                static_cast<int>(cos_t * px - sin_t * py + xc),
                static_cast<int>(sin_t * px + cos_t * py + yc));
        }

        // Draw quadrilateral: TL-TR, TL-BL, TR-BR, BL-BR
        cv::line(image, pts[0], pts[1], cv::Scalar(255, 0, 0), 3);
        cv::line(image, pts[0], pts[2], cv::Scalar(255, 0, 0), 3);
        cv::line(image, pts[1], pts[3], cv::Scalar(255, 0, 0), 3);
        cv::line(image, pts[2], pts[3], cv::Scalar(255, 0, 0), 3);
    }
    return image;
}

// ---------------------------------------------------------------------------
// postprocess (entry point)
// ---------------------------------------------------------------------------

cv::Mat ImagePostprocessorPalmDetection::postprocess(
    const cv::Mat& frame, const OutputMap& outputs)
{
    cv::Mat result = frame.clone();
    auto filtered = postprocess_palm_detection(outputs);

    if (!filtered.empty()) {
        denormalize_detections(filtered);
        result = draw_detections(result, filtered);
        result = draw_roi(result, filtered);
    }
    return result;
}
