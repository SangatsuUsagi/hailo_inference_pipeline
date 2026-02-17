#include "postprocess/nms_on_host.hpp"

#include <fstream>
#include <stdexcept>
#include <cmath>
#include <ranges>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Color palette generation
// ---------------------------------------------------------------------------

static cv::Scalar hsv_to_bgr(float h, float s, float v) {
    // h in [0,1], s in [0,1], v in [0,1]
    float r, g, b;
    int hi = static_cast<int>(h * 6.0f) % 6;
    float f = h * 6.0f - std::floor(h * 6.0f);
    float p = v * (1.0f - s);
    float q = v * (1.0f - f * s);
    float t = v * (1.0f - (1.0f - f) * s);

    switch (hi) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        default: r = v; g = p; b = q; break;
    }
    return cv::Scalar(b * 255.0f, g * 255.0f, r * 255.0f); // BGR
}

static std::pair<std::vector<cv::Scalar>, std::vector<cv::Scalar>>
generate_palettes(int num_classes) {
    std::vector<cv::Scalar> line_palette, text_palette;
    for (int i = 0; i < num_classes; ++i) {
        float h = static_cast<float>(i) / num_classes;
        cv::Scalar rgb = hsv_to_bgr(h, 1.0f, 1.0f);

        // Complementary: max + min - each channel
        double b = rgb[0], g = rgb[1], r = rgb[2];
        double mx = std::ranges::max({b, g, r});
        double mn = std::ranges::min({b, g, r});
        cv::Scalar comp(mx + mn - b, mx + mn - g, mx + mn - r);

        line_palette.push_back(rgb);
        text_palette.push_back(comp);
    }
    return {line_palette, text_palette};
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

ImagePostprocessorNmsOnHost::ImagePostprocessorNmsOnHost(
    std::pair<std::pair<float,float>, std::pair<int,int>> params,
    const std::string& configs)
{
    std::ifstream f(configs);
    if (!f.is_open())
        throw std::runtime_error("Config file not found: " + configs);

    json model_info;
    try { f >> model_info; }
    catch (const json::parse_error& e) {
        throw std::runtime_error("Error parsing config JSON: " + std::string(e.what()));
    }

    const auto& model_cfg    = model_info[0];
    const auto& input_shape  = model_cfg["preprocessing"]["input_shape"];
    float in_h = input_shape[0].get<float>();
    float in_w = input_shape[1].get<float>();

    pads_   = {static_cast<float>(params.second.first),
               static_cast<float>(params.second.second)};
    scales_ = {in_h * params.first.first,
               in_w * params.first.second};

    const auto& label_map = model_info[1];
    for (auto& [key, val] : label_map.items())
        labels_[key] = val.get<std::string>();

    auto [lp, tp] = generate_palettes(static_cast<int>(labels_.size()));
    palette_line_ = std::move(lp);
    palette_text_ = std::move(tp);
}

// ---------------------------------------------------------------------------
// Denormalization
// ---------------------------------------------------------------------------

std::array<float, 4> ImagePostprocessorNmsOnHost::denormalize_box(
    const std::array<float, 4>& box) const
{
    // box: [y_min, x_min, y_max, x_max]  (normalized 0-1 from model output)
    // scales_: (scale_h, scale_w) = input_h * scale, input_w * scale
    // pads_: (pad_h, pad_w) in original image pixels
    return {
        box[0] * scales_.first  - pads_.first,
        box[1] * scales_.second - pads_.second,
        box[2] * scales_.first  - pads_.first,
        box[3] * scales_.second - pads_.second
    };
}

// ---------------------------------------------------------------------------
// draw_detections
// ---------------------------------------------------------------------------

cv::Mat ImagePostprocessorNmsOnHost::draw_detections(
    cv::Mat frame, const OutputMap& outputs)
{
    if (outputs.empty()) return frame;

    const auto& [first_name, first_output] = *outputs.begin();
    if (!first_output.is_nms) return frame;

    const auto& nms = first_output.nms;

    for (size_t class_id = 0; class_id < nms.size(); ++class_id) {
        for (const auto& det : nms[class_id]) {
            // Denormalize
            std::array<float, 4> box = {det.y_min, det.x_min, det.y_max, det.x_max};
            auto db = denormalize_box(box);

            cv::Point top_left(static_cast<int>(db[1]), static_cast<int>(db[0]));
            cv::Point bottom_right(static_cast<int>(db[3]), static_cast<int>(db[2]));

            const cv::Scalar& line_color = palette_line_[class_id % palette_line_.size()];
            const cv::Scalar& text_color = palette_text_[class_id % palette_text_.size()];

            // Draw bounding box
            cv::rectangle(frame, top_left, bottom_right, line_color, 4);

            // Prepare label
            auto it = labels_.find(std::to_string(class_id));
            std::string class_name = (it != labels_.end()) ? it->second : "unknown";
            char conf_buf[32];
            std::snprintf(conf_buf, sizeof(conf_buf), "%.2f", det.confidence);
            std::string label = class_name + " " + conf_buf;

            const double font_scale = 0.8;
            const int text_thickness = 2;
            int baseline = 0;
            cv::Size text_sz = cv::getTextSize(
                label, cv::FONT_HERSHEY_SIMPLEX, font_scale, text_thickness, &baseline);

            cv::Point bg_tl = top_left;
            cv::Point bg_br(top_left.x + text_sz.width,
                             top_left.y + text_sz.height + baseline);
            cv::rectangle(frame, bg_tl, bg_br, line_color, cv::FILLED);

            cv::putText(frame, label,
                        cv::Point(top_left.x, top_left.y + text_sz.height),
                        cv::FONT_HERSHEY_SIMPLEX, font_scale,
                        text_color, text_thickness);
        }
    }
    return frame;
}

// ---------------------------------------------------------------------------
// postprocess (entry point)
// ---------------------------------------------------------------------------

cv::Mat ImagePostprocessorNmsOnHost::postprocess(
    const cv::Mat& frame, const OutputMap& outputs)
{
    if (outputs.empty()) return frame;
    return draw_detections(frame.clone(), outputs);
}
