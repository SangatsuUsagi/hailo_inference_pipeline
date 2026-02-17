#include "inference_pipeline.hpp"

#include <iostream>
#include <algorithm>

// ---------------------------------------------------------------------------
// Helper utilities
// ---------------------------------------------------------------------------

bool is_hailo_timeout(hailo_status status) {
    return status == HAILO_TIMEOUT;
}

bool is_hailo_error(hailo_status status) {
    return status != HAILO_SUCCESS;
}

std::tuple<cv::Mat, std::pair<float,float>, std::pair<int,int>>
preprocess_image_with_pad(const cv::Mat& image, int target_height, int target_width) {
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

    int height = rgb.rows;
    int width  = rgb.cols;

    int h1, w1, padh, padw;
    float scale;

    if (height >= width) {
        h1    = target_height;
        w1    = static_cast<int>(static_cast<float>(target_height) / height * width);
        padh  = 0;
        padw  = (target_width - w1) / 2;
        scale = static_cast<float>(height) / h1;
    } else {
        w1    = target_width;
        h1    = static_cast<int>(static_cast<float>(target_width) / width * height);
        padh  = (target_height - h1) / 2;
        padw  = 0;
        scale = static_cast<float>(width) / w1;
    }

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(w1, h1), 0, 0, cv::INTER_AREA);

    int padh2 = target_height - h1 - padh;
    int padw2 = target_width  - w1 - padw;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, padh, padh2, padw, padw2,
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    return {padded,
            {scale, scale},
            {static_cast<int>(padh * scale), static_cast<int>(padw * scale)}};
}

// ---------------------------------------------------------------------------
// InferPipeline - construction / teardown
// ---------------------------------------------------------------------------

InferPipeline::InferPipeline(
    const std::string& net_path,
    int batch_size,
    bool is_async,
    bool is_callback,
    bool is_nms,
    const std::vector<std::string>& layer_name_u8,
    const std::vector<std::string>& layer_name_u16)
    : is_async_(is_async)
    , is_callback_(is_callback)
    , is_nms_(is_nms)
    , layer_name_u8_(layer_name_u8)
    , layer_name_u16_(layer_name_u16)
{
    hailo_vdevice_params_t params = {};
    params.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;

    auto vdev_exp = hailort::VDevice::create(params);
    if (!vdev_exp)
        throw std::runtime_error("Failed to create VDevice: " +
                                 std::to_string(vdev_exp.status()));
    vdevice_ = vdev_exp.release();

    if (is_async_) {
        // ---- Async (InferModel) path ----
        auto model_exp = vdevice_->create_infer_model(net_path);
        if (!model_exp)
            throw std::runtime_error("Failed to create infer model: " +
                                     std::to_string(model_exp.status()));
        infer_model_ = model_exp.release();
        infer_model_->set_batch_size(batch_size);

        // Set output format types
        for (const auto& name : infer_model_->get_output_names()) {
            hailo_format_type_t fmt = HAILO_FORMAT_TYPE_FLOAT32;
            if (std::find(layer_name_u8_.begin(),  layer_name_u8_.end(),  name) != layer_name_u8_.end())
                fmt = HAILO_FORMAT_TYPE_UINT8;
            else if (std::find(layer_name_u16_.begin(), layer_name_u16_.end(), name) != layer_name_u16_.end())
                fmt = HAILO_FORMAT_TYPE_UINT16;
            infer_model_->output(name).set_format_type(fmt);
        }

        auto cfg_exp = infer_model_->configure();
        if (!cfg_exp)
            throw std::runtime_error("Failed to configure infer model: " +
                                     std::to_string(cfg_exp.status()));
        configured_infer_model_ = cfg_exp.release();

    } else {
        // ---- Sync (VStreams) path ----
        auto hef_exp = hailort::Hef::create(net_path);
        if (!hef_exp)
            throw std::runtime_error("Failed to load HEF: " +
                                     std::to_string(hef_exp.status()));
        hef_ = std::make_unique<hailort::Hef>(hef_exp.release());

        auto cfg_params_exp = hef_->create_configure_params(HAILO_STREAM_INTERFACE_PCIE);
        if (!cfg_params_exp)
            throw std::runtime_error("Failed to create configure params: " +
                                     std::to_string(cfg_params_exp.status()));

        auto net_groups_exp = vdevice_->configure(*hef_, cfg_params_exp.value());
        if (!net_groups_exp)
            throw std::runtime_error("Failed to configure network group: " +
                                     std::to_string(net_groups_exp.status()));
        network_group_ = net_groups_exp.value()[0];

        // NOTE: make_input/output_vstream_params signatures vary slightly by
        // HailoRT version; adjust if your SDK uses a different overload.
        auto in_p = network_group_->make_input_vstream_params(
            {}, HAILO_FORMAT_TYPE_UINT8,
            HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
        if (!in_p)
            throw std::runtime_error("Failed to create input vstream params: " +
                                     std::to_string(in_p.status()));
        input_vstream_params_ = in_p.release();

        auto out_p = network_group_->make_output_vstream_params(
            {}, HAILO_FORMAT_TYPE_FLOAT32,
            HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
        if (!out_p)
            throw std::runtime_error("Failed to create output vstream params: " +
                                     std::to_string(out_p.status()));
        output_vstream_params_ = out_p.release();
    }
}

InferPipeline::~InferPipeline() { close(); }

void InferPipeline::close() {
    // Releasing vdevice releases all child resources.
    vdevice_.reset();
}

// ---------------------------------------------------------------------------
// Public inference entry point
// ---------------------------------------------------------------------------

OutputMap InferPipeline::inference(const std::vector<cv::Mat>& dataset) {
    if (is_async_) { infer_async(dataset); return {}; }
    return infer_pipeline(dataset);
}

// ---------------------------------------------------------------------------
// Async mode
// ---------------------------------------------------------------------------

void InferPipeline::infer_async(const std::vector<cv::Mat>& inputs) {
    if (inputs.empty())
        throw std::invalid_argument("infer_inputs cannot be empty");

    const auto& input_names = infer_model_->get_input_names();
    if (inputs.size() != input_names.size())
        throw std::invalid_argument("Input count mismatch");

    // Create new bindings for this invocation
    auto bind_exp = configured_infer_model_.create_bindings();
    if (!bind_exp)
        throw InferenceSubmitError("Failed to create bindings: " +
                                   std::to_string(bind_exp.status()));
    bindings_ = bind_exp.release();

    // ---- Set input buffers ----
    for (size_t i = 0; i < input_names.size(); ++i) {
        cv::Mat cont = inputs[i].isContinuous() ? inputs[i] : inputs[i].clone();
        auto status = bindings_.input(input_names[i]).set_buffer(
            hailort::MemoryView(cont.data, cont.total() * cont.elemSize()));
        if (is_hailo_error(status))
            throw InferenceSubmitError("Failed to set input buffer '" + input_names[i] + "'");
    }

    // ---- Allocate and set output buffers ----
    // After job.wait(), the inference results will be written directly into
    // these member vectors, which we then return in collect_output_from_bindings().
    output_buffers_f32_.clear();
    output_buffers_u8_.clear();
    output_buffers_u16_.clear();

    for (const auto& out_name : infer_model_->get_output_names()) {
        // get_frame_size() returns the total byte size for this output stream.
        // NOTE: API: ConfiguredInferModel::output(name) returns InferStream& in most versions.
        size_t frame_size = configured_infer_model_.output(out_name).get_frame_size();

        bool is_u8  = std::find(layer_name_u8_.begin(),  layer_name_u8_.end(),  out_name) != layer_name_u8_.end();
        bool is_u16 = std::find(layer_name_u16_.begin(), layer_name_u16_.end(), out_name) != layer_name_u16_.end();

        if (is_nms_ || is_u8) {
            // Raw byte buffer for NMS or explicit uint8 layers
            output_buffers_u8_[out_name].assign(frame_size, 0);
            bindings_.output(out_name).set_buffer(
                hailort::MemoryView(output_buffers_u8_[out_name].data(), frame_size));
        } else if (is_u16) {
            output_buffers_u16_[out_name].assign(frame_size / sizeof(uint16_t), 0);
            bindings_.output(out_name).set_buffer(
                hailort::MemoryView(output_buffers_u16_[out_name].data(), frame_size));
        } else {
            output_buffers_f32_[out_name].assign(frame_size / sizeof(float), 0.0f);
            bindings_.output(out_name).set_buffer(
                hailort::MemoryView(output_buffers_f32_[out_name].data(), frame_size));
        }
    }

    // ---- Wait for device ready ----
    auto ready_status = configured_infer_model_.wait_for_async_ready(
        std::chrono::milliseconds(TIMEOUT_MS));
    if (is_hailo_error(ready_status)) {
        if (is_hailo_timeout(ready_status))
            throw InferenceTimeoutError("Inference device not ready: timeout after " +
                                        std::to_string(TIMEOUT_MS) + "ms");
        throw InferenceSubmitError("Hailo device error while waiting for async ready");
    }

    // ---- Submit job ----
    callback_error_ = false;
    callback_results_.clear();

    hailort::Expected<hailort::AsyncInferJob> job_exp;
    if (is_callback_) {
        auto cb = [this](const hailort::AsyncInferCompletionInfo& info) {
            if (info.status != HAILO_SUCCESS) {
                callback_error_ = true;
                std::cerr << "Inference callback error: " << info.status << "\n";
            } else {
                // Buffers are already filled; collect into callback_results_
                callback_results_ = collect_output_from_bindings();
            }
        };
        job_exp = configured_infer_model_.run_async({bindings_}, cb);
    } else {
        job_exp = configured_infer_model_.run_async({bindings_});
    }

    if (!job_exp)
        throw InferenceSubmitError("Failed to submit async job: " +
                                   std::to_string(job_exp.status()));
    async_job_ = job_exp.release();
    has_job_   = true;
}

OutputMap InferPipeline::wait_and_get_output() {
    if (!has_job_)
        throw std::runtime_error("No inference job pending. Call inference() first.");

    auto status = async_job_.wait(std::chrono::milliseconds(TIMEOUT_MS));
    has_job_ = false;

    if (is_hailo_error(status)) {
        if (is_hailo_timeout(status))
            throw InferenceTimeoutError("Inference job timed out after " +
                                        std::to_string(TIMEOUT_MS) + "ms");
        throw InferenceWaitError("Hailo device error waiting for inference results");
    }

    if (is_callback_) {
        if (callback_error_)
            throw InferenceWaitError("Inference callback reported an error");
        return callback_results_;
    }

    // Non-callback: results are now in our output_buffers_* vectors
    return collect_output_from_bindings();
}

// Assemble OutputMap from the member buffer vectors (called after job completion).
OutputMap InferPipeline::collect_output_from_bindings() {
    OutputMap results;
    for (const auto& out_name : infer_model_->get_output_names()) {
        InferenceOutput out;
        out.is_nms = is_nms_;

        if (is_nms_) {
            // NMS output was written into output_buffers_u8_[out_name] as raw bytes
            if (output_buffers_u8_.count(out_name))
                parse_nms_buffer_raw(output_buffers_u8_.at(out_name), out_name, out);
        } else if (output_buffers_f32_.count(out_name)) {
            out.regular.data = output_buffers_f32_.at(out_name);
            // Shape (H x W x C) is not strictly needed by the postprocessors for
            // classification; add shape retrieval here if needed for your use case.
        } else if (output_buffers_u16_.count(out_name)) {
            const auto& u16 = output_buffers_u16_.at(out_name);
            out.regular.data.resize(u16.size());
            for (size_t i = 0; i < u16.size(); ++i)
                out.regular.data[i] = static_cast<float>(u16[i]);
        } else if (output_buffers_u8_.count(out_name)) {
            // Explicit uint8 layer (not NMS)
            const auto& u8 = output_buffers_u8_.at(out_name);
            out.regular.data.resize(u8.size());
            for (size_t i = 0; i < u8.size(); ++i)
                out.regular.data[i] = static_cast<float>(u8[i]);
        }

        results[out_name] = std::move(out);
    }
    return results;
}

// ---------------------------------------------------------------------------
// Synchronous mode (VStreams)
// ---------------------------------------------------------------------------

OutputMap InferPipeline::infer_pipeline(const std::vector<cv::Mat>& inputs) {
    if (inputs.empty())
        throw std::invalid_argument("infer_inputs cannot be empty");

    auto in_streams_exp = hailort::VStreamsBuilder::create_input_vstreams(
        *network_group_, input_vstream_params_);
    if (!in_streams_exp)
        throw InferencePipelineError("Failed to create input vstreams: " +
                                     std::to_string(in_streams_exp.status()));

    auto out_streams_exp = hailort::VStreamsBuilder::create_output_vstreams(
        *network_group_, output_vstream_params_);
    if (!out_streams_exp)
        throw InferencePipelineError("Failed to create output vstreams: " +
                                     std::to_string(out_streams_exp.status()));

    auto& in_streams  = in_streams_exp.value();
    auto& out_streams = out_streams_exp.value();

    if (inputs.size() != in_streams.size())
        throw std::invalid_argument("Input count mismatch");

    // Activate the network group for this inference run
    auto active_exp = network_group_->activate();
    if (!active_exp)
        throw InferencePipelineError("Failed to activate network group: " +
                                     std::to_string(active_exp.status()));

    // Write inputs (add batch dimension: vstream expects [H, W, C])
    for (size_t i = 0; i < in_streams.size(); ++i) {
        cv::Mat cont = inputs[i].isContinuous() ? inputs[i] : inputs[i].clone();
        auto wst = in_streams[i].write(
            hailort::MemoryView(cont.data, cont.total() * cont.elemSize()));
        if (is_hailo_error(wst))
            throw InferencePipelineError("Failed to write input vstream " +
                                         std::to_string(i) + ": " + std::to_string(wst));
    }

    // Read outputs
    OutputMap results;
    for (auto& out_stream : out_streams) {
        const std::string& name = out_stream.name();
        size_t frame_bytes = out_stream.get_frame_size();

        InferenceOutput out;
        out.is_nms = is_nms_;

        if (is_nms_) {
            std::vector<uint8_t> raw(frame_bytes, 0);
            auto rst = out_stream.read(hailort::MemoryView(raw.data(), frame_bytes));
            if (is_hailo_error(rst))
                throw InferencePipelineError("Failed to read NMS output: " +
                                             std::to_string(rst));
            parse_nms_buffer_raw(raw, name, out);
        } else {
            size_t float_count = frame_bytes / sizeof(float);
            out.regular.data.resize(float_count, 0.0f);
            auto rst = out_stream.read(
                hailort::MemoryView(out.regular.data.data(), frame_bytes));
            if (is_hailo_error(rst))
                throw InferencePipelineError("Failed to read output vstream '" +
                                             name + "': " + std::to_string(rst));
        }
        results[name] = std::move(out);
    }

    return results;
}

// ---------------------------------------------------------------------------
// NMS buffer parsing
// ---------------------------------------------------------------------------

// HailoRT NMS_BY_CLASS output format:
//   For each class (num_classes total):
//     [uint16_t count][count × {float y_min, x_min, y_max, x_max, score}]
void InferPipeline::parse_nms_buffer(
    const hailort::MemoryView& buf,
    const std::string& /*name*/,
    InferenceOutput& out)
{
    const uint8_t* ptr = static_cast<const uint8_t*>(buf.data());
    size_t offset = 0;

    while (offset + sizeof(uint16_t) <= buf.size()) {
        uint16_t count = *reinterpret_cast<const uint16_t*>(ptr + offset);
        offset += sizeof(uint16_t);

        ClassDetections class_dets;
        constexpr size_t kDetSize = 5 * sizeof(float);
        for (uint16_t i = 0; i < count && offset + kDetSize <= buf.size(); ++i) {
            const float* b = reinterpret_cast<const float*>(ptr + offset);
            class_dets.push_back({b[0], b[1], b[2], b[3], b[4]});
            offset += kDetSize;
        }
        out.nms.push_back(std::move(class_dets));
    }
}

void InferPipeline::parse_nms_buffer_raw(
    const std::vector<uint8_t>& raw,
    const std::string& name,
    InferenceOutput& out)
{
    hailort::MemoryView view(const_cast<uint8_t*>(raw.data()), raw.size());
    parse_nms_buffer(view, name, out);
}
