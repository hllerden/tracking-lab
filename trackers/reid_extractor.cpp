#include "reid_extractor.h"
#include <iostream>
#include <stdexcept>
#include <cstring>

cv::Mat ReidExtractor::l2Normalize(const cv::Mat& feat) {
    double norm = cv::norm(feat, cv::NORM_L2);
    if (norm < 1e-10) return feat.clone();
    return feat / norm;
}

// ============================================================
#ifdef USE_ORT_REID
// ============================================================

ReidExtractor::ReidExtractor(const std::string& modelPath, bool useGPU)
    : env_(ORT_LOGGING_LEVEL_WARNING, "ReidExtractor"),
      session_(nullptr) {

    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (useGPU) {
        OrtCUDAProviderOptions cudaOpts{};
        cudaOpts.device_id = 0;
        opts.AppendExecutionProvider_CUDA(cudaOpts);
        std::cout << "[ReID] Running on CUDA (ORT batch)" << std::endl;
    } else {
        std::cout << "[ReID] Running on CPU (ORT)" << std::endl;
    }

    session_ = Ort::Session(env_, modelPath.c_str(), opts);

    auto inName  = session_.GetInputNameAllocated(0, allocator_);
    auto outName = session_.GetOutputNameAllocated(0, allocator_);
    inputName_  = inName.get();
    outputName_ = outName.get();

    std::cout << "[ReID] Model loaded: " << modelPath << std::endl;
}

cv::Mat ReidExtractor::extract(const cv::Mat& frame, const cv::Rect& bbox) const {
    auto feats = extractBatch(frame, {bbox});
    return feats.empty() ? cv::Mat() : feats[0];
}

std::vector<cv::Mat> ReidExtractor::extractBatch(const cv::Mat& frame,
                                                   const std::vector<cv::Rect>& bboxes) const {
    if (bboxes.empty()) return {};

    const int N = static_cast<int>(bboxes.size());
    const int H = inputSize_.height;  // 256
    const int W = inputSize_.width;   // 128
    const int C = 3;

    static const float mean[3] = {0.485f, 0.456f, 0.406f};
    static const float stdv[3] = {0.229f, 0.224f, 0.225f};

    // Build flat NCHW float buffer [N, C, H, W]
    std::vector<float> inputBuf(static_cast<size_t>(N * C * H * W), 0.f);

    // Each crop is independent → parallel preprocessing.
    // Write targets are non-overlapping regions of inputBuf (stride = C*H*W per n).
    cv::parallel_for_(cv::Range(0, N), [&](const cv::Range& r) {
        for (int n = r.start; n < r.end; ++n) {
            cv::Rect safe = bboxes[n] & cv::Rect(0, 0, frame.cols, frame.rows);
            if (safe.area() <= 0) continue;

            cv::Mat crop;
            cv::resize(frame(safe), crop, cv::Size(W, H));
            if (crop.channels() == 1) {
                cv::cvtColor(crop, crop, cv::COLOR_GRAY2RGB);
            } else if (crop.channels() == 4) {
                cv::cvtColor(crop, crop, cv::COLOR_BGRA2RGB);
            } else {
                cv::cvtColor(crop, crop, cv::COLOR_BGR2RGB);
            }
            crop.convertTo(crop, CV_32F, 1.0 / 255.0);

            std::vector<cv::Mat> channels(C);
            cv::split(crop, channels);
            for (int c = 0; c < C; ++c) {
                cv::Mat norm = (channels[c] - mean[c]) / stdv[c];
                float* dst = inputBuf.data() + (n * C + c) * H * W;
                std::memcpy(dst, norm.ptr<float>(0), static_cast<size_t>(H * W) * sizeof(float));
            }
        }
    });

    // Wrap in ORT tensor (CPU memory, ORT copies to GPU internally)
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<int64_t, 4> shape{N, C, H, W};
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, inputBuf.data(), inputBuf.size(), shape.data(), shape.size());

    const char* inNames[]  = {inputName_.c_str()};
    const char* outNames[] = {outputName_.c_str()};

    auto outputs = session_.Run(
        Ort::RunOptions{nullptr},
        inNames, &inputTensor, 1,
        outNames, 1);

    const float* outData  = outputs[0].GetTensorData<float>();
    const auto   outShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const int    D        = static_cast<int>(outShape.back());

    std::vector<cv::Mat> features;
    features.reserve(N);
    for (int n = 0; n < N; ++n) {
        cv::Mat feat(1, D, CV_32F);
        std::memcpy(feat.ptr<float>(), outData + n * D, static_cast<size_t>(D) * sizeof(float));
        features.push_back(l2Normalize(feat));
    }
    return features;
}

// ============================================================
#else  // OpenCV DNN backend (default)
// ============================================================

cv::Mat ReidExtractor::preprocess(const cv::Mat& crop) const {
    cv::Mat resized;
    cv::resize(crop, resized, inputSize_);

    cv::Mat rgbFloat;
    if (resized.channels() == 1) {
        cv::cvtColor(resized, rgbFloat, cv::COLOR_GRAY2RGB);
    } else if (resized.channels() == 4) {
        cv::cvtColor(resized, rgbFloat, cv::COLOR_BGRA2RGB);
    } else {
        cv::cvtColor(resized, rgbFloat, cv::COLOR_BGR2RGB);
    }
    rgbFloat.convertTo(rgbFloat, CV_32F, 1.0 / 255.0);

    static const float mean[3] = {0.485f, 0.456f, 0.406f};
    static const float stdv[3] = {0.229f, 0.224f, 0.225f};
    std::vector<cv::Mat> channels(3);
    cv::split(rgbFloat, channels);
    for (int c = 0; c < 3; ++c)
        channels[c] = (channels[c] - mean[c]) / stdv[c];
    cv::merge(channels, rgbFloat);

    cv::Mat blob;
    cv::dnn::blobFromImage(rgbFloat, blob);
    return blob;
}

ReidExtractor::ReidExtractor(const std::string& modelPath, bool useGPU) {
#if CV_VERSION_MAJOR >= 5
    // ENGINE_CLASSIC required for CUDA backend in OpenCV 5.0.
    // Model must be pre-simplified with onnxsim (yolo26n-reid-sim.onnx).
    const auto engine = useGPU ? cv::dnn::ENGINE_CLASSIC : cv::dnn::ENGINE_AUTO;
    net_ = cv::dnn::readNetFromONNX(modelPath, engine);
#else
    net_ = cv::dnn::readNetFromONNX(modelPath);
#endif
    if (net_.empty())
        throw std::runtime_error("[ReID] Failed to load model: " + modelPath);

    if (useGPU) {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        std::cout << "[ReID] Running on CUDA (OpenCV DNN)" << std::endl;
    } else {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        std::cout << "[ReID] Running on CPU (OpenCV DNN)" << std::endl;
    }
    std::cout << "[ReID] Model loaded: " << modelPath << std::endl;
}

cv::Mat ReidExtractor::extract(const cv::Mat& frame, const cv::Rect& bbox) const {
    cv::Rect safe = bbox & cv::Rect(0, 0, frame.cols, frame.rows);
    if (safe.area() <= 0) return cv::Mat();
    cv::Mat blob = preprocess(frame(safe));
    net_.setInput(blob);
    cv::Mat output = net_.forward();
    return l2Normalize(output.reshape(1, 1));
}

std::vector<cv::Mat> ReidExtractor::extractBatch(const cv::Mat& frame,
                                                   const std::vector<cv::Rect>& bboxes) const {
    std::vector<cv::Mat> features;
    features.reserve(bboxes.size());
    for (const auto& bbox : bboxes)
        features.push_back(extract(frame, bbox));
    return features;
}

#endif // USE_ORT_REID
