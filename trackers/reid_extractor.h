#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#ifdef USE_ORT_REID
#  include <onnxruntime_cxx_api.h>
#else
#  include <opencv2/dnn.hpp>
#endif

/**
 * @brief Crop-based ReID feature extractor using an ONNX model.
 *
 * Two interchangeable backends, selected at compile time:
 *
 *   USE_ORT_REID defined (cmake -DREID_BACKEND_ORT=ON):
 *     ONNX Runtime CUDA EP — true batch inference, ~4ms for 25 detections.
 *     Use the original model (yolo26n-reid.onnx); ORT handles dynamic shapes.
 *
 *   USE_ORT_REID NOT defined (default):
 *     OpenCV DNN ENGINE_CLASSIC + CUDA — sequential batch=1 calls.
 *     Requires onnxsim-simplified model (yolo26n-reid-sim.onnx).
 *
 * Public API is identical for both backends.
 */
class ReidExtractor {
public:
    explicit ReidExtractor(const std::string& modelPath, bool useGPU = true);

    cv::Mat extract(const cv::Mat& frame, const cv::Rect& bbox) const;
    std::vector<cv::Mat> extractBatch(const cv::Mat& frame,
                                      const std::vector<cv::Rect>& bboxes) const;

    void setInputSize(const cv::Size& size) { inputSize_ = size; }
    cv::Size inputSize() const              { return inputSize_; }

private:
    cv::Size inputSize_{128, 256};
    static cv::Mat l2Normalize(const cv::Mat& feat);

#ifdef USE_ORT_REID
    mutable Ort::Env     env_;
    mutable Ort::Session session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::string inputName_;
    std::string outputName_;
#else
    mutable cv::dnn::Net net_;
    cv::Mat preprocess(const cv::Mat& crop) const;
#endif
};
