#ifndef BENCHMARK_CONFIG_H
#define BENCHMARK_CONFIG_H

#include <string>
#include <vector>
#include <chrono>
#include "trackers/kalman_iou/kalman_iou_tracker.h"

/**
 * @file benchmark_config.h
 * @brief Configuration structures for MOT17 benchmark system
 * @author Halil Erden
 * @date 2025-01-20
 */

// ==================== BENCHMARK MODE ====================

/**
 * @brief Benchmark execution mode
 */
enum class BenchmarkMode {
    TRACKER_ONLY,   ///< Use pre-computed detections (det.txt) - Fast (~10x)
    END_TO_END,     ///< Use YOLO inference + tracking - Complete system
    BOTH            ///< Run both modes sequentially
};

// ==================== BENCHMARK CONFIG ====================

/**
 * @brief Main configuration for benchmark execution
 */
struct BenchmarkConfig {
    // Paths
    std::string projectBasePath;            ///< Base project directory
    std::string outputVersion = PROJECT_VERSION;  ///< Output version (from CMake)

    // Execution mode
    BenchmarkMode mode = BenchmarkMode::BOTH;

    // Test matrix
    std::vector<std::string> sequences = {
        "MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09",
        "MOT17-10", "MOT17-11", "MOT17-13"
    };

    std::vector<std::string> detectors = {
        "DPM",      ///< Deformable Part Model
        "FRCNN",    ///< Faster R-CNN
        "SDP"       ///< SDP detector
    };

    std::vector<KalmanIoUConfig::IoUType> iouTypes = {
        KalmanIoUConfig::IoUType::IOU,
        KalmanIoUConfig::IoUType::GIOU,
        KalmanIoUConfig::IoUType::DIOU,
        KalmanIoUConfig::IoUType::CIOU,
        KalmanIoUConfig::IoUType::SIOU,
        KalmanIoUConfig::IoUType::AIOU
    };

    // YOLO settings (for end-to-end mode)
    std::string yoloModelPath = "/models/yolov9s.onnx";
    bool runOnGPU = true;

    // Tracker settings
    float iouThreshold = 0.2f;
    int maxLostFrames = 50;
    bool usePredictionInLost = true;
    bool removeOutOfBounds = true;

    // Execution control
    bool verbose = true;                    ///< Print progress messages
    bool saveIntermediateResults = true;    ///< Save per-test results
};

// ==================== TEST RESULT ====================

/**
 * @brief Result of a single benchmark test
 */
struct TestResult {
    // Test identification
    std::string sequence;
    std::string detector;
    std::string iouType;
    std::string outputFile;

    // Performance metrics
    int totalFrames = 0;
    double processingTimeMs = 0.0;
    double fps = 0.0;

    // Tracking metrics
    int totalDetections = 0;
    int totalTracks = 0;

    // Status
    bool success = false;
    std::string errorMessage;

    // Timestamp
    std::chrono::system_clock::time_point timestamp;
};

// ==================== HELPER FUNCTIONS ====================

/**
 * @brief Convert IoUType enum to string
 */
inline std::string iouTypeToString(KalmanIoUConfig::IoUType type) {
    switch (type) {
        case KalmanIoUConfig::IoUType::IOU:   return "IOU";
        case KalmanIoUConfig::IoUType::GIOU:  return "GIOU";
        case KalmanIoUConfig::IoUType::DIOU:  return "DIOU";
        case KalmanIoUConfig::IoUType::CIOU:  return "CIOU";
        case KalmanIoUConfig::IoUType::SIOU:  return "SIOU";
        case KalmanIoUConfig::IoUType::AIOU:  return "AIOU";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Convert string to IoUType enum
 */
inline KalmanIoUConfig::IoUType stringToIoUType(const std::string& str) {
    if (str == "IOU")   return KalmanIoUConfig::IoUType::IOU;
    if (str == "GIOU")  return KalmanIoUConfig::IoUType::GIOU;
    if (str == "DIOU")  return KalmanIoUConfig::IoUType::DIOU;
    if (str == "CIOU")  return KalmanIoUConfig::IoUType::CIOU;
    if (str == "SIOU")  return KalmanIoUConfig::IoUType::SIOU;
    if (str == "AIOU")  return KalmanIoUConfig::IoUType::AIOU;
    return KalmanIoUConfig::IoUType::CIOU; // default
}

/**
 * @brief Convert BenchmarkMode enum to string
 */
inline std::string benchmarkModeToString(BenchmarkMode mode) {
    switch (mode) {
        case BenchmarkMode::TRACKER_ONLY: return "tracker-only";
        case BenchmarkMode::END_TO_END:   return "end-to-end";
        case BenchmarkMode::BOTH:         return "both";
        default: return "unknown";
    }
}

#endif // BENCHMARK_CONFIG_H
