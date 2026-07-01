# MOT17 Benchmark System - Kod Şablonları

**Oluşturulma Tarihi:** 2025-01-20
**Amaç:** Implementation sırasında copy-paste yapılabilir hazır kodlar

---

## 📋 İçindekiler

1. [CMakeLists.txt Eklentileri](#1-cmakeliststxt-eklentileri)
2. [benchmark_config.h (Tam Dosya)](#2-benchmark_configh-tam-dosya)
3. [mot17_benchmark.cpp (İskelet)](#3-mot17_benchmarkcpp-iskelet)
4. [BenchmarkRunner Implementation](#4-benchmarkrunner-implementation)
5. [Helper Functions](#5-helper-functions)
6. [run_trackeval.sh Güncellemeleri](#6-run_trackevalsh-güncellemeleri)
7. [compare_versions.sh (Tam Dosya)](#7-compare_versionssh-tam-dosya)

---

## 1. CMakeLists.txt Eklentileri

### 1.1 Version Tanımlamaları (Satır 1'in altına)

```cmake
cmake_minimum_required(VERSION 3.16)

# ==================== PROJECT VERSION ====================
set(PROJECT_VERSION_MAJOR 1)
set(PROJECT_VERSION_MINOR 0)
set(PROJECT_VERSION_PATCH 0)
set(PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")

project(yoloExercise VERSION ${PROJECT_VERSION} LANGUAGES CXX)

# Make version available at compile time
add_compile_definitions(PROJECT_VERSION="${PROJECT_VERSION}")

message(STATUS "Project version: ${PROJECT_VERSION}")
# ==========================================================
```

---

### 1.2 mot17_benchmark Executable (Satır 80'in altına, test_mot_reader'dan sonra)

```cmake
# ==================== MOT17 BENCHMARK EXECUTABLE ====================
# Headless benchmark system for systematic tracking algorithm testing
add_executable(mot17_benchmark
  mot17_benchmark.cpp
  benchmark_config.h

  # MOT17 readers
  mot_sequence_reader.h mot_sequence_reader.cpp
  mot_detection_reader.h mot_detection_reader.cpp

  # Tracking system
  kalman_iou_tracker.h kalman_iou_tracker.cpp
  inference.h inference.cpp
  impression.h impression.cpp

  # Telemetry
  TrackerTelemtryLogger/MotChallengeLogger.h
  TrackerTelemtryLogger/MotChallengeLogger.cpp

  # Utilities
  HungarianAlgorithm.h HungarianAlgorithm.cpp

  # Interfaces
  i_tracker_telemetry.h
  i_object_tracker.h
  i_visual_tracker.h
  i_configurable_tracker.h
)

target_link_libraries(mot17_benchmark
  ${OpenCV_LIBS}
  Eigen3::Eigen
)

message(STATUS "mot17_benchmark executable configured")
# ====================================================================
```

---

## 2. benchmark_config.h (Tam Dosya)

```cpp
#ifndef BENCHMARK_CONFIG_H
#define BENCHMARK_CONFIG_H

#include <string>
#include <vector>
#include <chrono>
#include "kalman_iou_tracker.h"

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
    int maxLostFrames = 20;
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
```

---

## 3. mot17_benchmark.cpp (İskelet)

```cpp
/**
 * @file mot17_benchmark.cpp
 * @brief MOT17 Benchmark System - Main executable
 * @author Halil Erden
 * @date 2025-01-20
 *
 * Headless benchmark system for systematic tracking algorithm testing
 * Supports two modes:
 *   - Tracker-Only: Uses pre-computed detections (det.txt) - Fast
 *   - End-to-End: Uses YOLO inference + tracking - Complete system
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <vector>
#include <string>

// Project headers
#include "benchmark_config.h"
#include "mot_sequence_reader.h"
#include "mot_detection_reader.h"
#include "kalman_iou_tracker.h"
#include "inference.h"
#include "impression.h"
#include "TrackerTelemtryLogger/MotChallengeLogger.h"

// Namespaces
namespace fs = std::filesystem;
using namespace std;
using namespace cv;

// ==================== UTILITIES ====================

/**
 * @brief Get current timestamp in milliseconds
 */
inline int64_t timeSinceEpochMillisec() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

/**
 * @brief Format timestamp to ISO 8601 string
 */
inline std::string timestampToISO8601(const std::chrono::system_clock::time_point& tp) {
    auto time = std::chrono::system_clock::to_time_t(tp);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%dT%H:%M:%S");
    return ss.str();
}

// ==================== BENCHMARK RUNNER ====================

/**
 * @brief Main benchmark runner class
 */
class BenchmarkRunner {
public:
    explicit BenchmarkRunner(const BenchmarkConfig& config)
        : config_(config) {}

    /**
     * @brief Run all benchmark tests based on configuration
     */
    void runAllTests();

private:
    // Mode execution
    void runTrackerOnlyMode();
    void runEndToEndMode();

    // Single test execution
    TestResult runSingleTest_TrackerOnly(
        const std::string& sequence,
        const std::string& detector,
        KalmanIoUConfig::IoUType iouType
    );

    TestResult runSingleTest_EndToEnd(
        const std::string& sequence,
        KalmanIoUConfig::IoUType iouType
    );

    // Utilities
    void createOutputDirectories();
    void printHeader();
    void printProgress(int current, int total, const TestResult& result);
    void printSummary(const std::string& mode, const std::vector<TestResult>& results);
    void saveSummaryJSON(const std::string& mode, const std::vector<TestResult>& results);
    void saveCombinedReport();

    // Member variables
    BenchmarkConfig config_;
    std::vector<TestResult> trackerOnlyResults_;
    std::vector<TestResult> endToEndResults_;
};

// ==================== MAIN FUNCTION ====================

int main(int argc, char** argv) {
    // Create configuration
    BenchmarkConfig config;
    config.projectBasePath = PROJECT_ROOT_DIR;

    // TODO: Parse command line arguments (optional)
    // --mode tracker-only|end-to-end|both
    // --sequences MOT17-02,MOT17-04
    // --detectors DPM,FRCNN
    // --iou-types CIOU,DIOU

    // Run benchmark
    try {
        BenchmarkRunner runner(config);
        runner.runAllTests();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }
}

// ==================== IMPLEMENTATION ====================
// (Implementations will be added in next steps)
```

---

## 4. BenchmarkRunner Implementation

### 4.1 runAllTests()

```cpp
void BenchmarkRunner::runAllTests() {
    printHeader();
    createOutputDirectories();

    if (config_.mode == BenchmarkMode::TRACKER_ONLY ||
        config_.mode == BenchmarkMode::BOTH) {
        runTrackerOnlyMode();
    }

    if (config_.mode == BenchmarkMode::END_TO_END ||
        config_.mode == BenchmarkMode::BOTH) {
        runEndToEndMode();
    }

    if (config_.mode == BenchmarkMode::BOTH) {
        saveCombinedReport();
    }

    std::cout << "\n=== Benchmark Complete ===\n";
    std::cout << "Results saved to: output/v" << config_.outputVersion << "/\n";
}
```

---

### 4.2 runTrackerOnlyMode()

```cpp
void BenchmarkRunner::runTrackerOnlyMode() {
    std::cout << "\n=== Running Tracker-Only Mode ===\n";
    std::cout << "Using pre-computed detections (det.txt)\n";

    int totalTests = config_.sequences.size() *
                     config_.detectors.size() *
                     config_.iouTypes.size();

    std::cout << "Total tests: " << totalTests << "\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";

    int testIndex = 0;
    for (const auto& sequence : config_.sequences) {
        for (const auto& detector : config_.detectors) {
            for (const auto& iouType : config_.iouTypes) {
                testIndex++;

                auto result = runSingleTest_TrackerOnly(sequence, detector, iouType);
                trackerOnlyResults_.push_back(result);

                printProgress(testIndex, totalTests, result);
            }
        }
    }

    printSummary("tracker-only", trackerOnlyResults_);
    saveSummaryJSON("tracker-only", trackerOnlyResults_);
}
```

---

### 4.3 runSingleTest_TrackerOnly()

```cpp
TestResult BenchmarkRunner::runSingleTest_TrackerOnly(
    const std::string& sequence,
    const std::string& detector,
    KalmanIoUConfig::IoUType iouType
) {
    TestResult result;
    result.sequence = sequence;
    result.detector = detector;
    result.iouType = iouTypeToString(iouType);
    result.timestamp = std::chrono::system_clock::now();

    try {
        // Paths
        std::string sequencePath = config_.projectBasePath + "/MOT17-Data/MOT17/train/" +
                                   sequence + "-" + detector;
        std::string outputPath = config_.projectBasePath + "/output/v" +
                                config_.outputVersion + "/tracker-only/" +
                                sequence + "-" + detector + "-" + result.iouType + ".txt";

        result.outputFile = outputPath;

        // Initialize components
        MotSequenceReader reader(sequencePath);
        MotDetectionReader detReader(sequencePath + "/det/det.txt");

        KalmanIoUTracker tracker;
        KalmanIoUConfig trackerConfig;
        trackerConfig.iouType = iouType;
        trackerConfig.iouThreshold = config_.iouThreshold;
        trackerConfig.maxLostFrames = config_.maxLostFrames;
        trackerConfig.usePredictionInLost = config_.usePredictionInLost;
        trackerConfig.removeOutOfBounds = config_.removeOutOfBounds;
        tracker.setConfig(trackerConfig);

        auto logger = std::make_shared<MotChallengeLogger>(outputPath, true);
        tracker.setTelemetry(logger);
        tracker.setFrameSizeHint(reader.getFrameSize(), reader.getFrameSize());

        // Process frames
        cv::Mat frame;
        int frameCount = 0;
        int totalDetections = 0;
        auto startTime = timeSinceEpochMillisec();

        while (reader.nextFrame(frame)) {
            int frameIndex = frameCount + 1; // MOT uses 1-based indexing

            // Set frame metadata
            ITrackerTelemetry::FrameMetadata metadata;
            metadata.frameIndex = frameCount;
            metadata.sourceSize = reader.getFrameSize();
            metadata.processedSize = frame.size();
            metadata.timestamp = frameCount / reader.getFPS();
            logger->onFrameStart(metadata);

            // Get detections and track
            auto detections = detReader.getDetectionsForFrame(frameIndex);
            totalDetections += detections.size();
            tracker.update(detections);

            frameCount++;
        }

        logger->flush();

        auto endTime = timeSinceEpochMillisec();
        result.totalFrames = frameCount;
        result.processingTimeMs = endTime - startTime;
        result.fps = (frameCount * 1000.0) / result.processingTimeMs;
        result.totalDetections = totalDetections;
        result.totalTracks = tracker.getTrackedObjects().size();
        result.success = true;

    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = e.what();
    }

    return result;
}
```

---

### 4.4 runEndToEndMode()

```cpp
void BenchmarkRunner::runEndToEndMode() {
    std::cout << "\n=== Running End-to-End Mode ===\n";
    std::cout << "Using YOLO inference + tracking\n";

    int totalTests = config_.sequences.size() * config_.iouTypes.size();

    std::cout << "Total tests: " << totalTests << "\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";

    int testIndex = 0;
    for (const auto& sequence : config_.sequences) {
        for (const auto& iouType : config_.iouTypes) {
            testIndex++;

            auto result = runSingleTest_EndToEnd(sequence, iouType);
            endToEndResults_.push_back(result);

            printProgress(testIndex, totalTests, result);
        }
    }

    printSummary("end-to-end", endToEndResults_);
    saveSummaryJSON("end-to-end", endToEndResults_);
}
```

---

### 4.5 runSingleTest_EndToEnd()

```cpp
TestResult BenchmarkRunner::runSingleTest_EndToEnd(
    const std::string& sequence,
    KalmanIoUConfig::IoUType iouType
) {
    TestResult result;
    result.sequence = sequence;
    result.detector = "YOLO";
    result.iouType = iouTypeToString(iouType);
    result.timestamp = std::chrono::system_clock::now();

    try {
        // Paths (use DPM variant for frames, ignore its detections)
        std::string sequencePath = config_.projectBasePath + "/MOT17-Data/MOT17/train/" +
                                   sequence + "-DPM";
        std::string outputPath = config_.projectBasePath + "/output/v" +
                                config_.outputVersion + "/end-to-end/" +
                                sequence + "-YOLO-" + result.iouType + ".txt";

        result.outputFile = outputPath;

        // Initialize components
        MotSequenceReader reader(sequencePath);

        Impression impression;
        std::string modelPath = config_.projectBasePath + config_.yoloModelPath;
        impression.createInfrance(modelPath, cv::Size(640, 640), "classes.txt", config_.runOnGPU);

        // Configure impression's tracker
        // Note: Impression uses internal KalmanIoUTracker
        // You may need to add a method to Impression to set IoU type
        // For now, this assumes default settings

        auto logger = std::make_shared<MotChallengeLogger>(outputPath, true);
        impression.setTelemetry(logger);

        // Process frames
        cv::Mat frame;
        int frameCount = 0;
        auto startTime = timeSinceEpochMillisec();

        while (reader.nextFrame(frame)) {
            // Set frame metadata
            ITrackerTelemetry::FrameMetadata metadata;
            metadata.frameIndex = frameCount;
            metadata.sourceSize = reader.getFrameSize();
            metadata.processedSize = frame.size();
            metadata.timestamp = frameCount / reader.getFPS();
            logger->onFrameStart(metadata);

            // Run YOLO + tracking
            cv::Mat outputFrame = impression.stalkImageAdvance(frame);

            frameCount++;
        }

        logger->flush();

        auto endTime = timeSinceEpochMillisec();
        result.totalFrames = frameCount;
        result.processingTimeMs = endTime - startTime;
        result.fps = (frameCount * 1000.0) / result.processingTimeMs;
        // Note: totalDetections and totalTracks not easily accessible from Impression
        result.success = true;

    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = e.what();
    }

    return result;
}
```

---

## 5. Helper Functions

### 5.1 createOutputDirectories()

```cpp
void BenchmarkRunner::createOutputDirectories() {
    std::string baseOutput = config_.projectBasePath + "/output/v" + config_.outputVersion;

    if (config_.mode == BenchmarkMode::TRACKER_ONLY || config_.mode == BenchmarkMode::BOTH) {
        fs::create_directories(baseOutput + "/tracker-only");
    }

    if (config_.mode == BenchmarkMode::END_TO_END || config_.mode == BenchmarkMode::BOTH) {
        fs::create_directories(baseOutput + "/end-to-end");
    }
}
```

---

### 5.2 printHeader()

```cpp
void BenchmarkRunner::printHeader() {
    std::cout << "\n";
    std::cout << "=================================\n";
    std::cout << "  MOT17 Benchmark System\n";
    std::cout << "  Version: " << config_.outputVersion << "\n";
    std::cout << "  Mode: " << benchmarkModeToString(config_.mode) << "\n";
    std::cout << "=================================\n";
}
```

---

### 5.3 printProgress()

```cpp
void BenchmarkRunner::printProgress(int current, int total, const TestResult& result) {
    std::cout << "[" << std::setw(3) << current << "/" << std::setw(3) << total << "] ";
    std::cout << result.sequence << "-" << result.detector << "-" << result.iouType;

    if (result.success) {
        std::cout << " ... Done ("
                  << std::fixed << std::setprecision(1)
                  << result.processingTimeMs / 1000.0 << "s, "
                  << std::fixed << std::setprecision(1)
                  << result.fps << " FPS)\n";
    } else {
        std::cout << " ... FAILED (" << result.errorMessage << ")\n";
    }
}
```

---

### 5.4 printSummary()

```cpp
void BenchmarkRunner::printSummary(const std::string& mode,
                                    const std::vector<TestResult>& results) {
    int successCount = 0;
    double totalTime = 0.0;
    double totalFps = 0.0;

    for (const auto& result : results) {
        if (result.success) {
            successCount++;
            totalTime += result.processingTimeMs;
            totalFps += result.fps;
        }
    }

    std::cout << "\n" << mode << " Summary:\n";
    std::cout << "  Total: " << results.size() << " tests\n";
    std::cout << "  Success: " << successCount << " ("
              << (successCount * 100.0 / results.size()) << "%)\n";
    std::cout << "  Failed: " << (results.size() - successCount) << "\n";
    std::cout << "  Duration: " << std::fixed << std::setprecision(1)
              << totalTime / 1000.0 << "s ("
              << totalTime / 60000.0 << " minutes)\n";
    std::cout << "  Avg FPS: " << std::fixed << std::setprecision(1)
              << totalFps / successCount << "\n";
}
```

---

### 5.5 saveSummaryJSON()

```cpp
void BenchmarkRunner::saveSummaryJSON(const std::string& mode,
                                       const std::vector<TestResult>& results) {
    std::string outputPath = config_.projectBasePath + "/output/v" +
                            config_.outputVersion + "/" + mode + "/benchmark_summary.json";

    std::ofstream file(outputPath);
    if (!file.is_open()) {
        std::cerr << "Failed to write summary JSON: " << outputPath << std::endl;
        return;
    }

    // Manual JSON construction (simple approach)
    file << "{\n";
    file << "  \"version\": \"" << config_.outputVersion << "\",\n";
    file << "  \"mode\": \"" << mode << "\",\n";
    file << "  \"timestamp\": \"" << timestampToISO8601(std::chrono::system_clock::now()) << "\",\n";
    file << "  \"total_tests\": " << results.size() << ",\n";

    int successCount = std::count_if(results.begin(), results.end(),
                                     [](const TestResult& r) { return r.success; });
    file << "  \"successful_tests\": " << successCount << ",\n";
    file << "  \"failed_tests\": " << (results.size() - successCount) << ",\n";

    double totalTime = 0.0;
    double totalFps = 0.0;
    for (const auto& r : results) {
        if (r.success) {
            totalTime += r.processingTimeMs;
            totalFps += r.fps;
        }
    }

    file << "  \"total_duration_seconds\": " << (totalTime / 1000.0) << ",\n";
    file << "  \"avg_fps\": " << (totalFps / successCount) << ",\n";

    // Results array
    file << "  \"results\": [\n";
    for (size_t i = 0; i < results.size(); i++) {
        const auto& r = results[i];
        file << "    {\n";
        file << "      \"test_id\": " << (i + 1) << ",\n";
        file << "      \"sequence\": \"" << r.sequence << "\",\n";
        file << "      \"detector\": \"" << r.detector << "\",\n";
        file << "      \"iou_type\": \"" << r.iouType << "\",\n";
        file << "      \"output_file\": \"" << r.outputFile << "\",\n";
        file << "      \"total_frames\": " << r.totalFrames << ",\n";
        file << "      \"processing_time_ms\": " << r.processingTimeMs << ",\n";
        file << "      \"fps\": " << r.fps << ",\n";
        file << "      \"total_detections\": " << r.totalDetections << ",\n";
        file << "      \"total_tracks\": " << r.totalTracks << ",\n";
        file << "      \"success\": " << (r.success ? "true" : "false");
        if (!r.success) {
            file << ",\n      \"error_message\": \"" << r.errorMessage << "\"";
        }
        file << "\n    }";
        if (i < results.size() - 1) file << ",";
        file << "\n";
    }
    file << "  ]\n";
    file << "}\n";

    file.close();
    std::cout << "Summary saved: " << outputPath << "\n";
}
```

---

### 5.6 saveCombinedReport()

```cpp
void BenchmarkRunner::saveCombinedReport() {
    std::string outputPath = config_.projectBasePath + "/output/v" +
                            config_.outputVersion + "/combined_report.json";

    std::ofstream file(outputPath);
    if (!file.is_open()) return;

    file << "{\n";
    file << "  \"version\": \"" << config_.outputVersion << "\",\n";
    file << "  \"timestamp\": \"" << timestampToISO8601(std::chrono::system_clock::now()) << "\",\n";

    // Tracker-only summary
    file << "  \"tracker_only\": {\n";
    file << "    \"total_tests\": " << trackerOnlyResults_.size() << ",\n";
    double toTotalTime = 0.0, toTotalFps = 0.0;
    for (const auto& r : trackerOnlyResults_) {
        if (r.success) {
            toTotalTime += r.processingTimeMs;
            toTotalFps += r.fps;
        }
    }
    file << "    \"total_duration_seconds\": " << (toTotalTime / 1000.0) << ",\n";
    file << "    \"avg_fps\": " << (toTotalFps / trackerOnlyResults_.size()) << "\n";
    file << "  },\n";

    // End-to-end summary
    file << "  \"end_to_end\": {\n";
    file << "    \"total_tests\": " << endToEndResults_.size() << ",\n";
    double etTotalTime = 0.0, etTotalFps = 0.0;
    for (const auto& r : endToEndResults_) {
        if (r.success) {
            etTotalTime += r.processingTimeMs;
            etTotalFps += r.fps;
        }
    }
    file << "    \"total_duration_seconds\": " << (etTotalTime / 1000.0) << ",\n";
    file << "    \"avg_fps\": " << (etTotalFps / endToEndResults_.size()) << "\n";
    file << "  },\n";

    // Combined summary
    file << "  \"summary\": {\n";
    file << "    \"total_tests\": " << (trackerOnlyResults_.size() + endToEndResults_.size()) << ",\n";
    file << "    \"total_duration_seconds\": " << ((toTotalTime + etTotalTime) / 1000.0) << ",\n";
    file << "    \"total_duration_minutes\": " << ((toTotalTime + etTotalTime) / 60000.0) << "\n";
    file << "  }\n";
    file << "}\n";

    file.close();
    std::cout << "Combined report saved: " << outputPath << "\n";
}
```

---

## 6. run_trackeval.sh Güncellemeleri

**Mevcut dosyaya eklenecek değişiklikler:**

```bash
#!/bin/bash

###############################################################################
# TrackEval Runner Script for MOT17 Evaluation (VERSION SUPPORT)
#
# Kullanım:
#   ./run_trackeval.sh                          # Latest version, tracker-only
#   ./run_trackeval.sh v1.0.0                   # Specific version, tracker-only
#   ./run_trackeval.sh v1.0.0 tracker-only      # Explicit mode
#   ./run_trackeval.sh v1.0.0 end-to-end        # End-to-end mode
###############################################################################

# Version parametresi (default: latest)
VERSION=${1:-$(ls -td output/v*/ 2>/dev/null | head -1 | xargs basename)}
if [ -z "$VERSION" ]; then
    echo "Error: No version found in output/ directory"
    exit 1
fi

# Mode parametresi (default: tracker-only)
MODE=${2:-"tracker-only"}
if [ "$MODE" != "tracker-only" ] && [ "$MODE" != "end-to-end" ]; then
    echo "Error: Invalid mode. Use 'tracker-only' or 'end-to-end'"
    exit 1
fi

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}  TrackEval Runner for MOT17${NC}"
echo -e "${BLUE}  Version: $VERSION${NC}"
echo -e "${BLUE}  Mode: $MODE${NC}"
echo -e "${BLUE}=====================================${NC}\n"

# ... (mevcut dizin kontrolü kodları) ...

# TRACKER_DIR'i versiyonlu yap
TRACKER_NAME="opencv-yolo-$VERSION-$MODE"
TRACKER_DIR="$TRACKEVAL_DIR/data/trackers/mot_challenge/MOT17-train/$TRACKER_NAME/data"

# Tracker dizini oluştur
echo -e "\n${YELLOW}[3/6] Tracker dizini hazırlanıyor ($TRACKER_NAME)...${NC}"
mkdir -p "$TRACKER_DIR"

# Sonuç dosyalarını kopyala (versiyonlu)
echo -e "\n${YELLOW}[4/6] Tracker sonuçları kopyalanıyor...${NC}"
SOURCE_DIR="$SCRIPT_DIR/$VERSION/$MODE"

if [ ! -d "$SOURCE_DIR" ]; then
    echo -e "${RED}HATA: Kaynak dizin bulunamadı: $SOURCE_DIR${NC}"
    exit 1
fi

COPIED_COUNT=0
for file in "$SOURCE_DIR"/*.txt; do
    if [ -f "$file" ] && [[ "$(basename "$file")" == MOT17-* ]]; then
        cp "$file" "$TRACKER_DIR/"
        echo -e "  → $(basename "$file")"
        COPIED_COUNT=$((COPIED_COUNT + 1))
    fi
done

if [ $COPIED_COUNT -eq 0 ]; then
    echo -e "${RED}HATA: Hiç tracker sonucu bulunamadı!${NC}"
    exit 1
fi
echo -e "${GREEN}✓ $COPIED_COUNT dosya kopyalandı${NC}"

# ... (seqmaps oluşturma kodu aynı) ...

# TrackEval çalıştır (versiyonlu tracker name)
python3 scripts/run_mot_challenge.py \
    --BENCHMARK MOT17 \
    --SPLIT_TO_EVAL train \
    --TRACKERS_TO_EVAL "$TRACKER_NAME" \
    --METRICS HOTA CLEAR Identity \
    --USE_PARALLEL False \
    --NUM_PARALLEL_CORES 1 \
    --PLOT_CURVES False

# ... (mevcut sonuç mesajları) ...
```

---

## 7. compare_versions.sh (Tam Dosya)

```bash
#!/bin/bash

###############################################################################
# Version Comparison Script
#
# Compares tracking performance between two benchmark versions
#
# Usage:
#   ./scripts/compare_versions.sh v1.0.0 v1.0.1
#   ./scripts/compare_versions.sh v1.0.0 v1.0.1 tracker-only
###############################################################################

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Parameters
V1=$1
V2=$2
MODE=${3:-"tracker-only"}

if [ -z "$V1" ] || [ -z "$V2" ]; then
    echo -e "${RED}Usage: $0 <version1> <version2> [mode]${NC}"
    echo -e "Example: $0 v1.0.0 v1.0.1 tracker-only"
    exit 1
fi

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}  MOT17 Version Comparison${NC}"
echo -e "${BLUE}  ${V1} vs ${V2}${NC}"
echo -e "${BLUE}  Mode: $MODE${NC}"
echo -e "${BLUE}=====================================${NC}\n"

# Check if versions exist
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
V1_DIR="$PROJECT_DIR/output/$V1/$MODE"
V2_DIR="$PROJECT_DIR/output/$V2/$MODE"

if [ ! -d "$V1_DIR" ]; then
    echo -e "${RED}Error: Version $V1 not found ($V1_DIR)${NC}"
    exit 1
fi

if [ ! -d "$V2_DIR" ]; then
    echo -e "${RED}Error: Version $V2 not found ($V2_DIR)${NC}"
    exit 1
fi

# Extract metrics from JSON summaries
V1_JSON="$V1_DIR/benchmark_summary.json"
V2_JSON="$V2_DIR/benchmark_summary.json"

if [ ! -f "$V1_JSON" ] || [ ! -f "$V2_JSON" ]; then
    echo -e "${RED}Error: Summary JSON files not found${NC}"
    exit 1
fi

# Use jq to parse JSON (install: sudo apt install jq)
if ! command -v jq &> /dev/null; then
    echo -e "${YELLOW}Warning: jq not installed. Install with: sudo apt install jq${NC}"
    echo -e "${YELLOW}Showing raw JSON summaries instead:${NC}\n"
    echo -e "${CYAN}=== $V1 ===${NC}"
    cat "$V1_JSON"
    echo -e "\n${CYAN}=== $V2 ===${NC}"
    cat "$V2_JSON"
    exit 0
fi

# Extract metrics
V1_TESTS=$(jq -r '.successful_tests' "$V1_JSON")
V2_TESTS=$(jq -r '.successful_tests' "$V2_JSON")
V1_FPS=$(jq -r '.avg_fps' "$V1_JSON")
V2_FPS=$(jq -r '.avg_fps' "$V2_JSON")
V1_TIME=$(jq -r '.total_duration_seconds' "$V1_JSON")
V2_TIME=$(jq -r '.total_duration_seconds' "$V2_JSON")

# Calculate deltas
FPS_DELTA=$(echo "$V2_FPS - $V1_FPS" | bc -l)
FPS_PERCENT=$(echo "scale=2; ($FPS_DELTA / $V1_FPS) * 100" | bc -l)
TIME_DELTA=$(echo "$V2_TIME - $V1_TIME" | bc -l)
TIME_PERCENT=$(echo "scale=2; ($TIME_DELTA / $V1_TIME) * 100" | bc -l)

# Print comparison table
echo -e "${CYAN}Performance Comparison:${NC}\n"
printf "┌─────────────────────┬──────────────┬──────────────┬──────────────┐\n"
printf "│ %-19s │ %-12s │ %-12s │ %-12s │\n" "Metric" "$V1" "$V2" "Delta"
printf "├─────────────────────┼──────────────┼──────────────┼──────────────┤\n"
printf "│ %-19s │ %12s │ %12s │ %12s │\n" "Successful Tests" "$V1_TESTS" "$V2_TESTS" "$(($V2_TESTS - $V1_TESTS))"
printf "│ %-19s │ %10.1f │ %10.1f │ %+10.1f │\n" "Avg FPS" "$V1_FPS" "$V2_FPS" "$FPS_DELTA"
printf "│ %-19s │ %12s │ %12s │ %+11.1f%% │\n" "  (Improvement)" "" "" "$FPS_PERCENT"
printf "│ %-19s │ %10.1fs │ %10.1fs │ %+10.1fs │\n" "Total Time" "$V1_TIME" "$V2_TIME" "$TIME_DELTA"
printf "│ %-19s │ %12s │ %12s │ %+11.1f%% │\n" "  (Change)" "" "" "$TIME_PERCENT"
printf "└─────────────────────┴──────────────┴──────────────┴──────────────┘\n"

echo -e "\n${YELLOW}Note: For HOTA/MOTA/IDF1 metrics, run TrackEval on both versions:${NC}"
echo -e "  cd output && ./run_trackeval.sh $V1 $MODE"
echo -e "  cd output && ./run_trackeval.sh $V2 $MODE"
echo -e "  Then compare the pedestrian_summary.txt files\n"
```

---

## 📝 Kullanım Notları

### Copy-Paste Sırası

1. **CMakeLists.txt**: İlk iki section'ı copy-paste et
2. **benchmark_config.h**: Tam dosyayı oluştur
3. **mot17_benchmark.cpp**: İskelet + implementation'ları sırayla ekle
4. **run_trackeval.sh**: Mevcut dosyaya VERSION/MODE desteği ekle
5. **scripts/compare_versions.sh**: Yeni dosya oluştur, `chmod +x` yap

### Build ve Test

```bash
# 1. Build
cd build
cmake .. && make mot17_benchmark

# 2. Küçük test
../mot17_benchmark --sequences MOT17-02 --detectors DPM --iou-types CIOU

# 3. Full test
../mot17_benchmark

# 4. Değerlendirme
cd ../output
./run_trackeval.sh v1.0.0 tracker-only

# 5. Version karşılaştırma
../scripts/compare_versions.sh v1.0.0 v1.0.1
```

---

**Son Güncelleme:** 2025-01-20
**Durum:** ✅ Tüm şablonlar hazır
