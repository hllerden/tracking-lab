// how to USE
/**
 * @file mot17_benchmark.cpp
 * @brief MOT17 Benchmark System - Main executable
 * @author Halil Erden
 * @date 2025-01-20
 *
 * Headless benchmark runner for systematic tracker testing.
 * Modes:
 *   - Tracker-Only: Uses pre-computed detections (det.txt)
 *   - End-to-End: Uses YOLO inference + tracking
 * Command line overrides:
 *   --mode tracker-only|end-to-end|both
 *   --sequences MOT17-02,MOT17-04,...
 *   --detectors DPM,FRCNN,...
 *   --iou-types IOU,GIOU,DIOU,...
 * Example usage:
 *   ./mot17_benchmark
 *   ./mot17_benchmark --mode tracker-only --sequences MOT17-02,MOT17-04
 *   ./mot17_benchmark --mode tracker-only --detectors DPM,FRCNN --iou-types IOU,GIOU
 *   ./mot17_benchmark --mode end-to-end --sequences MOT17-10 --iou-types CIOU
 *   ./mot17_benchmark --mode both --detectors DPM,FRCNN,SDP --iou-types IOU,GIOU,DIOU
 *   ./mot17_tracker_benchmark  (yalnız CIOU, 7×3 sabit test matrisi)
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>

// Project headers
#include "benchmark_config.h"
#include "mot/mot_detection_reader.h"
#include "trackers/tracker_factory.h"
#include "mot/mot_sequence_reader.h"
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

#ifndef MOT17_BENCHMARK_NO_CLI
int main(int argc, char** argv) {
    // Create configuration
    BenchmarkConfig config;
    config.projectBasePath = PROJECT_ROOT_DIR;

    auto trimInPlace = [](std::string& s) {
        size_t start = 0;
        while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
            ++start;
        }
        size_t end = s.size();
        while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
            --end;
        }
        if (start > 0 || end < s.size()) {
            s.assign(s, start, end - start);
        }
    };

    auto splitCommaSeparated = [&](const std::string& input) {
        std::vector<std::string> tokens;
        size_t pos = 0;
        while (pos <= input.size()) {
            size_t comma = input.find(',', pos);
            size_t end = (comma == std::string::npos) ? input.size() : comma;
            std::string token = input.substr(pos, end - pos);
            trimInPlace(token);
            if (!token.empty()) {
                tokens.emplace_back(std::move(token));
            }
            if (comma == std::string::npos) {
                break;
            }
            pos = comma + 1;
        }
        return tokens;
    };

    auto normalizeModeValue = [](std::string value) {
        for (char& ch : value) {
            if (ch == '_') {
                ch = '-';
            } else {
                ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            }
        }
        return value;
    };

    auto toUpperFast = [](std::string value) {
        for (char& ch : value) {
            ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
        }
        return value;
    };

    for (int i = 1; i < argc; ++i) {
        std::string rawArg = argv[i];
        std::string key = rawArg;
        std::string value;
        auto eqPos = rawArg.find('=');
        if (eqPos != std::string::npos) {
            key = rawArg.substr(0, eqPos);
            value = rawArg.substr(eqPos + 1);
        }

        if (key == "--mode" || key == "--sequences" || key == "--detectors" || key == "--iou-types") {
            if (value.empty() && i + 1 < argc) {
                value = argv[++i];
            }
            if (value.empty()) {
                continue;
            }
        } else {
            continue;
        }

        if (key == "--mode") {
            value = normalizeModeValue(value);
            if (value == "tracker-only") {
                config.mode = BenchmarkMode::TRACKER_ONLY;
            } else if (value == "end-to-end") {
                config.mode = BenchmarkMode::END_TO_END;
            } else if (value == "both") {
                config.mode = BenchmarkMode::BOTH;
            }
            continue;
        }

        if (key == "--sequences") {
            auto list = splitCommaSeparated(value);
            if (!list.empty()) {
                config.sequences = std::move(list);
            }
            continue;
        }

        if (key == "--detectors") {
            auto list = splitCommaSeparated(value);
            if (!list.empty()) {
                config.detectors = std::move(list);
            }
            continue;
        }

        if (key == "--iou-types") {
            auto list = splitCommaSeparated(value);
            if (!list.empty()) {
                std::vector<KalmanIoUConfig::IoUType> parsed;
                parsed.reserve(list.size());
                for (auto& entry : list) {
                    std::string upper = toUpperFast(std::move(entry));
                    if (upper == "IOU") {
                        parsed.emplace_back(KalmanIoUConfig::IoUType::IOU);
                    } else if (upper == "GIOU") {
                        parsed.emplace_back(KalmanIoUConfig::IoUType::GIOU);
                    } else if (upper == "DIOU") {
                        parsed.emplace_back(KalmanIoUConfig::IoUType::DIOU);
                    } else if (upper == "CIOU") {
                        parsed.emplace_back(KalmanIoUConfig::IoUType::CIOU);
                    } else if (upper == "SIOU") {
                        parsed.emplace_back(KalmanIoUConfig::IoUType::SIOU);
                    } else if (upper == "AIOU") {
                        parsed.emplace_back(KalmanIoUConfig::IoUType::AIOU);
                    }
                }
                if (!parsed.empty()) {
                    config.iouTypes = std::move(parsed);
                }
            }
            continue;
        }
    }

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
#endif // MOT17_BENCHMARK_NO_CLI

// ==================== BENCHMARK RUNNER IMPLEMENTATION ====================

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

        auto trackerOwner = TrackerFactory::create(DEFAULT_TRACKER_TYPE);
        IObjectTracker* tracker = trackerOwner.get();

        KalmanIoUConfig trackerConfig = TrackerFactory::getDefaultConfig(DEFAULT_TRACKER_TYPE);
        trackerConfig.iouType = iouType;
        trackerConfig.iouThreshold = config_.iouThreshold;
        trackerConfig.maxLostFrames = config_.maxLostFrames;
        trackerConfig.usePredictionInLost = config_.usePredictionInLost;
        trackerConfig.removeOutOfBounds = config_.removeOutOfBounds;
        trackerConfig.cameraBounds = cv::Rect(0, 0, reader.getFrameSize().width, reader.getFrameSize().height);
        TrackerFactory::applyConfig(tracker, trackerConfig);

        auto logger = std::make_shared<MotChallengeLogger>(outputPath, true);
        TrackerFactory::applyTelemetry(tracker, logger,
                                       reader.getFrameSize(), reader.getFrameSize());

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
            tracker->update(detections);

            frameCount++;
        }

        logger->flush();

        auto endTime = timeSinceEpochMillisec();
        result.totalFrames = frameCount;
        result.processingTimeMs = endTime - startTime;
        result.fps = (frameCount * 1000.0) / result.processingTimeMs;
        result.totalDetections = totalDetections;
        result.totalTracks = tracker->getTotalTrackCount();
        result.success = true;

    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = e.what();
    }

    return result;
}

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
        // For now, using default settings

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

// ==================== HELPER FUNCTIONS ====================

void BenchmarkRunner::createOutputDirectories() {
    std::string baseOutput = config_.projectBasePath + "/output/v" + config_.outputVersion;

    if (config_.mode == BenchmarkMode::TRACKER_ONLY || config_.mode == BenchmarkMode::BOTH) {
        fs::create_directories(baseOutput + "/tracker-only");
    }

    if (config_.mode == BenchmarkMode::END_TO_END || config_.mode == BenchmarkMode::BOTH) {
        fs::create_directories(baseOutput + "/end-to-end");
    }
}

void BenchmarkRunner::printHeader() {
    std::cout << "\n";
    std::cout << "=================================\n";
    std::cout << "  MOT17 Benchmark System\n";
    std::cout << "  Version: " << config_.outputVersion << "\n";
    std::cout << "  Mode: " << benchmarkModeToString(config_.mode) << "\n";
    std::cout << "=================================\n";
}

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
