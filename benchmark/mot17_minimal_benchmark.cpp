/**
 * @file mot17_minimal_benchmark.cpp
 * @brief Minimal MOT17 Benchmark - Fast testing with fixed IoU type
 * @author Halil Erden
 * @date 2025-01-21
 *
 * FAST BENCHMARK for quick algorithm testing.
 *
 * Test Matrix: sequences × detectors (NO IoU variation)
 * Default: 7 sequences × 3 detectors = 21 tests
 *
 * CONFIGURATION (Edit in code below):
 *   - IoU Type: Line 50 (default: CIOU)
 *   - Sequences: Line 56-58 (default: all 7 MOT17 sequences)
 *   - Detectors: Line 60 (default: DPM, FRCNN, SDP)
 *
 * OUTPUT:
 *   - Directory: output/v{VERSION}/minimal/
 *   - Filename: MOT17-{SEQ}-{DET}.txt (e.g., MOT17-02-DPM.txt)
 *   - No IoU suffix in filenames
 *
 * USAGE:
 *   ./mot17_minimal_benchmark
 *
 * EVALUATION:
 *   cd ../output
 *   ./run_trackeval.sh --minimal
 *   ./run_trackeval.sh v1.0.1 --minimal
 */

// ==================== INCLUDES ====================

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

#include "benchmark_config.h"
#include "mot/mot_sequence_reader.h"
#include "mot/mot_detection_reader.h"
#include "trackers/tracker_factory.h"
#include "TrackerTelemtryLogger/MotChallengeLogger.h"

// ==================== CONFIGURATION ====================

// Tracker Type Selection (CHANGE HERE to test different trackers)
#define MINIMAL_TRACKER_TYPE TrackerType::BYTETRACK

// IoU Type Selection (CHANGE HERE for different IoU algorithms)
#define MINIMAL_IOU_TYPE KalmanIoUConfig::IoUType::CIOU
#define MINIMAL_IOU_NAME "CIOU"

// Test sequences (CHANGE HERE to test specific sequences)
// static const std::vector<std::string> MINIMAL_SEQUENCES = {
//     "MOT17-02"
// };
static const std::vector<std::string> MINIMAL_SEQUENCES = {
    "MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09",
    "MOT17-10", "MOT17-11", "MOT17-13"
};

// Detectors (CHANGE HERE to test specific detectors)
static const std::vector<std::string> MINIMAL_DETECTORS = {/*"DPM",*/ "FRCNN"/*, "SDP"*/};

// ==================== HELPER UTILITIES ====================

namespace fs = std::filesystem;
using namespace std;
using namespace cv;

inline int64_t timeSinceEpochMillisec() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

inline std::string timestampToISO8601(const std::chrono::system_clock::time_point& tp) {
    auto time = std::chrono::system_clock::to_time_t(tp);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%dT%H:%M:%S");
    return ss.str();
}

// ==================== MINIMAL BENCHMARK IMPLEMENTATION ====================

void printProgress(int current, int total, const TestResult& result) {
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

TestResult runMinimalTest(
    const std::string& projectBasePath,
    const std::string& sequence,
    const std::string& detector,
    KalmanIoUConfig::IoUType iouType,
    const std::string& outputDir,
    float iouThreshold,
    int maxLostFrames,
    bool usePredictionInLost,
    bool removeOutOfBounds
) {
    TestResult result;
    result.sequence = sequence;
    result.detector = detector;
    result.iouType = MINIMAL_IOU_NAME;
    result.timestamp = std::chrono::system_clock::now();

    try {
        std::string sequencePath = projectBasePath + "/MOT17-Data/MOT17/train/" +
                                   sequence + "-" + detector;
        std::string outputPath = outputDir + "/" + sequence + "-" + detector + ".txt";
        result.outputFile = outputPath;

        MotSequenceReader reader(sequencePath);
        MotDetectionReader detReader(sequencePath + "/det/det.txt");

        // Create tracker using factory
        KalmanIoUConfig trackerConfig;
        trackerConfig.iouType = iouType;
        trackerConfig.iouThreshold = iouThreshold;
        trackerConfig.maxLostFrames = maxLostFrames;
        trackerConfig.usePredictionInLost = usePredictionInLost;
        trackerConfig.removeOutOfBounds = removeOutOfBounds;
        trackerConfig.cameraBounds = cv::Rect(0, 0,
                                              reader.getFrameSize().width,
                                              reader.getFrameSize().height);

        auto trackerPtr = TrackerFactory::create(MINIMAL_TRACKER_TYPE, trackerConfig);
        auto* tracker = trackerPtr.get();

        auto logger = std::make_shared<MotChallengeLogger>(outputPath, true);

        // Set telemetry and frame size hint for each tracker type
        if (auto* kt6 = dynamic_cast<KalmanIoUTracker6State*>(tracker)) {
            kt6->setTelemetry(logger);
            kt6->setFrameSizeHint(reader.getFrameSize(), reader.getFrameSize());
        } else if (auto* kt = dynamic_cast<KalmanIoUTracker*>(tracker)) {
            kt->setTelemetry(logger);
            kt->setFrameSizeHint(reader.getFrameSize(), reader.getFrameSize());
        } else if (auto* bt = dynamic_cast<KalmanIoUByteTrack*>(tracker)) {
            bt->setTelemetry(logger);
            bt->setFrameSizeHint(reader.getFrameSize(), reader.getFrameSize());
        }

        cv::Mat frame;
        int frameCount = 0;
        int totalDetections = 0;
        auto startTime = timeSinceEpochMillisec();

        while (reader.nextFrame(frame)) {
            int frameIndex = frameCount + 1;

            ITrackerTelemetry::FrameMetadata metadata;
            metadata.frameIndex = frameCount;
            metadata.sourceSize = reader.getFrameSize();
            metadata.processedSize = frame.size();
            metadata.timestamp = frameCount / reader.getFPS();
            logger->onFrameStart(metadata);

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

void printMinimalSummary(const std::vector<TestResult>& results) {
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

    std::cout << "\nMinimal Benchmark Summary:\n";
    std::cout << "  IoU Type: " << MINIMAL_IOU_NAME << "\n";
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

void saveMinimalSummaryJSON(const std::string& outputDir,
                           const std::vector<TestResult>& results,
                           const std::string& version) {
    std::string outputPath = outputDir + "/benchmark_summary.json";

    std::ofstream file(outputPath);
    if (!file.is_open()) {
        std::cerr << "Failed to write summary JSON: " << outputPath << std::endl;
        return;
    }

    file << "{\n";
    file << "  \"version\": \"" << version << "\",\n";
    file << "  \"mode\": \"minimal\",\n";
    file << "  \"iou_type\": \"" << MINIMAL_IOU_NAME << "\",\n";
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

// ==================== MAIN ====================

int main(int /*argc*/, char** /*argv*/) {
    BenchmarkConfig config;
    config.projectBasePath = PROJECT_ROOT_DIR;

    std::string version = PROJECT_VERSION;

    std::cout << "\n";
    std::cout << "=========================================\n";
    std::cout << "  MINIMAL MOT17 Benchmark\n";
    std::cout << "  Version: " << version << "\n";
    std::cout << "  IoU Type: " << MINIMAL_IOU_NAME << " (fixed)\n";
    std::cout << "  Sequences: " << MINIMAL_SEQUENCES.size() << "\n";
    std::cout << "  Detectors: " << MINIMAL_DETECTORS.size() << "\n";
    std::cout << "  Total Tests: " << (MINIMAL_SEQUENCES.size() * MINIMAL_DETECTORS.size()) << "\n";
    std::cout << "=========================================\n";

    try {
        std::cout << "\n=== Running Minimal Benchmark (Tracker-Only) ===\n";
        std::cout << "Using pre-computed detections (det.txt)\n";
        std::cout << "IoU Type: " << MINIMAL_IOU_NAME << "\n";

        int totalTests = MINIMAL_SEQUENCES.size() * MINIMAL_DETECTORS.size();
        std::cout << "Total tests: " << totalTests << "\n";
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";

        std::string minimalOutputDir = config.projectBasePath + "/output/v" + version + "/minimal";
        fs::create_directories(minimalOutputDir);

        std::vector<TestResult> results;
        int testIndex = 0;

        for (const auto& sequence : MINIMAL_SEQUENCES) {
            for (const auto& detector : MINIMAL_DETECTORS) {
                testIndex++;

                TestResult result = runMinimalTest(
                    config.projectBasePath,
                    sequence,
                    detector,
                    MINIMAL_IOU_TYPE,
                    minimalOutputDir,
                    config.iouThreshold,
                    config.maxLostFrames,
                    config.usePredictionInLost,
                    config.removeOutOfBounds
                );
                results.push_back(result);

                printProgress(testIndex, totalTests, result);
            }
        }

        printMinimalSummary(results);
        saveMinimalSummaryJSON(minimalOutputDir, results, version);

        std::cout << "\n=== Minimal Benchmark Complete ===\n";
        std::cout << "Results saved to: " << minimalOutputDir << "/\n";
        std::cout << "\nTo evaluate:\n";
        std::cout << "  cd ../output\n";
        std::cout << "  ./run_trackeval.sh --minimal\n";
        std::cout << "  ./run_trackeval.sh v" << version << " --minimal\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }
}
