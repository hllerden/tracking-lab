#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "benchmark_config.h"
#include "mot/mot_detection_reader.h"
#include "mot/mot_sequence_reader.h"
#include "trackers/reid_extractor.h"
#include "trackers/tracker_factory.h"
#include "trackers/bot_sort/bot_sort_tracker.h"
#include "TrackerTelemtryLogger/MotChallengeLogger.h"

namespace fs = std::filesystem;

namespace {

struct TrackerCase {
    TrackerType type;
    std::string name;
};

struct ReIdCase {
    bool enabled;
    std::string name;
};

struct CompareOptions {
    std::string projectBasePath = PROJECT_ROOT_DIR;
    std::string version = PROJECT_VERSION;
    std::vector<std::string> sequences = {
        "MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09",
        "MOT17-10", "MOT17-11", "MOT17-13"
    };
    std::vector<std::string> detectors = {"FRCNN"};
    KalmanIoUConfig::IoUType iouType = KalmanIoUConfig::IoUType::CIOU;
    float iouThreshold = 0.10f;
    int maxLostFrames = 30 * 2;
    bool usePredictionInLost = true;
    bool removeOutOfBounds = true;
    int maxFrames = 0;
#ifdef USE_ORT_REID
    std::string reidModelPath = std::string(PROJECT_ROOT_DIR) + "/models/yolo26n-reid.onnx";
#else
    std::string reidModelPath = std::string(PROJECT_ROOT_DIR) + "/models/yolo26n-reid-sim.onnx";
#endif
    bool reidUseGpu = true;
    std::vector<ReIdCase> reidCases = {
        {false, "reid-off"},
        {true,  "reid-on"}
    };
};

struct CompareResult : TestResult {
    std::string trackerName;
    std::string reidName;
    bool reidEnabled = false;
};

int64_t nowMs() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

std::string timestampToISO8601(const std::chrono::system_clock::time_point& tp) {
    auto time = std::chrono::system_clock::to_time_t(tp);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%dT%H:%M:%S");
    return ss.str();
}

void trimInPlace(std::string& value) {
    size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
        ++start;
    }

    size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }

    if (start > 0 || end < value.size()) {
        value.assign(value, start, end - start);
    }
}

std::vector<std::string> splitCommaSeparated(const std::string& input) {
    std::vector<std::string> values;
    size_t pos = 0;
    while (pos <= input.size()) {
        size_t comma = input.find(',', pos);
        size_t end = (comma == std::string::npos) ? input.size() : comma;
        std::string token = input.substr(pos, end - pos);
        trimInPlace(token);
        if (!token.empty()) {
            values.push_back(token);
        }
        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }
    return values;
}

std::string toUpper(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    return value;
}

bool parseBool(const std::string& value, bool fallback) {
    std::string normalized = toUpper(value);
    if (normalized == "1" || normalized == "TRUE" || normalized == "YES" || normalized == "ON") {
        return true;
    }
    if (normalized == "0" || normalized == "FALSE" || normalized == "NO" || normalized == "OFF") {
        return false;
    }
    return fallback;
}

void printUsage(const char* executable) {
    std::cout
        << "Usage:\n"
        << "  " << executable << " [options]\n\n"
        << "Options:\n"
        << "  --sequences MOT17-02,MOT17-04   Default: all MOT17 train sequences\n"
        << "  --detectors FRCNN,DPM,SDP        Default: FRCNN\n"
        << "  --iou-type CIOU                  One of IOU,GIOU,DIOU,CIOU,SIOU,AIOU\n"
        << "  --iou-threshold 0.2              Association threshold\n"
        << "  --max-lost-frames 50             Track lifetime after misses\n"
        << "  --max-frames 100                 Debug limit, 0 means full sequence\n"
        << "  --use-prediction true|false      Use Kalman prediction while lost\n"
        << "  --remove-out-of-bounds true|false\n"
        << "  --reid-mode both|off|on          Default: both\n"
        << "  --reid-model path.onnx           Default: models/yolo26n-reid*.onnx\n"
        << "  --reid-gpu true|false            Default: true\n"
        << "  --help\n";
}

CompareOptions parseArgs(int argc, char** argv) {
    CompareOptions options;

    for (int i = 1; i < argc; ++i) {
        std::string raw = argv[i];
        std::string key = raw;
        std::string value;
        size_t eq = raw.find('=');
        if (eq != std::string::npos) {
            key = raw.substr(0, eq);
            value = raw.substr(eq + 1);
        }

        auto readValue = [&]() {
            if (value.empty() && i + 1 < argc) {
                value = argv[++i];
            }
            return !value.empty();
        };

        if (key == "--help" || key == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (key == "--sequences" && readValue()) {
            auto parsed = splitCommaSeparated(value);
            if (!parsed.empty()) {
                options.sequences = std::move(parsed);
            }
        } else if (key == "--detectors" && readValue()) {
            auto parsed = splitCommaSeparated(value);
            if (!parsed.empty()) {
                options.detectors = std::move(parsed);
            }
        } else if (key == "--iou-type" && readValue()) {
            options.iouType = stringToIoUType(toUpper(value));
        } else if (key == "--iou-threshold" && readValue()) {
            options.iouThreshold = std::stof(value);
        } else if (key == "--max-lost-frames" && readValue()) {
            options.maxLostFrames = std::stoi(value);
        } else if (key == "--max-frames" && readValue()) {
            options.maxFrames = std::max(0, std::stoi(value));
        } else if (key == "--use-prediction" && readValue()) {
            options.usePredictionInLost = parseBool(value, options.usePredictionInLost);
        } else if (key == "--remove-out-of-bounds" && readValue()) {
            options.removeOutOfBounds = parseBool(value, options.removeOutOfBounds);
        } else if (key == "--reid-mode" && readValue()) {
            const std::string mode = toUpper(value);
            if (mode == "OFF" || mode == "FALSE" || mode == "0") {
                options.reidCases = {{false, "reid-off"}};
            } else if (mode == "ON" || mode == "TRUE" || mode == "1") {
                options.reidCases = {{true, "reid-on"}};
            } else if (mode == "BOTH") {
                options.reidCases = {{false, "reid-off"}, {true, "reid-on"}};
            }
        } else if (key == "--reid-model" && readValue()) {
            options.reidModelPath = value;
        } else if (key == "--reid-gpu" && readValue()) {
            options.reidUseGpu = parseBool(value, options.reidUseGpu);
        }
    }

    return options;
}

KalmanIoUConfig makeConfig(const CompareOptions& options,
                           TrackerType type,
                           const cv::Size& frameSize,
                           bool useReId) {
    KalmanIoUConfig config = TrackerFactory::getDefaultConfig(type);
    config.iouType = options.iouType;
    config.iouThreshold = options.iouThreshold;
    config.maxLostFrames = options.maxLostFrames;
    config.usePredictionInLost = options.usePredictionInLost;
    config.removeOutOfBounds = options.removeOutOfBounds;
    config.cameraBounds = cv::Rect(0, 0, frameSize.width, frameSize.height);
    config.lowConfIouThreshold = 0.05f;
    config.reidAlpha = 0.65f;
    config.featureEmaDecay = 0.9f;
    config.useReId = useReId;
    config.highConfThreshold = 0.55f;
    config.lostStateThreshold = 3;
    return config;
}

BotSortConfig makeBotSortConfig(const CompareOptions& options,
                                 const cv::Size& frameSize,
                                 bool useReId) {
    BotSortConfig cfg = TrackerFactory::getBotSortDefaultConfig();
    // IoUType enums have the same ordinal values in both config types
    cfg.iouType = static_cast<BotSortConfig::IoUType>(static_cast<int>(options.iouType));
    cfg.iouThreshold        = options.iouThreshold;
    cfg.maxLostFrames       = options.maxLostFrames;
    cfg.usePredictionInLost = options.usePredictionInLost;
    cfg.removeOutOfBounds   = options.removeOutOfBounds;
    cfg.cameraBounds        = cv::Rect(0, 0, frameSize.width, frameSize.height);
    cfg.lowConfIouThreshold = 0.05f;
    cfg.reidAlpha           = 0.65f;
    cfg.featureEmaDecay     = 0.9f;
    cfg.useReId             = useReId;
    cfg.highConfThreshold   = 0.55f;
    cfg.lostStateThreshold  = 3;
    cfg.cmcMethod           = BotSortConfig::CmcMethod::SPARSE_LK;
    return cfg;
}

CompareResult runOne(const CompareOptions& options,
                     const TrackerCase& trackerCase,
                     const ReIdCase& reidCase,
                     const std::string& outputBaseDir,
                     const std::string& sequence,
                     const std::string& detector) {
    CompareResult result;
    result.trackerName = trackerCase.name;
    result.reidName = reidCase.name;
    result.reidEnabled = reidCase.enabled;
    result.sequence = sequence;
    result.detector = detector;
    result.iouType = iouTypeToString(options.iouType);
    result.timestamp = std::chrono::system_clock::now();

    try {
        const std::string sequencePath = options.projectBasePath + "/MOT17-Data/MOT17/train/" +
                                         sequence + "-" + detector;
        const std::string trackerOutputDir = outputBaseDir + "/" + trackerCase.name;
        fs::create_directories(trackerOutputDir);

        const std::string outputPath = trackerOutputDir + "/" + sequence + "-" + detector +
                                       "-" + reidCase.name + ".txt";
        result.outputFile = outputPath;

        MotSequenceReader reader(sequencePath);
        MotDetectionReader detReader(sequencePath + "/det/det.txt");

        std::unique_ptr<IObjectTracker> tracker;
        if (trackerCase.type == TrackerType::BOT_SORT) {
            tracker = TrackerFactory::createBotSort(
                makeBotSortConfig(options, reader.getFrameSize(), reidCase.enabled));
        } else {
            tracker = TrackerFactory::create(
                trackerCase.type,
                makeConfig(options, trackerCase.type, reader.getFrameSize(), reidCase.enabled));
        }

        std::unique_ptr<ReidExtractor> reidExtractor;
        if (reidCase.enabled) {
            try {
                reidExtractor = std::make_unique<ReidExtractor>(
                    options.reidModelPath, options.reidUseGpu);
            } catch (const std::exception& e) {
                if (!options.reidUseGpu) {
                    throw;
                }
                std::cerr << "[ReID] GPU init failed for " << trackerCase.name
                          << "; retrying on CPU: " << e.what() << "\n";
                reidExtractor = std::make_unique<ReidExtractor>(
                    options.reidModelPath, false);
            }
        }

        auto logger = std::make_shared<MotChallengeLogger>(outputPath, true);
        TrackerFactory::applyTelemetry(tracker.get(), logger,
                                       reader.getFrameSize(), reader.getFrameSize());

        cv::Mat frame;
        int frameCount = 0;
        int totalDetections = 0;
        const int64_t startMs = nowMs();

        while (reader.nextFrame(frame)) {
            if (options.maxFrames > 0 && frameCount >= options.maxFrames) {
                break;
            }

            // Supply frame to BoT-SORT for Camera Motion Compensation
            if (auto* bst = dynamic_cast<BotSortTracker*>(tracker.get())) {
                bst->setFrame(frame);
            }

            const int motFrameIndex = frameCount + 1;

            ITrackerTelemetry::FrameMetadata metadata;
            metadata.frameIndex = frameCount;
            metadata.sourceSize = reader.getFrameSize();
            metadata.processedSize = frame.size();
            metadata.timestamp = frameCount / reader.getFPS();
            logger->onFrameStart(metadata);

            auto detections = detReader.getDetectionsForFrame(motFrameIndex);
            totalDetections += static_cast<int>(detections.size());

            if (reidExtractor) {
                std::vector<cv::Rect> bboxes;
                bboxes.reserve(detections.size());
                for (const auto& detection : detections) {
                    bboxes.push_back(detection.bbox);
                }
                auto features = reidExtractor->extractBatch(frame, bboxes);
                for (size_t i = 0; i < detections.size() && i < features.size(); ++i) {
                    detections[i].featureVector = features[i];
                }
            }

            tracker->update(detections);
            ++frameCount;
        }

        logger->flush();

        const int64_t elapsedMs = std::max<int64_t>(1, nowMs() - startMs);
        result.totalFrames = frameCount;
        result.processingTimeMs = static_cast<double>(elapsedMs);
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

void printResult(int current, int total, const CompareResult& result) {
    std::cout << "[" << std::setw(3) << current << "/" << std::setw(3) << total << "] "
              << result.trackerName << " "
              << result.sequence << "-" << result.detector << "-"
              << result.reidName;

    if (result.success) {
        std::cout << " ... Done ("
                  << std::fixed << std::setprecision(1)
                  << result.processingTimeMs / 1000.0 << "s, "
                  << result.fps << " FPS, tracks="
                  << result.totalTracks << ")\n";
    } else {
        std::cout << " ... FAILED (" << result.errorMessage << ")\n";
    }
}

void saveSummary(const std::string& outputBaseDir,
                 const CompareOptions& options,
                 const std::vector<CompareResult>& results) {
    const std::string summaryPath = outputBaseDir + "/benchmark_summary.json";
    std::ofstream file(summaryPath);
    if (!file.is_open()) {
        std::cerr << "Failed to write summary: " << summaryPath << "\n";
        return;
    }

    const int successCount = static_cast<int>(std::count_if(
        results.begin(), results.end(), [](const CompareResult& r) { return r.success; }));

    file << "{\n";
    file << "  \"version\": \"" << options.version << "\",\n";
    file << "  \"mode\": \"tracker-compare\",\n";
    file << "  \"iou_type\": \"" << iouTypeToString(options.iouType) << "\",\n";
    file << "  \"iou_threshold\": " << options.iouThreshold << ",\n";
    file << "  \"max_lost_frames\": " << options.maxLostFrames << ",\n";
    file << "  \"max_frames\": " << options.maxFrames << ",\n";
    file << "  \"timestamp\": \"" << timestampToISO8601(std::chrono::system_clock::now()) << "\",\n";
    file << "  \"total_tests\": " << results.size() << ",\n";
    file << "  \"successful_tests\": " << successCount << ",\n";
    file << "  \"failed_tests\": " << (results.size() - successCount) << ",\n";
    file << "  \"results\": [\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        file << "    {\n";
        file << "      \"tracker\": \"" << r.trackerName << "\",\n";
        file << "      \"reid\": \"" << r.reidName << "\",\n";
        file << "      \"reid_enabled\": " << (r.reidEnabled ? "true" : "false") << ",\n";
        file << "      \"sequence\": \"" << r.sequence << "\",\n";
        file << "      \"detector\": \"" << r.detector << "\",\n";
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
        if (i + 1 < results.size()) {
            file << ",";
        }
        file << "\n";
    }

    file << "  ]\n";
    file << "}\n";
    std::cout << "Summary saved: " << summaryPath << "\n";
}

} // namespace

int main(int argc, char** argv) {
    const CompareOptions options = parseArgs(argc, argv);
    const std::vector<TrackerCase> trackers = {
        //klaman 6 state artık kullanımdan kalktı
        // {TrackerType::KALMAN_6STATE, TrackerFactory::getTypeName(TrackerType::KALMAN_6STATE)},
        // kalman 8 state açılmayacak
        //{TrackerType::KALMAN_8STATE, TrackerFactory::getTypeName(TrackerType::KALMAN_8STATE)},
        {TrackerType::BYTETRACK,     TrackerFactory::getTypeName(TrackerType::BYTETRACK)},
        {TrackerType::BOT_SORT,      TrackerFactory::getTypeName(TrackerType::BOT_SORT)}
    };

    const std::string outputBaseDir = options.projectBasePath + "/output/v" +
                                      options.version + "/tracker-compare";
    fs::create_directories(outputBaseDir);

    const int totalTests = static_cast<int>(trackers.size() *
                                            options.reidCases.size() *
                                            options.sequences.size() *
                                            options.detectors.size());

    std::cout << "\n=========================================\n";
    std::cout << "  MOT17 Tracker Comparison Benchmark\n";
    std::cout << "  Version: " << options.version << "\n";
    std::cout << "  Trackers: " << trackers.size() << "\n";
    std::cout << "  ReID modes: " << options.reidCases.size()
              << "  model=" << options.reidModelPath << "\n";
    std::cout << "  Sequences: " << options.sequences.size() << "\n";
    std::cout << "  Detectors: " << options.detectors.size() << "\n";
    std::cout << "  IoU: " << iouTypeToString(options.iouType)
              << " threshold=" << options.iouThreshold << "\n";
    if (options.maxFrames > 0) {
        std::cout << "  Max Frames: " << options.maxFrames << "\n";
    }
    std::cout << "  Total Tests: " << totalTests << "\n";
    std::cout << "=========================================\n\n";

    std::vector<CompareResult> results;
    results.reserve(totalTests);

    int testIndex = 0;
    for (const auto& tracker : trackers) {
        for (const auto& reidCase : options.reidCases) {
            for (const auto& sequence : options.sequences) {
                for (const auto& detector : options.detectors) {
                    ++testIndex;
                    auto result = runOne(options, tracker, reidCase,
                                         outputBaseDir, sequence, detector);
                    printResult(testIndex, totalTests, result);
                    results.push_back(std::move(result));
                }
            }
        }
    }

    saveSummary(outputBaseDir, options, results);

    const int successCount = static_cast<int>(std::count_if(
        results.begin(), results.end(), [](const CompareResult& r) { return r.success; }));

    std::cout << "\n=== Tracker Comparison Benchmark Complete ===\n";
    std::cout << "Success: " << successCount << "/" << results.size() << "\n";
    std::cout << "Raw MOT txt files: " << outputBaseDir << "/<tracker>/\n";
    std::cout << "Evaluate all three trackers:\n";
    std::cout << "  cd output\n";
    std::cout << "  ./run_trackeval_tracker_compare.sh v" << options.version << "\n";

    return successCount == static_cast<int>(results.size()) ? 0 : 1;
}
