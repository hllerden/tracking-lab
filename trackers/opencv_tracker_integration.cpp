#include "opencv_tracker_integration.h"
#include "../inference.h"
#include "../i_object_tracker.h"
#include "../i_tracker_telemetry.h"
#include "../TrackerTelemtryLogger/MotChallengeLogger.h"
#include "../mot/mot_sequence_reader.h"
#include "../mot/mot_detection_reader.h"

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>

// ─── helpers ────────────────────────────────────────────────────────────────

static uint64_t nowMs() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

static double iou(const cv::Rect& a, const cv::Rect& b) {
    cv::Rect inter = a & b;
    if (inter.area() <= 0) return 0.0;
    return static_cast<double>(inter.area()) / (a.area() + b.area() - inter.area());
}

// Reinit threshold: tracker bu IoU altına düşerse sürüklenmiş kabul edilir.
static constexpr double kReinitDriftThreshold = 0.5;

static cv::Ptr<cv::Tracker> makeTracker(CVTrackerType type,
                                        const std::string& vitModelPath = "") {
    switch (type) {
        case CVTrackerType::CSRT: return cv::TrackerCSRT::create();
        case CVTrackerType::KCF:  return cv::TrackerKCF::create();
        case CVTrackerType::MIL:  return cv::TrackerMIL::create();
        case CVTrackerType::VIT: {
            cv::TrackerVit::Params p;
            p.net     = vitModelPath;
            p.backend = cv::dnn::DNN_BACKEND_CUDA;
            p.target  = cv::dnn::DNN_TARGET_CUDA;
            p.tracking_score_threshold = 0.3f;
            return cv::TrackerVit::create(p);
        }
    }
    return cv::TrackerKCF::create();
}

static std::string typeName(CVTrackerType type) {
    switch (type) {
        case CVTrackerType::CSRT: return "CSRT";
        case CVTrackerType::KCF:  return "KCF";
        case CVTrackerType::MIL:  return "MIL";
        case CVTrackerType::VIT:  return "VIT(GPU)";
    }
    return "UNKNOWN";
}

// ─── tracked object ─────────────────────────────────────────────────────────

struct CVTrackedObj {
    int id;
    cv::Ptr<cv::Tracker> tracker;
    cv::Rect bbox;
    cv::Scalar color;
    bool active;
    int lostFrames;
    int age;
    float confidence;
    int classId;
};

static cv::Scalar randomColor(int seed) {
    std::mt19937 rng(seed * 6364136223846793005ULL + 1442695040888963407ULL);
    std::uniform_int_distribution<int> dist(80, 255);
    return cv::Scalar(dist(rng), dist(rng), dist(rng));
}

// ─── sync: YOLO detections → trackers ───────────────────────────────────────

static void syncDetections(
    const std::vector<Detection>& detections,
    std::vector<CVTrackedObj>& trackers,
    const cv::Mat& frame,
    CVTrackerType type,
    int& nextId,
    const std::string& vitModelPath,
    double iouThreshold = 0.3
) {
    std::vector<bool> detMatched(detections.size(), false);

    for (auto& obj : trackers) {
        double bestIoU = iouThreshold;
        int bestDi = -1;
        for (size_t di = 0; di < detections.size(); ++di) {
            if (detMatched[di]) continue;
            double v = iou(obj.bbox, detections[di].box);
            if (v > bestIoU) { bestIoU = v; bestDi = static_cast<int>(di); }
        }
        if (bestDi >= 0) {
            detMatched[bestDi] = true;
            cv::Rect detBbox = detections[bestDi].box;
            // Reinit yalnızca: track kaybedilmişse VEYA tracker sürüklenmişse.
            if (obj.lostFrames > 0 || iou(obj.bbox, detBbox) < kReinitDriftThreshold) {
                obj.tracker = makeTracker(type, vitModelPath);
                obj.tracker->init(frame, detBbox);
            }
            obj.bbox       = detBbox;
            obj.active     = true;
            obj.lostFrames = 0;
            obj.confidence = detections[bestDi].confidence;
        }
    }

    trackers.erase(
        std::remove_if(trackers.begin(), trackers.end(),
            [](const CVTrackedObj& o) { return !o.active; }),
        trackers.end());

    for (size_t di = 0; di < detections.size(); ++di) {
        if (detMatched[di]) continue;
        cv::Rect safe = detections[di].box & cv::Rect(0, 0, frame.cols, frame.rows);
        if (safe.area() <= 0) continue;
        CVTrackedObj obj;
        obj.id         = nextId++;
        obj.color      = randomColor(obj.id);
        obj.bbox       = safe;
        obj.active     = true;
        obj.lostFrames = 0;
        obj.age        = 0;
        obj.confidence = detections[di].confidence;
        obj.classId    = detections[di].class_id;
        obj.tracker    = makeTracker(type, vitModelPath);
        obj.tracker->init(frame, obj.bbox);
        trackers.push_back(std::move(obj));
    }
}

// ─── sync: MOT det.txt detections → trackers ────────────────────────────────

static void syncMotDetections(
    const std::vector<IObjectTracker::Detection>& detections,
    std::vector<CVTrackedObj>& trackers,
    const cv::Mat& frame,
    CVTrackerType type,
    int& nextId,
    const std::string& vitModelPath,
    double iouThreshold = 0.3
) {
    std::vector<bool> detMatched(detections.size(), false);

    for (auto& obj : trackers) {
        double bestIoU = iouThreshold;
        int bestDi = -1;
        for (size_t di = 0; di < detections.size(); ++di) {
            if (detMatched[di]) continue;
            double v = iou(obj.bbox, detections[di].bbox);
            if (v > bestIoU) { bestIoU = v; bestDi = static_cast<int>(di); }
        }
        if (bestDi >= 0) {
            detMatched[bestDi] = true;
            cv::Rect detBbox = detections[bestDi].bbox;
            if (obj.lostFrames > 0 || iou(obj.bbox, detBbox) < kReinitDriftThreshold) {
                obj.tracker = makeTracker(type, vitModelPath);
                obj.tracker->init(frame, detBbox);
            }
            obj.bbox       = detBbox;
            obj.active     = true;
            obj.lostFrames = 0;
            obj.confidence = detections[bestDi].confidence;
        }
    }

    for (size_t di = 0; di < detections.size(); ++di) {
        if (detMatched[di]) continue;
        cv::Rect safe = detections[di].bbox & cv::Rect(0, 0, frame.cols, frame.rows);
        if (safe.area() <= 0) continue;
        CVTrackedObj obj;
        obj.id         = nextId++;
        obj.color      = randomColor(obj.id);
        obj.bbox       = safe;
        obj.active     = true;
        obj.lostFrames = 0;
        obj.age        = 0;
        obj.confidence = detections[di].confidence;
        obj.classId    = detections[di].classId;
        obj.tracker    = makeTracker(type, vitModelPath);
        obj.tracker->init(frame, obj.bbox);
        trackers.push_back(std::move(obj));
    }
}

// ─── per-frame tracker update ───────────────────────────────────────────────

static void updateTrackers(
    std::vector<CVTrackedObj>& trackers,
    const cv::Mat& frame,
    int maxLostFrames = 10
) {
    for (auto& obj : trackers) {
        if (!obj.active) continue;
        cv::Rect newBbox;
        bool ok = obj.tracker->update(frame, newBbox);
        obj.age++;
        if (ok) {
            obj.bbox       = newBbox;
            obj.lostFrames = 0;
        } else {
            obj.lostFrames++;
            if (obj.lostFrames > maxLostFrames)
                obj.active = false;
        }
    }
    trackers.erase(
        std::remove_if(trackers.begin(), trackers.end(),
            [](const CVTrackedObj& o) { return !o.active; }),
        trackers.end());
}

// ─── visualization ───────────────────────────────────────────────────────────

static void drawTrackers(
    cv::Mat& frame,
    const std::vector<CVTrackedObj>& trackers,
    CVTrackerType type,
    int frameCount,
    int refreshInterval,
    double fps
) {
    for (const auto& obj : trackers) {
        cv::rectangle(frame, obj.bbox, obj.color, 2);
        std::string label = "ID:" + std::to_string(obj.id);
        cv::putText(frame, label,
            cv::Point(obj.bbox.x, obj.bbox.y - 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.55, obj.color, 2);
    }
    int nextRefresh = refreshInterval - (frameCount % refreshInterval);
    std::string hud1 = "Tracker: " + typeName(type) +
                       "  Objects: " + std::to_string(trackers.size());
    std::string hud2 = "FPS: " + std::to_string(static_cast<int>(fps)) +
                       "  Next YOLO in: " + std::to_string(nextRefresh) + " frames";
    std::string hud3 = "Keys: 1=CSRT  2=KCF  3=MIL  4=VIT(GPU)  ESC=exit";
    cv::putText(frame, hud1, cv::Point(10, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2);
    cv::putText(frame, hud2, cv::Point(10, 56), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0,220,180), 2);
    cv::putText(frame, hud3, cv::Point(10, 82), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(180,180,180), 1);
}

static void drawTrackersSimple(
    cv::Mat& frame,
    const std::vector<CVTrackedObj>& trackers,
    CVTrackerType type,
    int frameIdx,
    double fps
) {
    for (const auto& obj : trackers) {
        cv::rectangle(frame, obj.bbox, obj.color, 2);
        std::string label = "ID:" + std::to_string(obj.id);
        if (obj.lostFrames > 0) label += " (lost:" + std::to_string(obj.lostFrames) + ")";
        cv::putText(frame, label,
            cv::Point(obj.bbox.x, obj.bbox.y - 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.55, obj.color, 2);
    }
    std::string hud1 = "Tracker: " + typeName(type) +
                       "  Frame: " + std::to_string(frameIdx) +
                       "  Objects: " + std::to_string(trackers.size());
    std::string hud2 = "FPS: " + std::to_string(static_cast<int>(fps)) +
                       "  Source: det.txt";
    cv::putText(frame, hud1, cv::Point(10, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2);
    cv::putText(frame, hud2, cv::Point(10, 56), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0,220,180), 2);
}

// ─── camera/video core loop ──────────────────────────────────────────────────

static int runLoop(
    cv::VideoCapture& cap,
    Inference& inf,
    CVTrackerType& currentType,
    int detectionRefreshFrames,
    const std::string& vitModelPath
) {
    std::vector<CVTrackedObj> tracked;
    int nextId = 0;
    int frameCount = 0;
    double fps = 0.0;
    uint64_t lastFpsTime = nowMs();
    int fpsTick = 0;

    std::cout << "Starting. Keys: 1=CSRT  2=KCF  3=MIL  4=VIT(GPU)  ESC=exit" << std::endl;

    cv::Mat frame;
    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            if (cap.get(cv::CAP_PROP_FRAME_COUNT) > 0) {
                cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                tracked.clear();
                frameCount = 0;
                continue;
            }
            std::cerr << "Failed to read frame" << std::endl;
            break;
        }

        bool runDetection = (frameCount % detectionRefreshFrames == 0);

        if (runDetection) {
            auto dets = inf.runInference(frame);
            syncDetections(dets, tracked, frame, currentType, nextId, vitModelPath);
        } else {
            updateTrackers(tracked, frame);
        }

        fpsTick++;
        uint64_t now = nowMs();
        if (now - lastFpsTime >= 1000) {
            fps = fpsTick * 1000.0 / (now - lastFpsTime);
            fpsTick = 0;
            lastFpsTime = now;
        }

        if (runDetection) {
            cv::putText(frame, "YOLO Detection", cv::Point(10, frame.rows - 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,80,255), 2);
        }
        drawTrackers(frame, tracked, currentType, frameCount, detectionRefreshFrames, fps);

        cv::imshow("OpenCV Trackers - MODE 7", frame);

        int key = cv::waitKey(1);
        if (key == 27) break;
        if (key == '1' && currentType != CVTrackerType::CSRT) {
            currentType = CVTrackerType::CSRT; tracked.clear();
            std::cout << "Switched to CSRT" << std::endl;
        } else if (key == '2' && currentType != CVTrackerType::KCF) {
            currentType = CVTrackerType::KCF; tracked.clear();
            std::cout << "Switched to KCF" << std::endl;
        } else if (key == '3' && currentType != CVTrackerType::MIL) {
            currentType = CVTrackerType::MIL; tracked.clear();
            std::cout << "Switched to MIL" << std::endl;
        } else if (key == '4' && currentType != CVTrackerType::VIT) {
            if (vitModelPath.empty()) {
                std::cout << "VIT: vitModelPath boş, önce modeli belirtin." << std::endl;
            } else {
                currentType = CVTrackerType::VIT; tracked.clear();
                std::cout << "Switched to VIT (GPU)" << std::endl;
            }
        }

        frameCount++;
    }

    cv::destroyAllWindows();
    return 0;
}

// ─── public: camera ──────────────────────────────────────────────────────────

int testOpenCVTrackers(
    int cameraIndex,
    const std::string& yoloModelPath,
    CVTrackerType trackerType,
    int detectionRefreshFrames,
    bool runOnGPU,
    const std::string& vitModelPath
) {
    std::cout << "=== OpenCV Tracker Test (camera " << cameraIndex << ") ===" << std::endl;
    std::cout << "Tracker: " << typeName(trackerType) << "  YOLO refresh: every "
              << detectionRefreshFrames << " frames" << std::endl;

    cv::VideoCapture cap(cameraIndex);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera " << cameraIndex << std::endl;
        return -1;
    }
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);

    Inference inf(yoloModelPath, cv::Size(640,640), "classes.txt", runOnGPU);
    return runLoop(cap, inf, trackerType, detectionRefreshFrames, vitModelPath);
}

// ─── public: video ───────────────────────────────────────────────────────────

int testOpenCVTrackersOnVideo(
    const std::string& videoPath,
    const std::string& yoloModelPath,
    CVTrackerType trackerType,
    int detectionRefreshFrames,
    bool runOnGPU,
    const std::string& vitModelPath
) {
    std::cout << "=== OpenCV Tracker Test (video) ===" << std::endl;
    std::cout << "Video: " << videoPath << std::endl;
    std::cout << "Tracker: " << typeName(trackerType) << "  YOLO refresh: every "
              << detectionRefreshFrames << " frames" << std::endl;

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video: " << videoPath << std::endl;
        return -1;
    }
    Inference inf(yoloModelPath, cv::Size(640,640), "classes.txt", runOnGPU);
    return runLoop(cap, inf, trackerType, detectionRefreshFrames, vitModelPath);
}

// ─── public: MODE 7c — MOT17 tracker-only ────────────────────────────────────

int testOpenCVTrackersTrackerOnly(
    const std::string& sequencePath,
    const std::string& outputPath,
    CVTrackerType trackerType,
    bool showVisualization,
    const std::string& vitModelPath
) {
    std::cout << "=== OpenCV Tracker Test - Tracker-Only Mode (MODE 7c) ===" << std::endl;
    std::cout << "Sequence : " << sequencePath << std::endl;
    std::cout << "Output   : " << outputPath << std::endl;
    std::cout << "Tracker  : " << typeName(trackerType) << std::endl;

    try {
        MotSequenceReader reader(sequencePath);
        std::cout << "Frames: " << reader.getTotalFrames()
                  << "  FPS: " << reader.getFPS()
                  << "  Size: " << reader.getFrameSize() << std::endl;

        MotDetectionReader detReader(sequencePath + "/det/det.txt");

        auto stats = detReader.getStatistics();
        std::cout << "det.txt  total=" << stats.totalDetections
                  << "  avg/frame=" << stats.avgDetPerFrame
                  << "  conf=" << stats.minConfidence << "-" << stats.maxConfidence << std::endl;

        auto logger = std::make_shared<MotChallengeLogger>(outputPath, true);

        std::vector<CVTrackedObj> tracked;
        int nextId = 0;
        int frameCount = 0;

        double fps = 0.0;
        uint64_t lastFpsTime = nowMs();
        int fpsTick = 0;
        uint64_t startTime = nowMs();

        cv::Mat frame;
        while (reader.nextFrame(frame)) {
            int frameIndex = frameCount + 1;  // MOT uses 1-based

            ITrackerTelemetry::FrameMetadata meta;
            meta.frameIndex    = frameCount;
            meta.sourceSize    = reader.getFrameSize();
            meta.processedSize = frame.size();
            meta.timestamp     = frameCount / reader.getFPS();
            logger->onFrameStart(meta);

            // 1. Var olan tracker'ları güncelle (tahmin)
            updateTrackers(tracked, frame, 20);

            // 2. det.txt'den bu frame'in detectionlarını al
            auto dets = detReader.getDetectionsForFrame(frameIndex);

            // 3. Detectionları tracker'larla eşleştir; yeni track'ler oluştur
            if (!dets.empty()) {
                syncMotDetections(dets, tracked, frame, trackerType, nextId, vitModelPath);
            }

            // 4. Telemetri çıktısı
            for (const auto& obj : tracked) {
                IObjectTracker::TrackedResult result;
                result.trackId      = obj.id;
                result.bbox         = obj.bbox;
                result.confidence   = obj.confidence;
                result.classId      = obj.classId;
                result.isPredicted  = (obj.lostFrames > 0);
                result.age          = obj.age;
                result.missedFrames = obj.lostFrames;
                result.velocity     = cv::Point2f(0.f, 0.f);
                logger->onTrackResult(frameCount, result);
            }
            logger->onFrameEnd(frameCount);

            fpsTick++;
            uint64_t now = nowMs();
            if (now - lastFpsTime >= 1000) {
                fps = fpsTick * 1000.0 / (now - lastFpsTime);
                fpsTick = 0;
                lastFpsTime = now;
            }

            if (frameCount % 30 == 0) {
                std::cout << "Frame " << frameIndex << "/" << reader.getTotalFrames()
                          << "  tracks=" << tracked.size()
                          << "  det=" << dets.size()
                          << "  FPS=" << static_cast<int>(fps) << std::endl;
            }

            if (showVisualization) {
                cv::Mat vis = frame.clone();
                drawTrackersSimple(vis, tracked, trackerType, frameIndex, fps);
                cv::imshow("OpenCV Trackers MODE 7c", vis);
                if (cv::waitKey(1) == 27) {
                    std::cout << "Interrupted by user" << std::endl;
                    break;
                }
            }

            frameCount++;
        }

        logger->flush();
        uint64_t totalMs = nowMs() - startTime;

        std::cout << "\n=== MODE 7c Complete ===" << std::endl;
        std::cout << "Frames processed : " << frameCount << std::endl;
        std::cout << "Total time       : " << totalMs / 1000.0 << " s" << std::endl;
        std::cout << "Average FPS      : " << (frameCount * 1000.0 / totalMs) << std::endl;
        std::cout << "Output           : " << outputPath << std::endl;

        if (showVisualization) cv::destroyAllWindows();
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}
