#include <iostream>
#include <vector>
#include <getopt.h>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "lowlightenhancer.h"

#include "inference.h"
#include "trackers/tracker_factory.h"
#include "i_object_tracker.h"
#include "i_visual_tracker.h"
#include "i_configurable_tracker.h"
#include "impression.h"
#include "mot/mot_sequence_reader.h"
#include "mot/mot_detection_reader.h"
#include "TrackerTelemtryLogger/MotChallengeLogger.h"
#include "trackers/opencv_tracker_integration.h"

using namespace std;
using namespace cv;

// Global trackers removed - now created using factory in each function


uint64_t timeSinceEpochMillisec() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

/**
 * @brief Process MOT17 sequence and generate tracking results in MOTChallenge format
 *
 * This function reads a MOT17 sequence frame by frame, runs detection and tracking,
 * and outputs results in MOTChallenge format for evaluation with TrackEval.
 *
 * @param sequencePath Path to MOT sequence directory (e.g., "MOT17-Data/MOT17/train/MOT17-02-DPM")
 * @param outputPath Path to output tracking results file (e.g., "output/MOT17-02-DPM.txt")
 * @param modelPath Path to YOLO ONNX model file
 * @param showVisualization Display tracking visualization window (default: true)
 * @param runOnGPU Use GPU acceleration for inference (default: true)
 */
int processMOT17Sequence(const std::string& sequencePath,
                         const std::string& outputPath,
                         const std::string& modelPath,
                         bool showVisualization = true,
                         bool runOnGPU = true,
                         YOLOOutputFormat yoloFormat = YOLOOutputFormat::AUTO) {

    std::cout << "=== Processing MOT17 Sequence ===" << std::endl;
    std::cout << "Sequence: " << sequencePath << std::endl;
    std::cout << "Output: " << outputPath << std::endl;
    std::cout << "Model: " << modelPath << std::endl;

    try {
        // Initialize sequence reader
        MotSequenceReader reader(sequencePath);
        std::cout << "Sequence: " << reader.getSequenceName() << std::endl;
        std::cout << "Total frames: " << reader.getTotalFrames() << std::endl;
        std::cout << "FPS: " << reader.getFPS() << std::endl;
        std::cout << "Frame size: " << reader.getFrameSize() << std::endl;

        // Initialize tracking system
        Impression impression;
        impression.createInfrance(modelPath, cv::Size(640, 640), "classes.txt", runOnGPU);
        impression.setYOLOFormat(yoloFormat);

        // Configure tracking settings
        ImpressionSettings settings;
        settings.usePredict = true;
        settings.predictFrameLimit = 20;
        settings.printTrajectory = true;
        settings.printCenter = true;
        impression.setSettingsParams(settings);

        // Initialize telemetry logger
        auto logger = std::make_shared<MotChallengeLogger>(outputPath, true);
        impression.setTelemetry(logger);

        // Process frames
        cv::Mat frame;
        int frameCount = 0;
        auto startTime = timeSinceEpochMillisec();
        const double frameDurationMs = reader.getFPS() > 0 ? (1000.0 / reader.getFPS()) : 0.0;

        while (reader.nextFrame(frame)) {
            auto frameStartTime = timeSinceEpochMillisec();

            // Set frame metadata for telemetry
            ITrackerTelemetry::FrameMetadata metadata;
            metadata.frameIndex = frameCount;
            metadata.sourceSize = reader.getFrameSize();
            metadata.processedSize = frame.size();
            metadata.timestamp = frameCount / reader.getFPS();
            logger->onFrameStart(metadata);

            // Run tracking
            cv::Mat outputFrame = impression.stalkImageAdvance(frame);

            auto frameEndTime = timeSinceEpochMillisec();
            auto processingTime = frameEndTime - frameStartTime;

            // Display progress
            frameCount++;
            if (frameCount % 30 == 0) {
                std::cout << "Processed " << frameCount << "/" << reader.getTotalFrames()
                         << " frames (" << processingTime << " ms/frame)" << std::endl;
            }

            // Show visualization
            if (showVisualization && !outputFrame.empty()) {
                std::string frameInfo = "Frame: " + std::to_string(frameCount) + " / " + std::to_string(reader.getTotalFrames()) + "  " + std::to_string(processingTime) + "ms";
                {
                    int bl = 0;
                    cv::Size ts = cv::getTextSize(frameInfo, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &bl);
                    cv::Mat ov = outputFrame.clone();
                    cv::rectangle(ov, cv::Rect(6, 6, ts.width + 18, ts.height + bl + 14), cv::Scalar(0, 0, 0), cv::FILLED);
                    cv::addWeighted(ov, 0.5, outputFrame, 0.5, 0, outputFrame);
                    cv::putText(outputFrame, frameInfo, cv::Point(14, ts.height + 12), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
                }
                cv::imshow("MOT17 Tracking", outputFrame);
                if (cv::waitKey(1) == 27) { // ESC to exit
                    std::cout << "Interrupted by user" << std::endl;
                    break;
                }
            }
        }

        // Finalize
        logger->flush();
        auto totalTime = timeSinceEpochMillisec() - startTime;

        std::cout << "\n=== Processing Complete ===" << std::endl;
        std::cout << "Total frames: " << frameCount << std::endl;
        std::cout << "Total time: " << totalTime / 1000.0 << " seconds" << std::endl;
        std::cout << "Average FPS: " << (frameCount * 1000.0 / totalTime) << std::endl;
        std::cout << "Output saved to: " << outputPath << std::endl;

        if (showVisualization) {
            cv::destroyAllWindows();
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error processing sequence: " << e.what() << std::endl;
        return -1;
    }
}

/**
 * @brief Process MOT17 sequence using provided detections (tracker-only mode)
 *
 * This function reads pre-computed detections from det.txt and only tests
 * the tracking algorithm. This is much faster than YOLO inference (~10x)
 * and required for official MOTChallenge submissions.
 *
 * @param sequencePath Path to MOT sequence directory
 * @param outputPath Path to output tracking results file
 * @param showVisualization Display tracking visualization window (default: true)
 * @return 0 on success, -1 on error
 */
int processMOT17SequenceTrackerOnly(const std::string& sequencePath,
                                     const std::string& outputPath,
                                     bool showVisualization = true,
                                     TrackerType trackerType = DEFAULT_TRACKER_TYPE) {

    std::cout << "=== Processing MOT17 Sequence (Tracker-Only Mode) ===" << std::endl;
    std::cout << "Sequence: " << sequencePath << std::endl;
    std::cout << "Output: " << outputPath << std::endl;
    std::cout << "Mode: Using provided detections (det.txt)" << std::endl;

    try {
        // Initialize sequence reader for frames
        MotSequenceReader reader(sequencePath);
        std::cout << "Sequence: " << reader.getSequenceName() << std::endl;
        std::cout << "Total frames: " << reader.getTotalFrames() << std::endl;
        std::cout << "FPS: " << reader.getFPS() << std::endl;
        std::cout << "Frame size: " << reader.getFrameSize() << std::endl;

        // Initialize detection reader
        std::string detFilePath = sequencePath + "/det/det.txt";
        MotDetectionReader detReader(detFilePath);

        // Show detection statistics
        auto stats = detReader.getStatistics();
        std::cout << "\nDetection Statistics:" << std::endl;
        std::cout << "  Total detections: " << stats.totalDetections << std::endl;
        std::cout << "  Avg detections/frame: " << stats.avgDetPerFrame << std::endl;
        std::cout << "  Confidence range: " << stats.minConfidence << " - " << stats.maxConfidence << std::endl;

        // Initialize tracker via factory
        auto trackerOwner = TrackerFactory::create(trackerType);
        IObjectTracker* tracker    = trackerOwner.get();
        IVisualTracker* visualTrk  = dynamic_cast<IVisualTracker*>(tracker);

        std::cout << "Tracker: " << TrackerFactory::getDescription(trackerType) << std::endl;

        // Configure tracker settings
        KalmanIoUConfig config = TrackerFactory::getDefaultConfig(trackerType);
        config.usePredictionInLost = true;
        config.maxLostFrames       = 30*2;
        config.iouThreshold       = 0.10f;  // Stage 1 gating gevşet
        config.lowConfIouThreshold = 0.05f; // Stage 2 neredeyse yalnızca appearance
        config.reidAlpha           = 0.65f;  // 65% IoU, 35% appearance
        config.featureEmaDecay     = 0.9f;
        config.highConfThreshold  = 0.55f;  // biraz daha fazla det Stage 1'e
        config.lostStateThreshold = 3;      // biraz daha toleranslı
        config.removeOutOfBounds = true;
        config.cameraBounds = cv::Rect(0, 0, reader.getFrameSize().width, reader.getFrameSize().height);
        TrackerFactory::applyConfig(tracker, config);

        // Visualization settings (applied once, reused every frame)
        IVisualTracker::VisualizationConfig visConfig;
        visConfig.showBoundingBox = true;
        visConfig.showTrackId = true;
        visConfig.showTrajectory = true;
        visConfig.showVelocity = false;
        visConfig.showMissedFrames = false;
        visConfig.showLostTracks=false;
        // Initialize telemetry logger
        auto logger = std::make_shared<MotChallengeLogger>(outputPath, true);
        TrackerFactory::applyTelemetry(tracker, logger,
                                       reader.getFrameSize(), reader.getFrameSize());

        // Process frames
        cv::Mat frame;
        int frameCount = 0;
        auto startTime = timeSinceEpochMillisec();
        const double frameDurationMs = reader.getFPS() > 0 ? (1000.0 / reader.getFPS()) : 0.0;

        while (reader.nextFrame(frame)) {
            auto frameStartTime = timeSinceEpochMillisec();
            int frameIndex = frameCount + 1;  // MOT uses 1-based indexing

            // Set frame metadata for telemetry
            ITrackerTelemetry::FrameMetadata metadata;
            metadata.frameIndex = frameCount;
            metadata.sourceSize = reader.getFrameSize();
            metadata.processedSize = frame.size();
            metadata.timestamp = frameCount / reader.getFPS();
            logger->onFrameStart(metadata);

            // Get detections for this frame from det.txt
            auto detections = detReader.getDetectionsForFrame(frameIndex);

            // BoT-SORT needs the raw frame for camera motion compensation
            if (auto* bst = dynamic_cast<BotSortTracker*>(tracker)) {
                bst->setFrame(frame);
            }

            // Run tracking (no YOLO inference!)
            auto trackedResults = tracker->update(detections);

            auto frameEndTime = timeSinceEpochMillisec();
            auto processingTime = frameEndTime - frameStartTime;

            // Visualize if enabled
            if (showVisualization && !frame.empty()) {
                if (visualTrk) visualTrk->draw(frame, visConfig);

                std::string frameInfo = "Frame: " + std::to_string(frameIndex) + " / " + std::to_string(reader.getTotalFrames()) + "  " + std::to_string(processingTime) + "ms";
                {
                    int bl = 0;
                    cv::Size ts = cv::getTextSize(frameInfo, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &bl);
                    cv::Mat ov = frame.clone();
                    cv::rectangle(ov, cv::Rect(6, 6, ts.width + 18, ts.height + bl + 14), cv::Scalar(0, 0, 0), cv::FILLED);
                    cv::addWeighted(ov, 0.5, frame, 0.5, 0, frame);
                    cv::putText(frame, frameInfo, cv::Point(14, ts.height + 12), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
                }
                cv::imshow("MOT17 Tracking (Tracker-Only)", frame);
                int waitMillis = 1;
                if (frameDurationMs > 0.0) {
                    double remaining = frameDurationMs - static_cast<double>(processingTime);
                    waitMillis = remaining > 1.0 ? static_cast<int>(remaining) : 1;
                }
                if (cv::waitKey(waitMillis) == 27) { // ESC to exit
                    std::cout << "Interrupted by user" << std::endl;
                    break;
                }
            }

            // Display progress
            frameCount++;
            if (frameCount % 30 == 0) {
                std::cout << "Processed " << frameCount << "/" << reader.getTotalFrames()
                         << " frames (" << processingTime << " ms/frame)" << std::endl;
            }
        }

        // Finalize
        logger->flush();
        auto totalTime = timeSinceEpochMillisec() - startTime;

        std::cout << "\n=== Processing Complete ===" << std::endl;
        std::cout << "Total frames: " << frameCount << std::endl;
        std::cout << "Total time: " << totalTime / 1000.0 << " seconds" << std::endl;
        std::cout << "Average FPS: " << (frameCount * 1000.0 / totalTime) << std::endl;
        std::cout << "Output saved to: " << outputPath << std::endl;

        if (showVisualization) {
            cv::destroyAllWindows();
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error processing sequence: " << e.what() << std::endl;
        return -1;
    }
}

/**
 * @brief MODE 2b: Tracker-Only + ReID appearance matching
 *
 * Same input pipeline as MODE 2 (reads pre-computed detections from det.txt,
 * no YOLO inference). Additionally crops each detected region from the raw
 * frame and extracts appearance embeddings with the ReID model. The embeddings
 * are stored in Detection::featureVector and the ByteTrack cost matrix is
 * extended to: cost = alpha*(1-IoU) + (1-alpha)*cosine_distance.
 *
 * This lets you isolate the contribution of appearance matching without the
 * variable of YOLO detection quality.
 *
 * @param sequencePath  Path to MOT17 sequence directory
 * @param outputPath    Path to output tracking results file
 * @param reidModelPath Path to ReID ONNX model (e.g. models/yolo26n-reid.onnx)
 * @param showVisualization Display tracking window (default: true)
 * @param runOnGPU      Use CUDA backend for ReID inference (default: true)
 * @param trackerType   Tracker type — must be BYTETRACK for ReID to take effect
 * @return 0 on success, -1 on error
 */
int processMOT17SequenceTrackerOnlyWithReId(
        const std::string& sequencePath,
        const std::string& outputPath,
        const std::string& reidModelPath,
        bool showVisualization = true,
        bool runOnGPU = true,
        TrackerType trackerType = DEFAULT_TRACKER_TYPE) {

    std::cout << "=== Processing MOT17 Sequence (Tracker-Only + ReID Mode) ===" << std::endl;
    std::cout << "Sequence:   " << sequencePath << std::endl;
    std::cout << "Output:     " << outputPath << std::endl;
    std::cout << "ReID model: " << reidModelPath << std::endl;
    std::cout << "Mode: det.txt detections + ReID appearance features" << std::endl;

    try {
        MotSequenceReader reader(sequencePath);
        std::cout << "Sequence:     " << reader.getSequenceName() << std::endl;
        std::cout << "Total frames: " << reader.getTotalFrames() << std::endl;
        std::cout << "FPS:          " << reader.getFPS() << std::endl;
        std::cout << "Frame size:   " << reader.getFrameSize() << std::endl;

        MotDetectionReader detReader(sequencePath + "/det/det.txt");
        auto stats = detReader.getStatistics();
        std::cout << "\nDetection Statistics:" << std::endl;
        std::cout << "  Total: " << stats.totalDetections << std::endl;
        std::cout << "  Avg/frame: " << stats.avgDetPerFrame << std::endl;

        // Build tracker
        auto trackerOwner = TrackerFactory::create(trackerType);
        IObjectTracker* tracker   = trackerOwner.get();
        IVisualTracker* visualTrk = dynamic_cast<IVisualTracker*>(tracker);

        std::cout << "Tracker: " << TrackerFactory::getDescription(trackerType) << std::endl;

        // Enable ReID in ByteTrack config
        KalmanIoUConfig config = TrackerFactory::getDefaultConfig(trackerType);
        config.usePredictionInLost = true;
        config.maxLostFrames       = 30*2;
        config.iouThreshold       = 0.10f;  // Stage 1 gating gevşet
        config.lowConfIouThreshold = 0.05f; // Stage 2 neredeyse yalnızca appearance
        config.reidAlpha           = 0.65f;  // 65% IoU, 35% appearance
        config.featureEmaDecay     = 0.9f;
        config.highConfThreshold  = 0.55f;  // biraz daha fazla det Stage 1'e
        config.lostStateThreshold = 3;      // biraz daha toleranslı

        config.removeOutOfBounds   = true;
        config.cameraBounds        = cv::Rect(0, 0,
            reader.getFrameSize().width, reader.getFrameSize().height);
        config.useReId             = true;   // hybrid IoU + cosine cost
        TrackerFactory::applyConfig(tracker, config);

        // Build ReID extractor
        ReidExtractor reidExtractor(reidModelPath, runOnGPU);

        IVisualTracker::VisualizationConfig visConfig;
        visConfig.showBoundingBox  = true;
        visConfig.showTrackId      = true;
        visConfig.showTrajectory   = true;
        visConfig.showVelocity     = true;
        visConfig.showMissedFrames = true;
        visConfig.showLostTracks   = true;
        visConfig.showReidScore    = true;
        auto logger = std::make_shared<MotChallengeLogger>(outputPath, true);
        TrackerFactory::applyTelemetry(tracker, logger,
                                       reader.getFrameSize(), reader.getFrameSize());

        cv::Mat frame;
        int frameCount  = 0;
        auto startTime  = timeSinceEpochMillisec();
        const double frameDurationMs = reader.getFPS() > 0
            ? (1000.0 / reader.getFPS()) : 0.0;

        while (reader.nextFrame(frame)) {
            auto frameStartTime = timeSinceEpochMillisec();
            int frameIndex = frameCount + 1;  // MOT17 is 1-based

            ITrackerTelemetry::FrameMetadata metadata;
            metadata.frameIndex    = frameCount;
            metadata.sourceSize    = reader.getFrameSize();
            metadata.processedSize = frame.size();
            metadata.timestamp     = frameCount / reader.getFPS();
            logger->onFrameStart(metadata);

            // Get bounding-box detections from det.txt
            auto detections = detReader.getDetectionsForFrame(frameIndex);

            // Extract ReID features for every detection crop
            std::vector<cv::Rect> bboxes;
            bboxes.reserve(detections.size());
            for (const auto& d : detections) bboxes.push_back(d.bbox);

            auto features = reidExtractor.extractBatch(frame, bboxes);
            for (size_t i = 0; i < detections.size() && i < features.size(); ++i)
                detections[i].featureVector = features[i];

            // BoT-SORT needs the raw frame for camera motion compensation
            if (auto* bst = dynamic_cast<BotSortTracker*>(tracker)) {
                bst->setFrame(frame);
            }

            // Run tracking with appearance-enriched detections
            tracker->update(detections);

            auto processingTime = timeSinceEpochMillisec() - frameStartTime;

            if (showVisualization && !frame.empty()) {
                if (visualTrk) visualTrk->draw(frame, visConfig);

                std::string frameInfo = "Frame: " + std::to_string(frameIndex) + " / " + std::to_string(reader.getTotalFrames()) + "  " + std::to_string(processingTime) + "ms";
                {
                    int bl = 0;
                    cv::Size ts = cv::getTextSize(frameInfo, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &bl);
                    cv::Mat ov = frame.clone();
                    cv::rectangle(ov, cv::Rect(6, 6, ts.width + 18, ts.height + bl + 14), cv::Scalar(0, 0, 0), cv::FILLED);
                    cv::addWeighted(ov, 0.5, frame, 0.5, 0, frame);
                    cv::putText(frame, frameInfo, cv::Point(14, ts.height + 12), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
                }
                cv::imshow("MOT17 Tracking (Tracker-Only + ReID)", frame);
                int waitMs = 1;
                if (frameDurationMs > 0.0) {
                    double rem = frameDurationMs - static_cast<double>(processingTime);
                    waitMs = rem > 1.0 ? static_cast<int>(rem) : 1;
                }
                if (cv::waitKey(waitMs) == 27) {
                    std::cout << "Interrupted by user" << std::endl;
                    break;
                }
            }

            frameCount++;
            if (frameCount % 30 == 0) {
                std::cout << "Processed " << frameCount << "/" << reader.getTotalFrames()
                          << " frames (" << processingTime << " ms/frame)" << std::endl;
            }
        }

        logger->flush();
        auto totalTime = timeSinceEpochMillisec() - startTime;

        std::cout << "\n=== Processing Complete ===" << std::endl;
        std::cout << "Total frames: " << frameCount << std::endl;
        std::cout << "Total time:   " << totalTime / 1000.0 << " s" << std::endl;
        std::cout << "Average FPS:  " << (frameCount * 1000.0 / totalTime) << std::endl;
        std::cout << "Output:       " << outputPath << std::endl;

        if (showVisualization) cv::destroyAllWindows();
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}

/**
 * @brief MODE 5: Process video file with YOLO inference and tracking
 *
 * This function processes a video file frame-by-frame with YOLO detection
 * and tracking, generating MOTChallenge format output for analysis.
 *
 * @param videoPath Path to video file (e.g., "/path/to/video.mp4")
 * @param outputPath Path to output tracking results file
 * @param modelPath Path to YOLO ONNX model file
 * @param showVisualization Display tracking visualization window (default: true)
 * @param runOnGPU Use GPU acceleration for inference (default: true)
 * @return 0 on success, -1 on error
 */
int processVideoFile(const std::string& videoPath,
                    const std::string& outputPath,
                    const std::string& modelPath,
                    bool showVisualization = true,
                    bool runOnGPU = true,
                    YOLOOutputFormat yoloFormat = YOLOOutputFormat::AUTO) {

    // ============= USER CONFIGURABLE PARAMETERS =============
    // Frame skip rate: Process every Nth frame
    // frameSkipRate = 1  -> Process every frame (no skip)
    // frameSkipRate = 2  -> Process every 2nd frame (2x faster)
    // frameSkipRate = 3  -> Process every 3rd frame (3x faster)
    // frameSkipRate = 5  -> Process every 5th frame (5x faster)
    constexpr int frameSkipRate = 1;
    constexpr bool enforceRealtimePlayback = true;
    // ========================================================

    std::cout << "=== Processing Video File ===" << std::endl;
    std::cout << "Video: " << videoPath << std::endl;
    std::cout << "Output: " << outputPath << std::endl;
    std::cout << "Model: " << modelPath << std::endl;
    std::cout << "Frame Skip Rate: " << frameSkipRate << " (processing 1/" << frameSkipRate << " frames)" << std::endl;

    try {
        // 1. Open video file
        cv::VideoCapture video(videoPath);
        if (!video.isOpened()) {
            std::cerr << "Error: Cannot open video file: " << videoPath << std::endl;
            return -1;
        }

        // 2. Get video properties
        int totalFrames = static_cast<int>(video.get(cv::CAP_PROP_FRAME_COUNT));
        double fps = video.get(cv::CAP_PROP_FPS);
        cv::Size frameSize(
            static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH)),
            static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT))
        );

        std::cout << "Total frames: " << totalFrames << std::endl;
        std::cout << "Effective frames to process: " << (totalFrames / frameSkipRate) << std::endl;
        std::cout << "FPS: " << fps << std::endl;
        std::cout << "Frame size: " << frameSize << std::endl;

        const int realtimeDelayMs = (fps > 0.0)
            ? std::max(1, static_cast<int>(std::lround(1000.0 / fps)))
            : 1;

        // 3. Initialize tracking system
        Impression impression;
        impression.createInfrance(modelPath, cv::Size(640, 640), "classes.txt", runOnGPU);
        impression.setYOLOFormat(yoloFormat);

        // 4. Configure tracking settings (user can modify these)
        ImpressionSettings settings;
        settings.usePredict = true;
        settings.predictFrameLimit = 20;
        settings.printTrajectory = true;
        settings.printCenter = true;
        impression.setSettingsParams(settings);

        // 5. Initialize telemetry logger
        auto logger = std::make_shared<MotChallengeLogger>(outputPath, true);
        impression.setTelemetry(logger);

        // 6. Process frames
        cv::Mat frame;
        int frameCount = 0;
        int processedFrameCount = 0;
        auto startTime = timeSinceEpochMillisec();

        while (video.read(frame)) {
            if (frame.empty()) break;

            frameCount++;

            // Skip frames based on frameSkipRate
            if (frameCount % frameSkipRate != 0) {
                continue;  // Skip this frame
            }

            auto frameStartTime = timeSinceEpochMillisec();

            // Set frame metadata
            ITrackerTelemetry::FrameMetadata metadata;
            metadata.frameIndex = frameCount - 1;  // 0-based index
            metadata.sourceSize = frameSize;
            metadata.processedSize = frame.size();
            metadata.timestamp = (frameCount - 1) / fps;
            logger->onFrameStart(metadata);

            // Run tracking
            cv::Mat outputFrame = impression.stalkImageAdvance(frame);

            auto frameEndTime = timeSinceEpochMillisec();
            auto processingTime = frameEndTime - frameStartTime;

            processedFrameCount++;

            // Progress display
            if (processedFrameCount % 30 == 0 || processedFrameCount == 1) {
                std::cout << "Frame " << frameCount << "/" << totalFrames
                         << " (Processed: " << processedFrameCount << ")"
                         << " (" << processingTime << "ms, "
                         << (1000.0 / processingTime) << " FPS)" << std::endl;
            }

            // Visualization
            if (showVisualization) {
                // Add info text to frame
                std::string infoText = "Frame: " + std::to_string(frameCount) +
                                      "/" + std::to_string(totalFrames) +
                                      " (Skip: " + std::to_string(frameSkipRate) + ")";
                std::string processedInfo = "Processed: " + std::to_string(processedFrameCount);
                {
                    int bl = 0;
                    cv::Size ts1 = cv::getTextSize(infoText,      cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &bl);
                    cv::Size ts2 = cv::getTextSize(processedInfo, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &bl);
                    int rectW = std::max(ts1.width, ts2.width) + 18;
                    int rectH = ts1.height + ts2.height + bl + 24;
                    cv::Mat ov = outputFrame.clone();
                    cv::rectangle(ov, cv::Rect(6, 6, rectW, rectH), cv::Scalar(0, 0, 0), cv::FILLED);
                    cv::addWeighted(ov, 0.5, outputFrame, 0.5, 0, outputFrame);
                    cv::putText(outputFrame, infoText,      cv::Point(14, ts1.height + 10),              cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
                    cv::putText(outputFrame, processedInfo, cv::Point(14, ts1.height + ts2.height + 18), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
                }

                cv::imshow("Video Tracking - MODE 5", outputFrame);
                int waitMs = enforceRealtimePlayback ? realtimeDelayMs : 1;
                if (cv::waitKey(waitMs) == 27) {  // ESC to exit
                    std::cout << "User interrupted processing" << std::endl;
                    break;
                }
            }

            logger->onFrameEnd(frameCount - 1);
        }

        auto totalTime = timeSinceEpochMillisec() - startTime;
        double avgFps = (processedFrameCount * 1000.0) / totalTime;

        std::cout << "\n=== Video Processing Complete ===" << std::endl;
        std::cout << "Total frames in video: " << frameCount << std::endl;
        std::cout << "Frames processed: " << processedFrameCount << " (skip rate: " << frameSkipRate << ")" << std::endl;
        std::cout << "Total time: " << totalTime / 1000.0 << " seconds" << std::endl;
        std::cout << "Average processing FPS: " << avgFps << std::endl;
        std::cout << "Time saved: " << ((1.0 - (1.0/frameSkipRate)) * 100.0) << "%" << std::endl;
        std::cout << "Output saved to: " << outputPath << std::endl;

        video.release();
        if (showVisualization) {
            cv::destroyAllWindows();
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        if (showVisualization) {
            cv::destroyAllWindows();
        }
        return -1;
    }
}

/**
 * @brief MODE 6: Process camera stream with YOLO inference and tracking
 *
 * This function processes a live camera stream with YOLO detection and tracking,
 * generating MOTChallenge format output. Runs until ESC key is pressed.
 *
 * @param cameraIndex Camera device index (0, 1, 2, ...) or RTSP URL
 * @param outputPath Path to output tracking results file
 * @param modelPath Path to YOLO ONNX model file
 * @param showVisualization Display tracking visualization window (default: true)
 * @param runOnGPU Use GPU acceleration for inference (default: true)
 * @return 0 on success, -1 on error
 */
int processCameraStream(int cameraIndex,
                       const std::string& outputPath,
                       const std::string& modelPath,
                       bool showVisualization = true,
                       bool runOnGPU = true,
                       YOLOOutputFormat yoloFormat = YOLOOutputFormat::AUTO) {

    std::cout << "=== Processing Camera Stream ===" << std::endl;
    std::cout << "Camera Index: " << cameraIndex << std::endl;
    std::cout << "Output: " << outputPath << std::endl;
    std::cout << "Model: " << modelPath << std::endl;

    try {
        // 1. Open camera
        cv::VideoCapture camera(cameraIndex);
        if (!camera.isOpened()) {
            std::cerr << "Error: Cannot open camera: " << cameraIndex << std::endl;
            return -1;
        }

        // 2. Set camera properties
        camera.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        camera.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        camera.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        camera.set(cv::CAP_PROP_FPS, 30);

        double width = camera.get(cv::CAP_PROP_FRAME_WIDTH);
        double height = camera.get(cv::CAP_PROP_FRAME_HEIGHT);
        double fps = camera.get(cv::CAP_PROP_FPS);

        std::cout << "Camera resolution: " << width << "x" << height << std::endl;
        std::cout << "Camera FPS: " << fps << std::endl;

        // 3. Initialize tracking system
        Impression impression;
        impression.createInfrance(modelPath, cv::Size(640, 640), "classes.txt", runOnGPU);
        impression.setYOLOFormat(yoloFormat);

        // 4. Configure tracking settings (user can modify these)
        ImpressionSettings settings;
        settings.usePredict = true;
        settings.predictFrameLimit = 20;
        settings.printTrajectory = true;
        settings.printCenter = true;
        impression.setSettingsParams(settings);

        // 5. Initialize telemetry logger
        auto logger = std::make_shared<MotChallengeLogger>(outputPath, true);
        impression.setTelemetry(logger);

        // 6. Process camera stream
        cv::Mat frame;
        int frameCount = 0;
        auto lastFpsTime = timeSinceEpochMillisec();
        int fpsFrameCount = 0;
        double currentFps = 0.0;

        std::cout << "\nCamera stream started. Press ESC to exit..." << std::endl;

        while (true) {
            auto frameStartTime = timeSinceEpochMillisec();

            // Read frame
            if (!camera.read(frame) || frame.empty()) {
                std::cerr << "Failed to read frame from camera" << std::endl;
                break;
            }

            // Set frame metadata
            ITrackerTelemetry::FrameMetadata metadata;
            metadata.frameIndex = frameCount;
            metadata.sourceSize = cv::Size(width, height);
            metadata.processedSize = frame.size();
            metadata.timestamp = frameCount / fps;
            logger->onFrameStart(metadata);

            // Run tracking
            cv::Mat outputFrame = impression.stalkImageAdvance(frame);

            auto frameEndTime = timeSinceEpochMillisec();
            auto processingTime = frameEndTime - frameStartTime;

            frameCount++;
            fpsFrameCount++;

            // Calculate FPS every second
            auto currentTime = timeSinceEpochMillisec();
            if (currentTime - lastFpsTime >= 1000) {
                currentFps = (fpsFrameCount * 1000.0) / (currentTime - lastFpsTime);
                fpsFrameCount = 0;
                lastFpsTime = currentTime;

                std::cout << "Frames: " << frameCount
                         << ", FPS: " << std::fixed << std::setprecision(1) << currentFps
                         << ", Processing: " << processingTime << "ms" << std::endl;
            }

            // Visualization
            if (showVisualization) {
                // Add info overlay
                std::string fpsText   = "FPS: " + std::to_string(static_cast<int>(currentFps));
                std::string frameText = "Frame: " + std::to_string(frameCount);
                std::string escText   = "ESC: exit";
                {
                    int bl = 0;
                    cv::Size ts1 = cv::getTextSize(fpsText,   cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &bl);
                    cv::Size ts2 = cv::getTextSize(frameText, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &bl);
                    cv::Size ts3 = cv::getTextSize(escText,   cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &bl);
                    int rectW = std::max({ts1.width, ts2.width, ts3.width}) + 18;
                    int rectH = ts1.height + ts2.height + ts3.height + bl + 34;
                    cv::Mat ov = outputFrame.clone();
                    cv::rectangle(ov, cv::Rect(6, 6, rectW, rectH), cv::Scalar(0, 0, 0), cv::FILLED);
                    cv::addWeighted(ov, 0.5, outputFrame, 0.5, 0, outputFrame);
                    cv::putText(outputFrame, fpsText,   cv::Point(14, ts1.height + 10),                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
                    cv::putText(outputFrame, frameText, cv::Point(14, ts1.height + ts2.height + 18),           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
                    cv::putText(outputFrame, escText,   cv::Point(14, ts1.height + ts2.height + ts3.height + 26), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
                }

                cv::imshow("Camera Tracking - MODE 6", outputFrame);
                if (cv::waitKey(1) == 27) {  // ESC to exit
                    std::cout << "User stopped camera stream" << std::endl;
                    break;
                }
            }

            logger->onFrameEnd(frameCount - 1);
        }

        std::cout << "\n=== Camera Stream Processing Complete ===" << std::endl;
        std::cout << "Total frames processed: " << frameCount << std::endl;
        std::cout << "Output saved to: " << outputPath << std::endl;

        camera.release();
        if (showVisualization) {
            cv::destroyAllWindows();
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        if (showVisualization) {
            cv::destroyAllWindows();
        }
        return -1;
    }
}


int  readOnImage(){

    std::string projectBasePath = PROJECT_ROOT_DIR;

    bool runOnGPU = true;

    //
    // Pass in either:
    //
    // "yolov8s.onnx" or "yolov5s.onnx"
    //
    // To run Inference with yolov8/yolov5 (ONNX)
    //

    // Note that in this example the classes are hard-coded and 'classes.txt' is a place holder.
    Inference inf(projectBasePath + "/models/best.onnx", cv::Size(640, 640), "classes.txt", runOnGPU);
    // Explicit format: YOLOv8/v11/v12 için YOLOV11, YOLOv10 NMS-free için YOLOV10, varsayılan AUTO
    // inf.setYOLOFormat(YOLOOutputFormat::YOLOV11);
    // inf.setOpenCVLayout(OpenCVDNNTensorLayout::LAYOUT_2D); // OpenCV 5.x'te #if ile otomatik

    std::vector<std::string> imageNames;
    imageNames.push_back(projectBasePath + "/videoplayback.png");
    // imageNames.push_back(projectBasePath + "/ultralytics/assets/zidane.jpg");

    for (int i = 0; i < imageNames.size(); ++i)
    {
        cv::Mat frame = cv::imread(imageNames[i]);

        // Inference starts here...
        std::vector<Detection> output = inf.runInference(frame);

        int detections = output.size();
        std::cout << "Number of detections:" << detections << std::endl;

        for (int i = 0; i < detections; ++i)
        {
            Detection detection = output[i];

            cv::Rect box = detection.box;
            cv::Scalar color = detection.color;

            // Detection box
            cv::rectangle(frame, box, color, 2);

            // Detection box text
            std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

            cv::rectangle(frame, textBox, color, cv::FILLED);
            cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        }
        // Inference ends here...

        // This is only for preview purposes
        float scale = 0.8;
        cv::resize(frame, frame, cv::Size(frame.cols*scale, frame.rows*scale));
        cv::imshow("Inference", frame);

        cv::waitKey(-1);
    }
    return 0;
}



int main(int argc, char **argv)
{

    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;
    std::string projectBasePath = PROJECT_ROOT_DIR;

    bool runOnGPU = true;

    // ==================== MOT17 Sequence Processing Examples ====================
    // Uncomment the following lines to process MOT17 sequences instead of video
    //
    // MODE 1: YOLO + Tracking (End-to-End) - Tests complete system
    // return processMOT17Sequence(
    //     projectBasePath + "/MOT17-Data/MOT17/train/MOT17-04-FRCNN",
    //     projectBasePath + "/output/MOT17-04-FRCNN-YOLO.txt",
    //     projectBasePath + "/models/yolo26l.onnx",
    //     true,                         // showVisualization
    //     runOnGPU,
    //     YOLOOutputFormat::YOLOV26     // NMS-free köşe koordinat formatı
    // );

    // MODE 2: Tracker-Only (det.txt) - Tests only tracking algorithm (~10x faster!)
    // processMOT17SequenceTrackerOnly(
    //     projectBasePath + "/MOT17-Data/MOT17/train/MOT17-13-FRCNN",
    //     projectBasePath + "/output/MOT17-13-FRCNN-tracker.txt",
    //     true,  // showVisualization
    //     TrackerType::BOT_SORT
    // );

    // MODE 2b: Tracker-Only + ReID appearance matching
    // Same as MODE 2 but each detection crop is passed through the ReID model.
    // Cost matrix: alpha*(1-IoU) + (1-alpha)*cosine_distance  (alpha=0.65)
    // Compare output with MODE 2 via compare_versions.sh to measure ReID impact.
    return processMOT17SequenceTrackerOnlyWithReId(
        projectBasePath + "/MOT17-Data/MOT17/train/MOT17-04-FRCNN",
        projectBasePath + "/output/MOT17-04-FRCNN-reid-tracker.txt",
        // ORT backend: use original model (dynamic batch supported)
        // OpenCV DNN backend: use yolo26n-reid-sim.onnx (onnxsim, batch=1)
#ifdef USE_ORT_REID
        projectBasePath + "/models/yolo26n-reid.onnx",
#else
        projectBasePath + "/models/yolo26n-reid-sim.onnx",
#endif
        true,       // showVisualization
        runOnGPU,    // useGPU for ReID inference
        TrackerType::BYTETRACK
    );

    // MODE 3: Compare different detectors (DPM, FRCNN, SDP) - Tracker performance test
    // std::vector<std::string> detectors = {"DPM", "FRCNN", "SDP"};
    // for (const auto& det : detectors) {
    //     std::string seqPath = projectBasePath + "/MOT17-Data/MOT17/train/MOT17-02-" + det;
    //     std::string outPath = projectBasePath + "/output/MOT17-02-" + det + "-tracker.txt";
    //     processMOT17SequenceTrackerOnly(seqPath, outPath, false);
    // }
    // return 0;

    // MODE 4: Batch processing - All train sequences
    // std::vector<std::string> sequences = {
    //     "MOT17-02-DPM", "MOT17-04-DPM", "MOT17-05-DPM",
    //     "MOT17-09-DPM", "MOT17-10-DPM", "MOT17-11-DPM", "MOT17-13-DPM"
    // };
    // for (const auto& seqName : sequences) {
    //     std::string seqPath = projectBasePath + "/MOT17-Data/MOT17/train/" + seqName;
    //     std::string outPath = projectBasePath + "/output/" + seqName + "-tracker.txt";
    //     processMOT17SequenceTrackerOnly(seqPath, outPath, false);
    // }
    // return 0;

    // MODE 5: Video File Processing - Process any video file with tracking
    // return processVideoFile(
    //     "/path/to/your/video.mp4",
    //     projectBasePath + "/output/video_tracking.txt",
    //     projectBasePath + "/models/yolo26s.onnx",
    //     true,      // showVisualization
    //     runOnGPU,   // useGPU
    //     YOLOOutputFormat::AUTO
    // );

    // MODE 6: Camera Stream Processing - Process live camera feed with tracking
    // return processCameraStream(
    //     0,  // cameraIndex (0 = default camera, or use RTSP URL string)
    //     projectBasePath + "/output/camera_tracking.txt",
    //     projectBasePath + "/models/yolo26s.onnx",
    //     true,      // showVisualization
    //     runOnGPU   // useGPU
    // );

    // VIT model indirme:
    //   git clone --filter=blob:none --sparse https://github.com/opencv/opencv_zoo.git /tmp/ocvzoo
    //   cd /tmp/ocvzoo && git lfs install
    //   git sparse-checkout set models/object_tracking_vittrack && git lfs pull
    //   cp /tmp/ocvzoo/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx \
    //      <proje>/models/vittrack.onnx
    const std::string vitModel = projectBasePath + "/models/vittrack.onnx";

    // MODE 7: OpenCV built-in trackers (CSRT / KCF / MIL / VIT) + YOLO re-detection
    // Keys during runtime: 1=CSRT  2=KCF  3=MIL  4=VIT(GPU)  ESC=exit
    // return testOpenCVTrackers(
    //     0,                                          // cameraIndex
    //     projectBasePath + "/models/yolo26s.onnx",  // YOLO model
    //     CVTrackerType::KCF,                         // başlangıç tracker
    //     30,                                         // YOLO her N frame'de çalışır
    //     runOnGPU,
    //     vitModel                                    // VIT için (4 tuşuyla geçiş)
    // );

    // MODE 7b: Aynısı video dosyasında
    // return testOpenCVTrackersOnVideo(
    //     "/path/to/your/video.mp4",
    //     projectBasePath + "/models/yolo26s.onnx",
    //     CVTrackerType::KCF,
    //     30,
    //     runOnGPU,
    //     vitModel
    // );

    // MODE 7c: MOT17 tracker-only — KalmanIoUTracker yerine OpenCV tracker
    // det.txt'den detection okur; eşleşme bulunamayan frame'lerde tracker tahmin yapar.
    // Çıktı TrackEval ile değerlendirilebilir (MODE 2 ile aynı format).
    // return testOpenCVTrackersTrackerOnly(
    //     projectBasePath + "/MOT17-Data/MOT17/train/MOT17-04-FRCNN",
    //     projectBasePath + "/output/MOT17-04-FRCNN-cvtracker.txt",
    //     CVTrackerType::CSRT,   // CSRT / KCF / MIL / VIT
    //     true,                  // showVisualization
    //     vitModel               // VIT dışı tracker için görmezden gelinir
    // );



    // ============================================================================
 return 0;
    //
    // Pass in either:
    //
    // "yolov8s.onnx" or "yolov5s.onnx"
    //
    // To run Inference with yolov8/yolov5 (ONNX)
    //



    // // Note that in this example the classes are hard-coded and 'classes.txt' is a place holder.
    // // Inference inf(projectBasePath + "/models/best.onnx", cv::Size(640, 640), "classes.txt", runOnGPU);

    // Mat videoFrame;
    // //Front_View
    // //newyork
    // //walking_360
    // //highway
    // //thermal_1
    // //MOT_2.mp4
    // VideoCapture camera("/path/to/your/video.mp4");
    // // VideoCapture camera(0, CAP_V4L2);
    // // VideoCapture camera("rtsp://172.28.0.164:554/substream");
    // if (!camera.isOpened()) {
    //     cerr << "Error: Unable to access the camera" << endl;
    //     return -1;
    // }
    // camera.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
    // // Çözünürlüğü 1280x720 olarak ayarla

    // camera.set(CAP_PROP_FRAME_WIDTH, 640);
    // camera.set(CAP_PROP_FRAME_HEIGHT, 480);

    // // Ayarların başarılı olup olmadığını kontrol et
    // double width = camera.get(CAP_PROP_FRAME_WIDTH);
    // double height = camera.get(CAP_PROP_FRAME_HEIGHT);

    // cout << "Camera resolution set to: " << width << "x" << height << endl;

    // int lostTarget = 0 ;

    //  if (camera.set(CAP_PROP_FPS, 10.0))
    // {
    //     std::cout << "camera setted 20 fps" << std::endl;
    // }
    // auto lapse = timeSinceEpochMillisec();
    // // LowLightEnhancer enhancer;

    // Impression impression;
    // impression.createInfrance(projectBasePath + "/models/yolov9s.onnx", cv::Size(640, 640), "classes.txt", runOnGPU);



    // while (true) {
    //     // cv::imshow("videoFrame", videoFrame);


    //     camera.read(videoFrame);
    //     // bool success =camera.read(videoFrame);
    //     // if (!success) {
    //     //     camera.set(cv::CAP_PROP_POS_FRAMES, 0); // Videoyu başa döndür
    //     //     continue; // Döngüyü yeniden başlat
    //     // }
    //     auto start = timeSinceEpochMillisec();
    //     // cv::Mat frame; //

    //     std::cout << "camera width:" << videoFrame.rows << "height:" << videoFrame.cols << std::endl;



    //     impression.stalkImageAdvance(videoFrame);

    //     std::cout << timeSinceEpochMillisec() - start << " ms işlem süresi" << std::endl;

    //     // Çıkış kontrolü (ESC ile çıkış)
    //     if (cv::waitKey(1) == 27) {
    //         break;
    //     }
    //     if (!camera.read(videoFrame) || videoFrame.empty()) {
    //         std::cout << "Video bitti, cikiliyor..." << std::endl;
    //          break;
    //     }

    // }


    // camera.release();
    // destroyAllWindows();
    return 0;
}
