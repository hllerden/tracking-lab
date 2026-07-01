#include "impression.h"
#include "TrackerTelemtryLogger/MotChallengeLogger.h"

using cv::Mat;

Impression::Impression() {
    // Create tracker using factory with default tracker type
    auto trackerPtr = TrackerFactory::create(DEFAULT_TRACKER_TYPE);

    // Store as raw pointer (IObjectTracker interface)
    tracker = trackerPtr.release();

    // Get IVisualTracker interface (all our trackers implement it)
    visualTracker = dynamic_cast<IVisualTracker*>(tracker);

    // Set telemetry
    telemetry = std::make_shared<MotChallengeLogger>("telemetry/mot_output.csv");

    // Try to set telemetry for each tracker type
    if (auto* kt = dynamic_cast<KalmanIoUTracker6State*>(tracker)) {
        kt->setTelemetry(telemetry);
    } else if (auto* kt = dynamic_cast<KalmanIoUTracker*>(tracker)) {
        kt->setTelemetry(telemetry);
    } else if (auto* bt = dynamic_cast<KalmanIoUByteTrack*>(tracker)) {
        bt->setTelemetry(telemetry);
    }
}

Impression::~Impression()
{
    delete tracker; // Only delete once since visualTracker points to same object
    delete inference;
}

void Impression::init()
{
  // initiliase

}

void Impression::createInfrance(const std::string &onnxModelPath, const cv::Size &modelInputShape, const std::string &classesTxtFile, const bool &runWithCuda )
{
    if (inference==nullptr){
        inference = new Inference(onnxModelPath ,modelInputShape, classesTxtFile, runWithCuda);
    }
}

void Impression::setYOLOFormat(YOLOOutputFormat format)
{
    if (inference) inference->setYOLOFormat(format);
}

void Impression::setOpenCVLayout(OpenCVDNNTensorLayout layout)
{
    if (inference) inference->setOpenCVLayout(layout);
}

ImpressionSettings Impression::getSettingsParams() const
{
    return settingsParams;
}

void Impression::setSettingsParams(const ImpressionSettings &newSettingsParams)
{
    const bool reidChanged = (settingsParams.useReId      != newSettingsParams.useReId ||
                              settingsParams.reidModelPath != newSettingsParams.reidModelPath ||
                              settingsParams.reidUseGPU   != newSettingsParams.reidUseGPU);

    settingsParams = newSettingsParams;

    if (reidChanged) {
        reidExtractor_.reset();
        if (settingsParams.useReId && !settingsParams.reidModelPath.empty()) {
            reidExtractor_ = std::make_unique<ReidExtractor>(
                settingsParams.reidModelPath, settingsParams.reidUseGPU);
            // Propagate useReId flag to ByteTrack config
            if (auto* bt = dynamic_cast<KalmanIoUByteTrack*>(tracker)) {
                KalmanIoUConfig cfg = bt->getConfig();
                cfg.useReId = true;
                bt->setConfig(cfg);
            }
        }
    }
}

void Impression::setTelemetryScalingEnabled(bool enabled)
{
    telemetryScalingEnabled = enabled;
}

void Impression::setTelemetry(const std::shared_ptr<ITrackerTelemetry>& telemetryLogger)
{
    telemetry = telemetryLogger;

    // Try to set telemetry for each tracker type
    if (auto* kt = dynamic_cast<KalmanIoUTracker6State*>(tracker)) {
        kt->setTelemetry(telemetry);
    } else if (auto* kt = dynamic_cast<KalmanIoUTracker*>(tracker)) {
        kt->setTelemetry(telemetry);
    } else if (auto* bt = dynamic_cast<KalmanIoUByteTrack*>(tracker)) {
        bt->setTelemetry(telemetry);
    }
}


/**
 * @brief Impression::stalkImageDefault
 * @detail standart tracking and imshow with stalker class
 * @param inputFrame is input Frame
 */
Mat Impression::stalkImageDefault(const cv::Mat inputFrame) {
    if (inputFrame.empty()) {
        return {};
    }

    // frame alınır.
    cv::Mat frame;
    // boyutu modele göre ayarlanır
    cv::Size inputSize(640, 640);
    // resize edilir

     cv::Size rawSize(inputFrame.cols, inputFrame.rows);
     cv::resize(inputFrame, frame, inputSize);
    // modele sokulur.
    std::vector<Detection> output = inference->runInference(frame);
    // model çıktıları verir.
 cv::imshow("model Input", frame);
    // kaç adet detect edilmiş.
    int detections = output.size();

    // her detect için box oluşturulur.
    // detectedBoxes sayısı kadar girdi oluşturulur.
    std::vector<cv::Rect> detectedBoxes;
    for (int i = 0; i < detections; ++i) {
        detectedBoxes.push_back(output[i].box);

        // if (output[i].class_id == 2 /* 2 // car */) { // Sadece class_id == 2 olanları alıyoruz
        //     detectedBoxes.push_back(output[i].box);
        //     // std::cout << "basildi : " << std::endl;
        // }
    }

    // Set frame size hint for all tracker types
    if (auto* kt6 = dynamic_cast<KalmanIoUTracker6State*>(tracker)) {
        if (telemetryScalingEnabled) {
            kt6->setFrameSizeHint(rawSize, inputSize);
        } else {
            kt6->clearFrameSizeHint();
        }
    } else if (auto* kt = dynamic_cast<KalmanIoUTracker*>(tracker)) {
        if (telemetryScalingEnabled) {
            kt->setFrameSizeHint(rawSize, inputSize);
        } else {
            kt->clearFrameSizeHint();
        }
    } else if (auto* bt = dynamic_cast<KalmanIoUByteTrack*>(tracker)) {
        if (telemetryScalingEnabled) {
            bt->setFrameSizeHint(rawSize, inputSize);
        } else {
            bt->clearFrameSizeHint();
        }
    }

    std::vector<IObjectTracker::Detection> trackerDetections;
    for (int i = 0; i < detections; ++i) {
        IObjectTracker::Detection det;
        det.bbox = output[i].box;
        det.confidence = output[i].confidence;
        det.classId = output[i].class_id;
        trackerDetections.push_back(det);
    }

    // ReID: populate featureVector for each detection before matching
    if (reidExtractor_) {
        std::vector<cv::Rect> bboxes;
        bboxes.reserve(trackerDetections.size());
        for (const auto& d : trackerDetections) bboxes.push_back(d.bbox);
        auto feats = reidExtractor_->extractBatch(frame, bboxes);
        for (size_t i = 0; i < trackerDetections.size() && i < feats.size(); ++i)
            trackerDetections[i].featureVector = feats[i];
    }

    // Update tracker with new detections
    auto trackedResults = tracker->update(trackerDetections);

    // Use IVisualTracker for visualization
    if (visualTracker) {
        IVisualTracker::VisualizationConfig vizConfig;
        vizConfig.showBoundingBox = true;
        vizConfig.showTrackId = true;
        vizConfig.showCenter = settingsParams.printCenter;
        vizConfig.showTrajectory = settingsParams.printTrajectory;
        vizConfig.trajectoryLength = 50;

        visualTracker->draw(frame, vizConfig);
    }

    // frame imshow edilir.
    float scale = 1;
    cv::resize(frame, frame, rawSize);
    // isteğe göre slale edilir.

    cv::imshow("Inference standart process", frame);
    return frame;
}

Mat Impression::stalkImageAdvance(const cv::Mat inputFrame  )
{
    if (inputFrame.empty()) {
        return {};
    }
    // Configure tracker via IConfigurableTracker interface
    auto* configurableTracker = dynamic_cast<IConfigurableTracker<KalmanIoUConfig>*>(tracker);
    if (configurableTracker) {
        KalmanIoUConfig config = configurableTracker->getConfig();
        config.usePredictionInLost = settingsParams.usePredict;
        config.maxLostFrames = settingsParams.predictFrameLimit;
        configurableTracker->setConfig(config);
    }

    // frame alınır.
    cv::Mat frame;
    // boyutu modele göre ayarlanır
    cv::Size inputSize(640, 640);
    // resize edilir

    cv::Size rawSize(inputFrame.cols, inputFrame.rows);
    cv::resize(inputFrame, frame, inputSize);
    // modele sokulur.
    std::vector<Detection> output = inference->runInference(frame);
    // model çıktıları verir.
    cv::imshow("model Input", frame);
    // kaç adet detect edilmiş.
    int detections = output.size();

    // Convert to IObjectTracker::Detection format
    auto* kalmanTracker = dynamic_cast<KalmanIoUTracker*>(tracker);
    if (kalmanTracker) {
        if (telemetryScalingEnabled) {
            kalmanTracker->setFrameSizeHint(rawSize, inputSize);
        } else {
            kalmanTracker->clearFrameSizeHint();
        }
    }

    std::vector<IObjectTracker::Detection> trackerDetections;
    for (int i = 0; i < detections; ++i) {
        IObjectTracker::Detection det;
        det.bbox = output[i].box;
        det.confidence = output[i].confidence;
        det.classId = output[i].class_id;
        trackerDetections.push_back(det);
    }

    // ReID: populate featureVector for each detection before matching
    if (reidExtractor_) {
        std::vector<cv::Rect> bboxes;
        bboxes.reserve(trackerDetections.size());
        for (const auto& d : trackerDetections) bboxes.push_back(d.bbox);
        auto feats = reidExtractor_->extractBatch(frame, bboxes);
        for (size_t i = 0; i < trackerDetections.size() && i < feats.size(); ++i)
            trackerDetections[i].featureVector = feats[i];
    }

    // Update tracker with new detections
    auto trackedResults = tracker->update(trackerDetections);

    // Use IVisualTracker for advanced visualization
    if (visualTracker) {
        IVisualTracker::VisualizationConfig vizConfig;
        vizConfig.showBoundingBox = true;
        vizConfig.showTrackId = true;
        vizConfig.showCenter = settingsParams.printCenter;
        vizConfig.showTrajectory = settingsParams.printTrajectory;
        vizConfig.trajectoryLength = 50;
        vizConfig.colorByState = true; // Show different colors for tracking/lost
        vizConfig.showMissedFrames = false;

        visualTracker->draw(frame, vizConfig);
    }

    // frame imshow edilir.
    float scale = 1;
    cv::resize(frame, frame, rawSize);
    // isteğe göre slale edilir.

    // cv::imshow("Inference standart process", frame);
    return frame;
}
