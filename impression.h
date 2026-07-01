
/**
 * @date 24.01.2025
 * @author Halil Erden
 * @class Impression
 *
 * yolo for inferance and stalker for object tracking
 *
*/


#ifndef IMPRESSION_H
#define IMPRESSION_H

#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include <getopt.h>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "inference.h"
#include "trackers/tracker_factory.h"
#include "trackers/reid_extractor.h"
#include "i_object_tracker.h"
#include "i_tracker_telemetry.h"
#include "i_visual_tracker.h"
#include "i_configurable_tracker.h"
using namespace std;
using namespace cv;

/**
 * @brief The ImpressionSettings function
 * @param usePredict (The Kalman filter will be used in any case. It is only to count Kalman as a detection if there is no detection.)
 * true -  when model cant detect tracking obj , tracker (KalmanIoUTracker) use a pediction and show users
 * false - standart usage if model cant detect obj is missing and predict is not replace
 * @param predictFrameLimit  if usePredict is true then limit of prediction replace if cant find 20 frame later tracking remove
 * @param printTrajectory this is a print trajoctery on output frame
 * @param printCenter this is a print center points on output frame
*/
struct ImpressionSettings {

    bool usePredict = true ;
    u_int8_t predictFrameLimit = 20;
    bool printTrajectory = true ;
    bool printCenter = true;

    // ReID appearance model (ByteTrack only)
    bool        useReId       = false;  ///< Enable hybrid IoU+cosine matching
    std::string reidModelPath = "";     ///< Path to ReID ONNX model
    bool        reidUseGPU    = true;   ///< Use CUDA backend for ReID inference

    // Custom assignment operator
    ImpressionSettings& operator=(const ImpressionSettings& other) {
        if (this != &other) {
            usePredict        = other.usePredict;
            predictFrameLimit = other.predictFrameLimit;
            printTrajectory   = other.printTrajectory;
            printCenter       = other.printCenter;
            useReId           = other.useReId;
            reidModelPath     = other.reidModelPath;
            reidUseGPU        = other.reidUseGPU;
        }
        return *this;
    }
    // Karşılaştırma Operatörü
    bool operator==(const ImpressionSettings& other) const {
        return usePredict        == other.usePredict &&
               predictFrameLimit == other.predictFrameLimit &&
               printTrajectory   == other.printTrajectory &&
               printCenter       == other.printCenter &&
               useReId           == other.useReId &&
               reidModelPath     == other.reidModelPath &&
               reidUseGPU        == other.reidUseGPU;
    }
};
class Impression
{

public:
    Impression();
    ~Impression();
    void init();
    Mat stalkImageDefault(const cv::Mat frame);
    // Mat stalkImagePredict(const cv::Mat inputFrame);
    Mat stalkImage(const cv::Mat inputFrame);
    Mat stalkImageAdvance(const cv::Mat inputFrame);
    void setTelemetryScalingEnabled(bool enabled);


    void createInfrance(const std::string &onnxModelPath, const cv::Size &modelInputShape = {640, 640}, const std::string &classesTxtFile = "", const bool &runWithCuda = true);

    void setYOLOFormat(YOLOOutputFormat format);
    void setOpenCVLayout(OpenCVDNNTensorLayout layout);

    void setTelemetry(const std::shared_ptr<ITrackerTelemetry>& telemetryLogger);

    ImpressionSettings getSettingsParams() const;
    void setSettingsParams(const ImpressionSettings &newSettingsParams);

private:
    IObjectTracker *tracker = nullptr; // Core tracker interface
    IVisualTracker *visualTracker = nullptr; // Optional visualization interface
    Inference *inference = nullptr; // YOLO inference
    ImpressionSettings settingsParams;
    std::shared_ptr<ITrackerTelemetry> telemetry;
    bool telemetryScalingEnabled = true;
    std::unique_ptr<ReidExtractor> reidExtractor_;
};

#endif // IMPRESSION_H
