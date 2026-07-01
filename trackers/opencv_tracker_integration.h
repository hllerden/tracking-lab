#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <string>

enum class CVTrackerType {
    CSRT,   // CPU — HOG + spatial reliability, en doğru ama yavaş
    KCF,    // CPU — hız/doğruluk dengesi
    MIL,    // CPU — Multiple Instance Learning
    VIT,    // GPU — Vision Transformer, DNN_BACKEND_CUDA ile çalışır
};

// TrackerVit için gerekli ONNX model dosyası:
//   models/object_tracking_vittrack_2023sep.onnx
// İndirme: https://github.com/opencv/opencv_zoo/tree/main/models/object_tracking_vittrack

// MODE 7: YOLO + OpenCV built-in trackers (camera)
// Keyboard: 1=CSRT  2=KCF  3=MIL  4=VIT  ESC=exit
int testOpenCVTrackers(
    int cameraIndex,
    const std::string& yoloModelPath,
    CVTrackerType trackerType = CVTrackerType::KCF,
    int detectionRefreshFrames = 30,
    bool runOnGPU = true,
    const std::string& vitModelPath = ""
);

// MODE 7b: YOLO + OpenCV built-in trackers (video file)
int testOpenCVTrackersOnVideo(
    const std::string& videoPath,
    const std::string& yoloModelPath,
    CVTrackerType trackerType = CVTrackerType::KCF,
    int detectionRefreshFrames = 30,
    bool runOnGPU = true,
    const std::string& vitModelPath = ""
);

// MODE 7c: MOT17 tracker-only mode using OpenCV built-in trackers
// Reads pre-computed detections from det.txt. OpenCV tracker predicts
// positions on frames where a track has no matching detection.
// Outputs MOTChallenge format for TrackEval evaluation.
int testOpenCVTrackersTrackerOnly(
    const std::string& sequencePath,
    const std::string& outputPath,
    CVTrackerType trackerType = CVTrackerType::KCF,
    bool showVisualization = true,
    const std::string& vitModelPath = ""
);
