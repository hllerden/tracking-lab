#include "kalman_iou_tracker_6state.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

// Use namespaced TrackedObject within this file
using TrackedObject = KalmanIoU6State::TrackedObject;

KalmanIoUTracker6State::KalmanIoUTracker6State() : nextId(0) {}

KalmanIoUTracker6State::KalmanIoUTracker6State(const KalmanIoUConfig& cfg)
    : nextId(0), config(cfg) {}

uint64_t KalmanIoUTracker6State::timeSinceEpochMillisec() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

float KalmanIoUTracker6State::calculateIoU(const cv::Rect& box1, const cv::Rect& box2) {
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);

    int intersectionWidth = std::max(0, x2 - x1);
    int intersectionHeight = std::max(0, y2 - y1);
    int intersectionArea = intersectionWidth * intersectionHeight;

    int unionArea = box1.area() + box2.area() - intersectionArea;
    if (unionArea <= 0) {
        throw std::runtime_error("Invalid union area: cannot divide by zero or negative.");
    }
    return static_cast<float>(intersectionArea) / unionArea;
}

cv::KalmanFilter KalmanIoUTracker6State::createKalmanFilter() {
    cv::KalmanFilter kf(6, 4, 0);
    kf.transitionMatrix = (cv::Mat_<float>(6, 6) << 1, 0, 0, 0, 1, 0,
                           0, 1, 0, 0, 0, 1,
                           0, 0, 1, 0, 0, 0,
                           0, 0, 0, 1, 0, 0,
                           0, 0, 0, 0, 1, 0,
                           0, 0, 0, 0, 0, 1);
    kf.measurementMatrix = (cv::Mat_<float>(4, 6) << 1, 0, 0, 0, 0, 0,
                            0, 1, 0, 0, 0, 0,
                            0, 0, 1, 0, 0, 0,
                            0, 0, 0, 1, 0, 0);
    kf.processNoiseCov = cv::Mat::eye(6, 6, CV_32F) * 0.01F;
    kf.measurementNoiseCov = cv::Mat::eye(4, 4, CV_32F) * 0.5F;
    kf.errorCovPost = cv::Mat::eye(6, 6, CV_32F) * 0.1F;
    return kf;
}

double KalmanIoUTracker6State::computeIoU(KalmanIoUConfig::IoUType type, const cv::Rect& boxA, const cv::Rect& boxB) {
    // Defensive programming: return 0 for invalid boxes (will not match)
    if (boxA.area() <= 0 || boxB.area() <= 0) {
        return 0.0;
    }

    // Additional safety check for negative dimensions
    if (boxA.width <= 0 || boxA.height <= 0 || boxB.width <= 0 || boxB.height <= 0) {
        return 0.0;
    }

    switch (type) {
    case KalmanIoUConfig::IoUType::IOU:
        return calculateIoU(boxA, boxB);
    case KalmanIoUConfig::IoUType::GIOU:
        return calculateGIoU(boxA, boxB);
    case KalmanIoUConfig::IoUType::DIOU:
        return calculateDIoU(boxA, boxB);
    case KalmanIoUConfig::IoUType::CIOU:
        return calculateCIoU(boxA, boxB);
    case KalmanIoUConfig::IoUType::SIOU:
        return calculateSoftIoU(boxA, boxB);
    case KalmanIoUConfig::IoUType::AIOU:
        return calculateAlphaIoU(boxA, boxB);
    default:
        return 0.0; // Unknown type -> no match
    }
}

// double KalmanIoUTracker6State::calculateGIoU(const cv::Rect& boxA, const cv::Rect& boxB) {
//     double iou = calculateIoU(boxA, boxB);

//     cv::Rect enclosingBox = boxA | boxB;
//     double enclosingArea = enclosingBox.area();
//     if (enclosingArea <= 0.0) {
//         throw std::runtime_error("Invalid enclosing box: area cannot be zero or negative.");
//     }

//     double unionArea = boxA.area() + boxB.area() - (boxA & boxB).area();
//     double giou = iou - (enclosingArea - unionArea) / enclosingArea;

//     return std::max(0.0, giou);
// }
double KalmanIoUTracker6State::calculateGIoU(const cv::Rect& boxA, const cv::Rect& boxB) {
    double iou = calculateIoU(boxA, boxB);

    // Enclosing box'u doğru şekilde hesapla
    int enclosing_x1 = std::min(boxA.x, boxB.x);
    int enclosing_y1 = std::min(boxA.y, boxB.y);
    int enclosing_x2 = std::max(boxA.x + boxA.width, boxB.x + boxB.width);
    int enclosing_y2 = std::max(boxA.y + boxA.height, boxB.y + boxB.height);

    cv::Rect enclosingBox(enclosing_x1, enclosing_y1,
                          enclosing_x2 - enclosing_x1,
                          enclosing_y2 - enclosing_y1);

    double enclosingArea = enclosingBox.area();
    if (enclosingArea <= 0.0) return iou; // Exception yerine fallback

    double unionArea = boxA.area() + boxB.area() - (boxA & boxB).area();

    // GIoU formülü: IoU - (enclosingArea - unionArea) / enclosingArea
    double giou = iou - (enclosingArea - unionArea) / enclosingArea;

    return std::max(-1.0, std::min(giou, 1.0)); // [-1, 1] aralığı
}

double KalmanIoUTracker6State::calculateDIoU(const cv::Rect& boxA, const cv::Rect& boxB) {
    double iou = calculateIoU(boxA, boxB);

    cv::Point2f centerA(boxA.x + static_cast<double>(boxA.width) / 2.0,
                        boxA.y + static_cast<double>(boxA.height) / 2.0);
    cv::Point2f centerB(boxB.x + static_cast<double>(boxB.width) / 2.0,
                        boxB.y + static_cast<double>(boxB.height) / 2.0);

    double distance = cv::norm(centerA - centerB);

    cv::Rect enclosingBox = boxA | boxB;
    double enclosingDiagonal = std::sqrt(static_cast<double>(enclosingBox.width * enclosingBox.width +
                                                             enclosingBox.height * enclosingBox.height));

    if (enclosingDiagonal == 0.0) {
        return iou;
    }

    double diou = iou - (distance * distance) / (enclosingDiagonal * enclosingDiagonal);
    return std::max(0.0, diou);
}

// double KalmanIoUTracker6State::calculateCIoU(const cv::Rect& boxA, const cv::Rect& boxB) {
//     double diou = calculateDIoU(boxA, boxB);

//     double aspectRatioA = static_cast<double>(boxA.width) / boxA.height;
//     double aspectRatioB = static_cast<double>(boxB.width) / boxB.height;

//     double v = (4.0 / (M_PI * M_PI)) * std::pow(std::atan(aspectRatioA) - std::atan(aspectRatioB), 2);
//     double alpha = (v > 1e-10) ? (v / (1 - calculateIoU(boxA, boxB) + v)) : 0.0;

//     double ciou = diou - alpha * v;
//     return ciou;
// }
double KalmanIoUTracker6State::calculateCIoU(const cv::Rect& boxA, const cv::Rect& boxB) {
    double iou = calculateIoU(boxA, boxB);
    double diou = calculateDIoU(boxA, boxB); // DIoU'yu yeniden kullan

    // Aspect ratio consistency - zero division koruması
    if (boxA.height <= 0 || boxB.height <= 0) return diou;

    double wA = static_cast<double>(boxA.width);
    double hA = static_cast<double>(boxA.height);
    double wB = static_cast<double>(boxB.width);
    double hB = static_cast<double>(boxB.height);

    double arctanA = std::atan(wA / hA); // radyan
    double arctanB = std::atan(wB / hB); // radyan

    double v = (4.0 / (M_PI * M_PI)) * std::pow(arctanA - arctanB, 2);

    // α hesaplaması düzeltildi
    double alpha = v / ((1 - iou) + v + 1e-8);

    double ciou = diou - alpha * v;
    return std::max(-1.0, std::min(ciou, 1.0));
}

double KalmanIoUTracker6State::calculateSoftIoU(const cv::Rect& boxA, const cv::Rect& boxB) {
    const double iou = std::clamp(static_cast<double>(calculateIoU(boxA, boxB)), 0.0, 1.0);
    const double slack = std::max(0.0, static_cast<double>(config.softIoUSlack));
    const double denominator = iou + slack * (1.0 - iou) + 1e-12;
    if (denominator <= 0.0) {
        return 0.0;
    }
    return std::clamp(iou / denominator, 0.0, 1.0);
}

double KalmanIoUTracker6State::calculateAlphaIoU(const cv::Rect& boxA, const cv::Rect& boxB) {
    const double iou = std::clamp(static_cast<double>(calculateIoU(boxA, boxB)), 0.0, 1.0);
    const double alpha = std::max(1e-3, static_cast<double>(config.alphaIoUExponent));
    return std::pow(iou, alpha);
}

// Legacy getter/setter methods removed - use getConfig()/setConfig() instead

void KalmanIoUTracker6State::updateTrackersWithHungarian(const std::vector<cv::Rect>& detections, const std::vector<float>& confidences) {
    std::vector<cv::Rect> predictedBoxes;
    for (auto& tracker : trackers) {
        cv::Mat prediction = tracker.kalmanFilter.predict();

        // Extract predicted values
        int pred_x = static_cast<int>(prediction.at<float>(0));
        int pred_y = static_cast<int>(prediction.at<float>(1));
        int pred_w = static_cast<int>(prediction.at<float>(2));
        int pred_h = static_cast<int>(prediction.at<float>(3));

        // With proper Kalman initialization, negative dimensions shouldn't occur
        // Only clamp coordinates to positive (boundary safety)
        if (pred_x < 0) pred_x = 0;
        if (pred_y < 0) pred_y = 0;

        predictedBoxes.emplace_back(pred_x, pred_y, pred_w, pred_h);

        tracker.velocity = cv::Point(
            static_cast<int>(prediction.at<float>(4)),
            static_cast<int>(prediction.at<float>(5)));

        tracker.lastPredictedCenter = cv::Point(
            pred_x + pred_w / 2,
            pred_y + pred_h / 2);
    }

    int numTrackers = static_cast<int>(trackers.size());
    int numDetections = static_cast<int>(detections.size());

    if (numTrackers == 0 || numDetections == 0) {
        std::vector<bool> matchedDetections(numDetections, false);
        addNewTrackers(detections, matchedDetections, confidences);
        removeLostTrackers();
        return;
    }
    // new
    std::vector<std::vector<double>> costMatrix(numTrackers, std::vector<double>(numDetections, 100.0));
    for (int i = 0; i < numTrackers; ++i) {
        for (int j = 0; j < numDetections; ++j) {
            double iou = computeIoU(config.iouType, predictedBoxes[i], detections[j]);
            if (iou > config.iouThreshold) {
                // Clamp to [0.0, 1.0] to handle GIOU/DIOU/SIOU negative values
                double cost = 1.0 - std::max(0.0, std::min(1.0, iou));
                costMatrix[i][j] = std::max(0.0, cost);  // Ensure non-negative for Hungarian
            } else {
                costMatrix[i][j] = 100.0;
            }
        }
    }
    // old
    // std::vector<std::vector<double>> costMatrix(numTrackers, std::vector<double>(numDetections, 100.0));
    // for (int i = 0; i < numTrackers; ++i) {
    //     for (int j = 0; j < numDetections; ++j) {
    //         double iou = computeIoU(config.iouType, predictedBoxes[i], detections[j]);
    //         costMatrix[i][j] = (iou > config.iouThreshold) ? 1.0 - iou : 100.0;
    //     }
    // }
    HungarianAlgorithm hungarian;
    std::vector<int> assignment;
    hungarian.Solve(costMatrix, assignment);

    std::vector<bool> matchedDetections(numDetections, false);
    for (int i = 0; i < numTrackers; ++i) {
        if (assignment[i] >= 0 && assignment[i] < numDetections) {
            if (costMatrix[i][assignment[i]] > 0.9) {
                assignment[i] = -1;
                continue;
            }

            cv::Mat measurement(4, 1, CV_32F);
            measurement.at<float>(0) = static_cast<float>(detections[assignment[i]].x);
            measurement.at<float>(1) = static_cast<float>(detections[assignment[i]].y);
            measurement.at<float>(2) = static_cast<float>(detections[assignment[i]].width);
            measurement.at<float>(3) = static_cast<float>(detections[assignment[i]].height);
            trackers[i].kalmanFilter.correct(measurement);
            trackers[i].boundingBox = detections[assignment[i]];
            trackers[i].missedFrames = 0;
            trackers[i].state = TrackedObject::TRACKING;

            // Store detection confidence
            if (!confidences.empty() && assignment[i] < static_cast<int>(confidences.size())) {
                trackers[i].detectionConfidence = confidences[assignment[i]];
            }

            // Update center and trajectory (matches original stalker.cpp)
            trackers[i].lastActualCenter = cv::Point(
                trackers[i].boundingBox.x + trackers[i].boundingBox.width / 2,
                trackers[i].boundingBox.y + trackers[i].boundingBox.height / 2);

            trackers[i].trajectory.push_back(trackers[i].lastActualCenter);
            if (trackers[i].trajectory.size() > 35) {
                trackers[i].trajectory.pop_front();
            }

            matchedDetections[assignment[i]] = true;
        } else {
            trackers[i].missedFrames++;
            if (config.usePredictionInLost && trackers[i].missedFrames <= config.maxLostFrames) {
                // Use prediction for lost tracks (matches original stalker.cpp behavior)
                cv::Mat state = trackers[i].kalmanFilter.statePost;
                trackers[i].boundingBox = cv::Rect(
                    static_cast<int>(state.at<float>(0)),
                    static_cast<int>(state.at<float>(1)),
                    static_cast<int>(state.at<float>(2)),
                    static_cast<int>(state.at<float>(3)));
            }

            if (trackers[i].missedFrames > config.lostStateThreshold) {
                trackers[i].state = TrackedObject::LOST;
            }
        }
    }

    addNewTrackers(detections, matchedDetections, confidences);
    removeLostTrackers();
}

void KalmanIoUTracker6State::updateTrackers(const std::vector<cv::Rect>& detections, std::vector<bool>& matchedDetections) {
    for (auto& tracker : trackers) {
        cv::Mat prediction = tracker.kalmanFilter.predict();

        // Extract predicted values
        int pred_x = static_cast<int>(prediction.at<float>(0));
        int pred_y = static_cast<int>(prediction.at<float>(1));
        int pred_w = static_cast<int>(prediction.at<float>(2));
        int pred_h = static_cast<int>(prediction.at<float>(3));

        // Only clamp coordinates (width/height should be valid with proper init)
        if (pred_x < 0) pred_x = 0;
        if (pred_y < 0) pred_y = 0;

        tracker.boundingBox = cv::Rect(pred_x, pred_y, pred_w, pred_h);
        tracker.velocity = cv::Point(
            static_cast<int>(prediction.at<float>(4)),
            static_cast<int>(prediction.at<float>(5)));
    }

    for (size_t i = 0; i < trackers.size(); ++i) {
        double bestIoU = 0.0;
        int bestIndex = -1;
        for (size_t j = 0; j < detections.size(); ++j) {
            double iou = computeIoU(config.iouType, trackers[i].boundingBox, detections[j]);
            if (iou > bestIoU && iou > config.iouThreshold) {
                bestIoU = iou;
                bestIndex = static_cast<int>(j);
            }
        }

        if (bestIndex != -1) {
            cv::Mat measurement(4, 1, CV_32F);
            measurement.at<float>(0) = static_cast<float>(detections[bestIndex].x);
            measurement.at<float>(1) = static_cast<float>(detections[bestIndex].y);
            measurement.at<float>(2) = static_cast<float>(detections[bestIndex].width);
            measurement.at<float>(3) = static_cast<float>(detections[bestIndex].height);
            trackers[i].kalmanFilter.correct(measurement);
            trackers[i].boundingBox = detections[bestIndex];
            trackers[i].missedFrames = 0;
            trackers[i].state = TrackedObject::TRACKING;

            // Update trajectory (matches original stalker.cpp)
            trackers[i].lastActualCenter = cv::Point(
                trackers[i].boundingBox.x + trackers[i].boundingBox.width / 2,
                trackers[i].boundingBox.y + trackers[i].boundingBox.height / 2
                );
            trackers[i].trajectory.push_back(trackers[i].lastActualCenter);
            if (trackers[i].trajectory.size() > 35) {
                trackers[i].trajectory.pop_front();
            }

            matchedDetections[bestIndex] = true;
        } else {
            trackers[i].missedFrames++;
            if (config.usePredictionInLost) {
                cv::Mat state = trackers[i].kalmanFilter.statePost;
                trackers[i].boundingBox = cv::Rect(
                    static_cast<int>(state.at<float>(0)),
                    static_cast<int>(state.at<float>(1)),
                    static_cast<int>(state.at<float>(2)),
                    static_cast<int>(state.at<float>(3)));
            }
            trackers[i].state = TrackedObject::LOST;
        }
    }
}

void KalmanIoUTracker6State::addNewTrackers(const std::vector<cv::Rect>& detections, const std::vector<bool>& matchedDetections, const std::vector<float>& confidences) {
    for (size_t i = 0; i < detections.size(); ++i) {
        if (!matchedDetections[i]) {
            TrackedObject newTracker;
            newTracker.id = nextId++;
            newTracker.boundingBox = detections[i];
            newTracker.kalmanFilter = createKalmanFilter();

            // CRITICAL FIX: Initialize Kalman filter state (was missing in refactoring)
            // Without this, filter starts with random state causing incorrect predictions
            cv::Mat initialState(6, 1, CV_32F);
            initialState.at<float>(0) = static_cast<float>(detections[i].x);
            initialState.at<float>(1) = static_cast<float>(detections[i].y);
            initialState.at<float>(2) = static_cast<float>(detections[i].width);
            initialState.at<float>(3) = static_cast<float>(detections[i].height);
            initialState.at<float>(4) = 0.0f;  // vx (initial velocity = 0)
            initialState.at<float>(5) = 0.0f;  // vy (initial velocity = 0)

            newTracker.kalmanFilter.statePost = initialState;

            // Store detection confidence
            if (!confidences.empty() && i < confidences.size()) {
                newTracker.detectionConfidence = confidences[i];
            }

            newTracker.color = generateRandomColor();
            newTracker.lastActualCenter = cv::Point(
                detections[i].x + detections[i].width / 2,
                detections[i].y + detections[i].height / 2);
            newTracker.trajectory.push_back(newTracker.lastActualCenter);

            trackers.push_back(newTracker);
        }
    }
}

void KalmanIoUTracker6State::removeLostTrackers() {
    trackers.erase(
        std::remove_if(trackers.begin(), trackers.end(),
                       [this](const TrackedObject& tracker) {
                           bool outOfBounds = config.removeOutOfBounds &&
                                              (tracker.boundingBox.x + tracker.boundingBox.width < config.cameraBounds.x ||
                                               tracker.boundingBox.y + tracker.boundingBox.height < config.cameraBounds.y ||
                                               tracker.boundingBox.x > config.cameraBounds.x + config.cameraBounds.width ||
                                               tracker.boundingBox.y > config.cameraBounds.y + config.cameraBounds.height);

                           bool exceededMissedFrames = tracker.missedFrames > config.maxLostFrames;
                           return exceededMissedFrames || outOfBounds;
                       }),
        trackers.end());
}

// ========== Legacy Methods (Deprecated) ==========

std::vector<std::pair<int, cv::Rect>> KalmanIoUTracker6State::processDetections(const std::vector<cv::Rect>& detections) {
    std::vector<bool> matchedDetections(detections.size(), false);
    updateTrackers(detections, matchedDetections);
    addNewTrackers(detections, matchedDetections);
    removeLostTrackers();

    std::vector<std::pair<int, cv::Rect>> results;
    for (const auto& tracker : trackers) {
        results.emplace_back(tracker.id, tracker.boundingBox);
    }
    return results;
}

std::vector<std::pair<int, cv::Rect>> KalmanIoUTracker6State::processDetectionsWithHungarian(const std::vector<cv::Rect>& detections) {
    updateTrackersWithHungarian(detections);

    std::vector<std::pair<int, cv::Rect>> results;
    for (const auto& tracker : trackers) {
        results.emplace_back(tracker.id, tracker.boundingBox);
    }
    return results;
}

std::vector<TrackedObject> KalmanIoUTracker6State::processDetectionsWithHungarianTrackers(const std::vector<cv::Rect>& detections) {
    updateTrackersWithHungarian(detections);
    return trackers;
}

cv::Scalar KalmanIoUTracker6State::generateRandomColor() {
    cv::RNG rng(static_cast<uint64_t>(timeSinceEpochMillisec()));
    return cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
}

// ========== IObjectTracker Implementation ==========

std::vector<IObjectTracker::TrackedResult> KalmanIoUTracker6State::update(const std::vector<Detection>& detections) {
    if (telemetry) {
        ITrackerTelemetry::FrameMetadata metadata;
        metadata.frameIndex = frameIndexCounter;
        if (frameSizeHint.has_value()) {
            metadata.sourceSize = frameSizeHint->source;
            metadata.processedSize = frameSizeHint->processed;
        }
        telemetry->onFrameStart(metadata);
    }

    // Convert IObjectTracker::Detection to cv::Rect for internal processing
    // Filter out invalid bounding boxes and keep confidence values
    std::vector<cv::Rect> rects;
    std::vector<float> confidences;
    rects.reserve(detections.size());
    confidences.reserve(detections.size());
    for (const auto& det : detections) {
        // Validate bounding box
        if (det.bbox.width > 0 && det.bbox.height > 0 &&
            det.bbox.x >= 0 && det.bbox.y >= 0 &&
            det.bbox.area() > 0) {
            rects.push_back(det.bbox);
            confidences.push_back(det.confidence);
        }
        // Silently skip invalid detections
    }

    // Use Hungarian algorithm for optimal matching
    updateTrackersWithHungarian(rects, confidences);

    // Convert internal trackers to TrackedResult format
    std::vector<TrackedResult> results;
    results.reserve(trackers.size());

    for (const auto& tracker : trackers) {
        TrackedResult result;
        result.trackId = tracker.id;
        result.bbox = tracker.boundingBox;
        result.velocity = cv::Point2f(tracker.velocity.x, tracker.velocity.y);
        // Use stored detection confidence, fallback to state-based if not available
        result.confidence = tracker.detectionConfidence;
        result.classId = -1; // We don't track class ID in this implementation
        result.isPredicted = (tracker.state == TrackedObject::LOST);
        result.age = tracker.age;
        result.missedFrames = tracker.missedFrames;

        results.push_back(result);
    }

    if (telemetry) {
        for (const auto& result : results) {
            telemetry->onTrackResult(frameIndexCounter, result);
        }
        telemetry->onFrameEnd(frameIndexCounter);
        telemetry->flush();
    }

    // Increment age for all trackers
    for (auto& tracker : trackers) {
        tracker.age++;
    }

    frameIndexCounter++;

    return results;
}

void KalmanIoUTracker6State::reset() {
    trackers.clear();
    nextId = 0;
    frameIndexCounter = 0;
    frameSizeHint.reset();
    if (telemetry) {
        telemetry->flush();
    }
}

int KalmanIoUTracker6State::getActiveTrackCount() const {
    return std::count_if(trackers.begin(), trackers.end(),
                         [](const TrackedObject& t) { return t.state == TrackedObject::TRACKING; });
}

int KalmanIoUTracker6State::getTotalTrackCount() const {
    return static_cast<int>(trackers.size());
}

std::optional<IObjectTracker::TrackState> KalmanIoUTracker6State::getTrackState(int trackId) const {
    const TrackedObject* tracker = findTracker(trackId);
    if (!tracker) {
        return std::nullopt;
    }
    return toTrackState(tracker->state);
}

IObjectTracker::TrackState KalmanIoUTracker6State::toTrackState(TrackedObject::State state) const {
    switch (state) {
    case TrackedObject::TRACKING:
        return TrackState::TRACKING;
    case TrackedObject::LOST:
        return TrackState::LOST;
    default:
        return TrackState::DELETED;
    }
}

const TrackedObject* KalmanIoUTracker6State::findTracker(int trackId) const {
    auto it = std::find_if(trackers.begin(), trackers.end(),
                           [trackId](const TrackedObject& t) { return t.id == trackId; });
    return (it != trackers.end()) ? &(*it) : nullptr;
}

// ========== IVisualTracker Implementation ==========

void KalmanIoUTracker6State::draw(cv::Mat& frame, const VisualizationConfig& config) {
    for (auto& tracker : trackers) {
        if (!config.showLostTracks && tracker.state == TrackedObject::LOST)
            continue;

        cv::Scalar color = tracker.color;

        // State-based coloring
        if (config.colorByState) {
            color = (tracker.state == TrackedObject::TRACKING) ? config.trackingColor : config.lostColor;
        }

        // Bounding box
        if (config.showBoundingBox) {
            cv::rectangle(frame, tracker.boundingBox, color, config.boundingBoxThickness);
        }

        // Track ID
        if (config.showTrackId) {
            std::string idText = "ID:" + std::to_string(tracker.id);
            cv::Point textPos(tracker.boundingBox.x, tracker.boundingBox.y - 5);
            cv::putText(frame, idText, textPos, cv::FONT_HERSHEY_SIMPLEX,
                        config.trackIdFontScale, config.trackIdColor, config.trackIdThickness);
        }

        // Center point
        if (config.showCenter) {
            cv::Point center(tracker.boundingBox.x + tracker.boundingBox.width / 2,
                             tracker.boundingBox.y + tracker.boundingBox.height / 2);
            tracker.lastActualCenter = center;
            cv::circle(frame, center, config.centerRadius, config.centerColor, cv::FILLED);
        }

        // Velocity/Direction
        if (config.showVelocity) {
            cv::Point startPoint = tracker.lastActualCenter;
            cv::Point endPoint = startPoint + cv::Point(
                                     static_cast<int>(tracker.velocity.x * config.velocityScale),
                                     static_cast<int>(tracker.velocity.y * config.velocityScale));
            cv::arrowedLine(frame, startPoint, endPoint, config.velocityColor, config.velocityThickness);
        }

        // Trajectory
        if (config.showTrajectory) {
            // Update trajectory
            tracker.trajectory.push_back(tracker.lastActualCenter);
            if (static_cast<int>(tracker.trajectory.size()) > config.trajectoryLength) {
                tracker.trajectory.pop_front();
            }

            // Draw trajectory
            for (size_t i = 1; i < tracker.trajectory.size(); ++i) {
                cv::line(frame, tracker.trajectory[i - 1], tracker.trajectory[i],
                         config.trajectoryColor, config.trajectoryThickness);
            }
        }

        // Confidence
        if (config.showConfidence) {
            float conf = (tracker.state == TrackedObject::TRACKING) ? 0.9f : 0.5f;
            std::string confText = "Conf:" + std::to_string(conf).substr(0, 4);
            cv::Point textPos(tracker.boundingBox.x, tracker.boundingBox.y + tracker.boundingBox.height + 15);
            cv::putText(frame, confText, textPos, cv::FONT_HERSHEY_SIMPLEX,
                        0.4, config.confidenceColor, 1);
        }

        // Age
        if (config.showAge) {
            std::string ageText = "Age:" + std::to_string(tracker.age);
            cv::Point textPos(tracker.boundingBox.x + tracker.boundingBox.width - 60,
                              tracker.boundingBox.y - 5);
            cv::putText(frame, ageText, textPos, cv::FONT_HERSHEY_SIMPLEX,
                        0.4, cv::Scalar(255, 255, 255), 1);
        }

        // Missed frames
        if (config.showMissedFrames && tracker.missedFrames > 0) {
            std::string missedText = "Miss:" + std::to_string(tracker.missedFrames);
            cv::Point textPos(tracker.boundingBox.x, tracker.boundingBox.y + tracker.boundingBox.height + 30);
            cv::putText(frame, missedText, textPos, cv::FONT_HERSHEY_SIMPLEX,
                        0.4, cv::Scalar(0, 0, 255), 1);
        }
    }
}

std::vector<std::vector<cv::Point>> KalmanIoUTracker6State::getTrajectories() const {
    std::vector<std::vector<cv::Point>> trajectories;
    trajectories.reserve(trackers.size());

    for (const auto& tracker : trackers) {
        std::vector<cv::Point> traj(tracker.trajectory.begin(), tracker.trajectory.end());
        trajectories.push_back(traj);
    }

    return trajectories;
}

std::vector<cv::Point> KalmanIoUTracker6State::getTrajectory(int trackId) const {
    const TrackedObject* tracker = findTracker(trackId);
    if (!tracker) {
        return {};
    }
    return std::vector<cv::Point>(tracker->trajectory.begin(), tracker->trajectory.end());
}

void KalmanIoUTracker6State::clearTrajectories() {
    for (auto& tracker : trackers) {
        tracker.trajectory.clear();
    }
}

void KalmanIoUTracker6State::setTrajectoryLength(int length) {
    config.maxTrajectoryLength = length;
}

// ========== IConfigurableTracker Implementation ==========

KalmanIoUConfig KalmanIoUTracker6State::getConfig() const {
    return config;
}

void KalmanIoUTracker6State::setConfig(const KalmanIoUConfig& newConfig) {
    if (validateConfig(newConfig)) {
        config = newConfig;
    } else {
        throw std::invalid_argument("Invalid configuration");
    }
}

void KalmanIoUTracker6State::setParameter(const std::string& key, const std::any& value) {
    if (key == "iouType") {
        config.iouType = std::any_cast<KalmanIoUConfig::IoUType>(value);
    } else if (key == "iouThreshold") {
        config.iouThreshold = std::any_cast<float>(value);
    } else if (key == "usePredictionInLost") {
        config.usePredictionInLost = std::any_cast<bool>(value);
    } else if (key == "maxLostFrames") {
        config.maxLostFrames = std::any_cast<int>(value);
    } else if (key == "removeOutOfBounds") {
        config.removeOutOfBounds = std::any_cast<bool>(value);
    } else if (key == "cameraBounds") {
        config.cameraBounds = std::any_cast<cv::Rect>(value);
    } else if (key == "processNoise") {
        config.processNoise = std::any_cast<float>(value);
    } else if (key == "measurementNoise") {
        config.measurementNoise = std::any_cast<float>(value);
    } else if (key == "errorCovPost") {
        config.errorCovPost = std::any_cast<float>(value);
    } else if (key == "maxTrajectoryLength") {
        config.maxTrajectoryLength = std::any_cast<int>(value);
    } else if (key == "softIoUSlack") {
        config.softIoUSlack = std::any_cast<float>(value);
    } else if (key == "alphaIoUExponent") {
        config.alphaIoUExponent = std::any_cast<float>(value);
    } else {
        throw std::invalid_argument("Unknown parameter key: " + key);
    }
}

std::any KalmanIoUTracker6State::getParameter(const std::string& key) const {
    if (key == "iouType") {
        return config.iouType;
    } else if (key == "iouThreshold") {
        return config.iouThreshold;
    } else if (key == "usePredictionInLost") {
        return config.usePredictionInLost;
    } else if (key == "maxLostFrames") {
        return config.maxLostFrames;
    } else if (key == "removeOutOfBounds") {
        return config.removeOutOfBounds;
    } else if (key == "cameraBounds") {
        return config.cameraBounds;
    } else if (key == "processNoise") {
        return config.processNoise;
    } else if (key == "measurementNoise") {
        return config.measurementNoise;
    } else if (key == "errorCovPost") {
        return config.errorCovPost;
    } else if (key == "maxTrajectoryLength") {
        return config.maxTrajectoryLength;
    } else if (key == "softIoUSlack") {
        return config.softIoUSlack;
    } else if (key == "alphaIoUExponent") {
        return config.alphaIoUExponent;
    } else {
        throw std::invalid_argument("Unknown parameter key: " + key);
    }
}

bool KalmanIoUTracker6State::validateConfig(const KalmanIoUConfig& cfg) const {
    if (cfg.iouThreshold < 0.0f || cfg.iouThreshold > 1.0f) {
        return false;
    }
    if (cfg.maxLostFrames < 0) {
        return false;
    }
    if (cfg.processNoise < 0.0f || cfg.measurementNoise < 0.0f || cfg.errorCovPost < 0.0f) {
        return false;
    }
    if (cfg.maxTrajectoryLength < 1) {
        return false;
    }
    if (cfg.softIoUSlack <= 0.0f) {
        return false;
    }
    if (cfg.alphaIoUExponent <= 0.0f) {
        return false;
    }
    return true;
}

void KalmanIoUTracker6State::setTelemetry(const std::shared_ptr<ITrackerTelemetry>& telemetryLogger) {
    telemetry = telemetryLogger;
}

void KalmanIoUTracker6State::clearTelemetry() {
    telemetry.reset();
}

void KalmanIoUTracker6State::setFrameSizeHint(const cv::Size& sourceSize, const cv::Size& processedSize) {
    frameSizeHint = FrameSizeInfo{sourceSize, processedSize};
}

void KalmanIoUTracker6State::clearFrameSizeHint() {
    frameSizeHint.reset();
}
