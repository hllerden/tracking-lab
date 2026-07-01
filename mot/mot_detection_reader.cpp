#include "mot_detection_reader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <iostream>

MotDetectionReader::MotDetectionReader(const std::string& detFilePath)
    : filePath_(detFilePath)
    , maxFrameIndex_(0)
    , totalDetections_(0)
    , valid_(false) {

    loadDetections();
}

std::vector<IObjectTracker::Detection> MotDetectionReader::getDetectionsForFrame(int frameIndex) const {
    auto it = cache_.find(frameIndex);
    if (it != cache_.end()) {
        return it->second;
    }
    return {};  // Empty vector if no detections for this frame
}

int MotDetectionReader::getTotalFrames() const {
    return maxFrameIndex_;
}

int MotDetectionReader::getTotalDetections() const {
    return totalDetections_;
}

int MotDetectionReader::getDetectionCount(int frameIndex) const {
    auto it = cache_.find(frameIndex);
    return (it != cache_.end()) ? static_cast<int>(it->second.size()) : 0;
}

bool MotDetectionReader::isValid() const {
    return valid_;
}

std::string MotDetectionReader::getFilePath() const {
    return filePath_;
}

void MotDetectionReader::reload() {
    cache_.clear();
    maxFrameIndex_ = 0;
    totalDetections_ = 0;
    valid_ = false;
    loadDetections();
}

MotDetectionReader::Statistics MotDetectionReader::getStatistics() const {
    Statistics stats;
    stats.totalFrames = static_cast<int>(cache_.size());
    stats.totalDetections = totalDetections_;
    stats.minDetPerFrame = std::numeric_limits<int>::max();
    stats.maxDetPerFrame = 0;
    stats.avgDetPerFrame = 0.0f;
    stats.minConfidence = std::numeric_limits<float>::max();
    stats.maxConfidence = std::numeric_limits<float>::lowest();
    stats.avgConfidence = 0.0f;

    if (cache_.empty()) {
        stats.minDetPerFrame = 0;
        return stats;
    }

    float totalConf = 0.0f;
    int totalDets = 0;

    for (const auto& [frameIdx, detections] : cache_) {
        int count = static_cast<int>(detections.size());
        stats.minDetPerFrame = std::min(stats.minDetPerFrame, count);
        stats.maxDetPerFrame = std::max(stats.maxDetPerFrame, count);

        for (const auto& det : detections) {
            stats.minConfidence = std::min(stats.minConfidence, det.confidence);
            stats.maxConfidence = std::max(stats.maxConfidence, det.confidence);
            totalConf += det.confidence;
            totalDets++;
        }
    }

    stats.avgDetPerFrame = static_cast<float>(totalDets) / stats.totalFrames;
    stats.avgConfidence = (totalDets > 0) ? (totalConf / totalDets) : 0.0f;

    return stats;
}

void MotDetectionReader::loadDetections() {
    std::ifstream file(filePath_);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open det.txt file: " + filePath_);
    }

    std::string line;
    int lineNumber = 0;
    int successCount = 0;
    int failCount = 0;

    while (std::getline(file, line)) {
        lineNumber++;

        // Skip empty lines
        if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }

        int frameIndex;
        IObjectTracker::Detection detection;

        if (parseLine(line, frameIndex, detection)) {
            cache_[frameIndex].push_back(detection);
            maxFrameIndex_ = std::max(maxFrameIndex_, frameIndex);
            totalDetections_++;
            successCount++;
        } else {
            failCount++;
            if (failCount <= 5) {  // Only warn for first 5 failures
                std::cerr << "Warning: Failed to parse line " << lineNumber
                         << " in " << filePath_ << std::endl;
            }
        }
    }

    file.close();

    if (successCount == 0) {
        throw std::runtime_error("No valid detections found in: " + filePath_);
    }

    valid_ = true;

    std::cout << "MotDetectionReader loaded: " << filePath_ << std::endl;
    std::cout << "  Total detections: " << totalDetections_ << std::endl;
    std::cout << "  Total frames: " << cache_.size() << std::endl;
    std::cout << "  Max frame index: " << maxFrameIndex_ << std::endl;

    if (failCount > 0) {
        std::cout << "  Warning: " << failCount << " lines failed to parse" << std::endl;
    }
}

bool MotDetectionReader::parseLine(const std::string& line, int& frameIndex, IObjectTracker::Detection& detection) {
    std::stringstream ss(line);
    std::string token;
    std::vector<std::string> tokens;

    // Split by comma
    while (std::getline(ss, token, ',')) {
        // Trim whitespace
        token.erase(0, token.find_first_not_of(" \t\r\n"));
        token.erase(token.find_last_not_of(" \t\r\n") + 1);
        tokens.push_back(token);
    }

    // MOTChallenge det.txt format: <frame>, -1, <x>, <y>, <w>, <h>, <conf>, -1, -1, -1
    // We need at least 7 values (frame, id, x, y, w, h, conf)
    if (tokens.size() < 7) {
        return false;
    }

    try {
        // Frame number (1-based)
        frameIndex = std::stoi(tokens[0]);

        // tokens[1] is always -1 (no track ID in detections)

        // Bounding box: x, y, width, height
        float x = std::stof(tokens[2]);
        float y = std::stof(tokens[3]);
        float w = std::stof(tokens[4]);
        float h = std::stof(tokens[5]);

        // Confidence score
        float conf = std::stof(tokens[6]);

        // Create detection
        detection.bbox = cv::Rect(
            static_cast<int>(x),
            static_cast<int>(y),
            static_cast<int>(w),
            static_cast<int>(h)
        );
        detection.confidence = conf;
        detection.classId = 0;  // MOTChallenge: all pedestrian class

        // Validate bounding box
        if (w <= 0 || h <= 0) {
            return false;
        }

        return true;

    } catch (const std::exception&) {
        return false;
    }
}
