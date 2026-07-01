#include "MotChallengeLogger.h"

#include <filesystem>
#include <iomanip>

using namespace std;

MotChallengeLogger::MotChallengeLogger(const std::string& filePath, bool overwrite)
    : outputPath(filePath) {
    openStream(overwrite);
}

MotChallengeLogger::~MotChallengeLogger() {
    flush();
    if (stream.is_open()) {
        stream.close();
    }
}

void MotChallengeLogger::openStream(bool overwrite) {
    namespace fs = std::filesystem;
    fs::path path(outputPath);
    if (path.has_parent_path()) {
        fs::create_directories(path.parent_path());
    }
    ios_base::openmode mode = overwrite ? ios::out | ios::trunc : ios::out | ios::app;
    stream.open(outputPath, mode);
}

void MotChallengeLogger::onFrameStart(const FrameMetadata& metadata) {
    currentFrame = metadata;
    updateScaling();
}

void MotChallengeLogger::writeLine(int frameIndex, const IObjectTracker::TrackedResult& result) {
    if (!stream.is_open()) {
        return;
    }
    int motFrameIndex = frameIndex + 1; // MOT formatı 1 tabanlı kare index'i kullanır
    float confidence = result.isPredicted ? 0.01f : result.confidence;

    float x = result.bbox.x * scaleX;
    float y = result.bbox.y * scaleY;
    float w = result.bbox.width * scaleX;
    float h = result.bbox.height * scaleY;

    stream << motFrameIndex << ','
           << result.trackId << ','
           << x << ','
           << y << ','
           << w << ','
           << h << ','
           << fixed << setprecision(3) << confidence << ','
           << -1 << ',' << -1 << ',' << -1 << '\n';
}

void MotChallengeLogger::onTrackResult(int frameIndex, const IObjectTracker::TrackedResult& result) {
    writeLine(frameIndex, result);
}

void MotChallengeLogger::onFrameEnd(int frameIndex) {
    (void)frameIndex;
}

void MotChallengeLogger::flush() {
    if (stream.is_open()) {
        stream.flush();
    }
}

void MotChallengeLogger::updateScaling() {
    scaleX = 1.0f;
    scaleY = 1.0f;
    if (!currentFrame.has_value()) {
        return;
    }
    if (currentFrame->sourceSize.has_value() && currentFrame->processedSize.has_value()) {
        const cv::Size& src = *currentFrame->sourceSize;
        const cv::Size& proc = *currentFrame->processedSize;
        if (proc.width > 0 && proc.height > 0) {
            scaleX = static_cast<float>(src.width) / static_cast<float>(proc.width);
            scaleY = static_cast<float>(src.height) / static_cast<float>(proc.height);
        }
    }
}
