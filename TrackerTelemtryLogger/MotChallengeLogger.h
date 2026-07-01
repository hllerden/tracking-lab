#ifndef MOT_CHALLENGE_LOGGER_H
#define MOT_CHALLENGE_LOGGER_H

#include "../i_tracker_telemetry.h"
#include <fstream>
#include <optional>
#include <string>

/**
 * @brief MOTChallenge formatında metrik üreten telemetri logger'ı.
 */
class MotChallengeLogger : public ITrackerTelemetry {
public:
    explicit MotChallengeLogger(const std::string& filePath, bool overwrite = true);
    ~MotChallengeLogger() override;

    void onFrameStart(const FrameMetadata& metadata) override;
    void onTrackResult(int frameIndex, const IObjectTracker::TrackedResult& result) override;
    void onFrameEnd(int frameIndex) override;
    void flush() override;

private:
    void openStream(bool overwrite);
    void writeLine(int frameIndex, const IObjectTracker::TrackedResult& result);
    void updateScaling();

    std::string outputPath;
    std::ofstream stream;
    std::optional<FrameMetadata> currentFrame;
    float scaleX = 1.0f;
    float scaleY = 1.0f;
};

#endif // MOT_CHALLENGE_LOGGER_H
