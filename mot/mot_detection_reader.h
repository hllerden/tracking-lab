#ifndef MOT_DETECTION_READER_H
#define MOT_DETECTION_READER_H

#include "../i_object_tracker.h"
#include <string>
#include <map>
#include <vector>

/**
 * @brief MOTChallenge detection reader for tracker-only evaluation
 *
 * Reads pre-computed detections from det.txt files in MOTChallenge format.
 * This allows testing tracking algorithms independently from the detector,
 * which is required for official MOTChallenge submissions.
 *
 * The det.txt format (MOTChallenge 2D Box):
 * <frame>, -1, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, -1, -1, -1
 *
 * Example usage:
 * @code
 * MotDetectionReader reader("MOT17-Data/MOT17/train/MOT17-02-DPM/det/det.txt");
 *
 * for (int frame = 1; frame <= reader.getTotalFrames(); frame++) {
 *     auto detections = reader.getDetectionsForFrame(frame);
 *     // Process detections with tracker
 * }
 * @endcode
 *
 * Benefits over YOLO inference:
 * - ~10x faster (no inference overhead)
 * - Tests only tracking algorithm
 * - Compare different detector qualities (DPM, FRCNN, SDP)
 * - Required for official MOTChallenge submissions
 */
class MotDetectionReader {
public:
    /**
     * @brief Construct a new MotDetectionReader
     * @param detFilePath Path to det.txt file (e.g., "MOT17-02-DPM/det/det.txt")
     * @throws std::runtime_error if file cannot be opened or format is invalid
     */
    explicit MotDetectionReader(const std::string& detFilePath);

    /**
     * @brief Get detections for a specific frame
     * @param frameIndex Frame number (1-based, MOTChallenge format)
     * @return Vector of detections for this frame (empty if no detections)
     */
    std::vector<IObjectTracker::Detection> getDetectionsForFrame(int frameIndex) const;

    /**
     * @brief Get total number of frames with detections
     * @return Maximum frame number found in det.txt
     */
    int getTotalFrames() const;

    /**
     * @brief Get total number of detections across all frames
     * @return Sum of all detections
     */
    int getTotalDetections() const;

    /**
     * @brief Get detection count for a specific frame
     * @param frameIndex Frame number (1-based)
     * @return Number of detections in this frame
     */
    int getDetectionCount(int frameIndex) const;

    /**
     * @brief Check if file was loaded successfully
     * @return true if detections are available
     */
    bool isValid() const;

    /**
     * @brief Get source file path
     * @return Path to det.txt file
     */
    std::string getFilePath() const;

    /**
     * @brief Clear cached detections and reload
     */
    void reload();

    /**
     * @brief Get statistics about loaded detections
     */
    struct Statistics {
        int totalFrames;        ///< Number of frames with detections
        int totalDetections;    ///< Total detection count
        int minDetPerFrame;     ///< Minimum detections in a frame
        int maxDetPerFrame;     ///< Maximum detections in a frame
        float avgDetPerFrame;   ///< Average detections per frame
        float minConfidence;    ///< Minimum confidence score
        float maxConfidence;    ///< Maximum confidence score
        float avgConfidence;    ///< Average confidence score
    };

    /**
     * @brief Get detection statistics
     * @return Statistics struct with detection info
     */
    Statistics getStatistics() const;

private:
    /**
     * @brief Parse det.txt file and load all detections
     * @throws std::runtime_error if file cannot be read or format is invalid
     */
    void loadDetections();

    /**
     * @brief Parse a single line from det.txt
     * @param line CSV line in MOTChallenge format
     * @param frameIndex Output: frame number (1-based)
     * @param detection Output: parsed detection
     * @return true if line was successfully parsed
     */
    bool parseLine(const std::string& line, int& frameIndex, IObjectTracker::Detection& detection);

    std::string filePath_;                                          ///< Path to det.txt file
    std::map<int, std::vector<IObjectTracker::Detection>> cache_;  ///< Frame → detections mapping
    int maxFrameIndex_;                                             ///< Highest frame number
    int totalDetections_;                                           ///< Total detection count
    bool valid_;                                                    ///< File loaded successfully
};

#endif // MOT_DETECTION_READER_H
