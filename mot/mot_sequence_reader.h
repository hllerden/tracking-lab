#ifndef MOT_SEQUENCE_READER_H
#define MOT_SEQUENCE_READER_H

#include <opencv2/opencv.hpp>
#include <string>
#include <map>
#include <vector>

/**
 * @brief MOTChallenge sequence reader for frame-by-frame processing
 *
 * Reads MOT17 dataset sequences frame by frame, similar to video playback.
 * Supports reading metadata from seqinfo.ini and sequential frame loading.
 *
 * Expected directory structure:
 * <sequencePath>/
 *   ├── img1/
 *   │   ├── 000001.jpg
 *   │   ├── 000002.jpg
 *   │   └── ...
 *   ├── seqinfo.ini
 *   └── gt/ (optional)
 *       └── gt.txt
 */
class MotSequenceReader {
public:
    /**
     * @brief Construct a new MotSequenceReader
     * @param sequencePath Path to MOT sequence directory (e.g., "MOT17-Data/MOT17/train/MOT17-02-DPM")
     * @throws std::runtime_error if sequence path is invalid or seqinfo.ini cannot be read
     */
    explicit MotSequenceReader(const std::string& sequencePath);

    /**
     * @brief Read the next frame in sequence
     * @param frame Output frame (will be loaded from disk)
     * @return true if frame was successfully loaded, false if end of sequence reached
     */
    bool nextFrame(cv::Mat& frame);

    /**
     * @brief Check if more frames are available
     * @return true if there are more frames to read
     */
    bool hasNext() const;

    /**
     * @brief Get current frame index (0-based)
     * @return Current frame index
     */
    int getCurrentFrameIndex() const;

    /**
     * @brief Get total number of frames in sequence
     * @return Total frame count from seqinfo.ini
     */
    int getTotalFrames() const;

    /**
     * @brief Get frames per second
     * @return FPS value from seqinfo.ini
     */
    double getFPS() const;

    /**
     * @brief Get frame size (width x height)
     * @return Frame dimensions from seqinfo.ini
     */
    cv::Size getFrameSize() const;

    /**
     * @brief Get sequence name
     * @return Sequence name from seqinfo.ini
     */
    std::string getSequenceName() const;

    /**
     * @brief Reset reader to beginning of sequence
     */
    void reset();

    /**
     * @brief Get ground truth data for current frame (if available)
     * @param frameIndex Frame index (1-based, MOT format)
     * @return Vector of ground truth boxes for this frame
     */
    struct GroundTruthBox {
        int trackId;
        cv::Rect2f bbox;  // x, y, width, height
        float confidence;
        int classId;
    };
    std::vector<GroundTruthBox> getGroundTruthForFrame(int frameIndex) const;

private:
    /**
     * @brief Parse seqinfo.ini file
     * @return Map of key-value pairs from INI file
     */
    std::map<std::string, std::string> parseSeqInfo();

    /**
     * @brief Load ground truth file (gt/gt.txt) if available
     */
    void loadGroundTruth();

    /**
     * @brief Build frame file path for given index
     * @param frameIndex 0-based frame index
     * @return Full path to frame image file
     */
    std::string getFramePath(int frameIndex) const;

    std::string sequencePath_;
    std::string imageDir_;
    std::string imageExt_;
    std::string sequenceName_;

    int currentFrameIndex_;
    int totalFrames_;
    double frameRate_;
    cv::Size frameSize_;

    // Ground truth data: frameIndex (1-based) -> vector of boxes
    std::map<int, std::vector<GroundTruthBox>> groundTruth_;
    bool groundTruthLoaded_;
};

#endif // MOT_SEQUENCE_READER_H
