#ifndef I_OBJECT_TRACKER_H
#define I_OBJECT_TRACKER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <optional>

/**
 * @file i_object_tracker.h
 * @brief Core interface for all object tracking implementations
 * @author Halil Erden
 * @date 17.10.2025
 *
 * This is the minimal contract that all trackers must implement.
 * It defines the essential operations for multi-object tracking:
 * - Processing detections and maintaining tracks
 * - Querying track states
 * - Resetting tracker state
 *
 * Design Philosophy:
 * - Keep it minimal: only core tracking operations
 * - Algorithm-agnostic: works with IoU-based, feature-based, or hybrid trackers
 * - No visualization: separate concern (see IVisualTracker)
 * - No configuration: separate concern (see IConfigurableTracker)
 */

class IObjectTracker {
public:
    /**
     * @brief Detection input from object detector (e.g., YOLO)
     */
    struct Detection {
        cv::Rect bbox;                  ///< Bounding box
        float confidence = 1.0f;        ///< Detection confidence [0-1]
        int classId = -1;               ///< Object class ID (-1 for unknown)
        cv::Mat featureVector;          ///< Optional: appearance features (for DeepSORT, etc.)

        Detection() = default;
        Detection(const cv::Rect& box, float conf = 1.0f, int cls = -1)
            : bbox(box), confidence(conf), classId(cls) {}
    };

    /**
     * @brief Tracking result for a single object
     */
    struct TrackedResult {
        int trackId;                    ///< Unique track ID (persistent across frames)
        cv::Rect bbox;                  ///< Current bounding box
        cv::Point2f velocity;           ///< Velocity vector (pixels/frame)
        float confidence;               ///< Confidence score [0-1]
        int classId;                    ///< Object class ID
        bool isPredicted;               ///< True if this is a prediction (no matching detection)
        int age;                        ///< How many frames this track has existed
        int missedFrames;               ///< Consecutive frames without detection

        TrackedResult()
            : trackId(-1), velocity(0, 0), confidence(0), classId(-1),
              isPredicted(false), age(0), missedFrames(0) {}
    };

    /**
     * @brief Tracker state for querying
     */
    enum class TrackState {
        TENTATIVE,  ///< Newly created, not yet confirmed
        TRACKING,   ///< Active tracking with recent detections
        LOST,       ///< Lost track (no detection for several frames)
        DELETED     ///< Marked for deletion
    };

    virtual ~IObjectTracker() = default;

    /**
     * @brief Update tracker with new detections
     * @param detections Vector of detections from current frame
     * @return Vector of tracked results (active tracks)
     *
     * This is the main tracker interface. It:
     * 1. Predicts current state of existing tracks
     * 2. Associates detections with tracks
     * 3. Updates matched tracks
     * 4. Creates new tracks for unmatched detections
     * 5. Marks tracks as lost if no match found
     */
    virtual std::vector<TrackedResult> update(const std::vector<Detection>& detections) = 0;

    /**
     * @brief Reset tracker to initial state
     * Clears all tracks and resets internal state.
     */
    virtual void reset() = 0;

    /**
     * @brief Get number of currently active tracks
     * @return Count of tracks in TRACKING or TENTATIVE state
     */
    virtual int getActiveTrackCount() const = 0;

    /**
     * @brief Get total number of tracks (including lost/deleted)
     * @return Total track count
     */
    virtual int getTotalTrackCount() const = 0;

    /**
     * @brief Get state of a specific track
     * @param trackId Track ID to query
     * @return Track state or nullopt if track doesn't exist
     */
    virtual std::optional<TrackState> getTrackState(int trackId) const = 0;
};

#endif // I_OBJECT_TRACKER_H
