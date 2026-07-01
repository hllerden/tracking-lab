#ifndef I_VISUAL_TRACKER_H
#define I_VISUAL_TRACKER_H

#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @file i_visual_tracker.h
 * @brief Optional interface for tracker visualization capabilities
 * @author Halil Erden
 * @date 17.10.2025
 *
 * This interface is separate from IObjectTracker to follow the
 * Interface Segregation Principle. Not all trackers need visualization,
 * and some applications may want to implement custom visualization.
 *
 * Design Philosophy:
 * - Separate concern: visualization is independent of tracking logic
 * - Optional implementation: trackers can choose to implement this
 * - Flexible configuration: VisualizationConfig allows runtime control
 */

class IVisualTracker {
public:
    /**
     * @brief Configuration for visualization options
     */
    struct VisualizationConfig {
        // Bounding Box Options
        bool showBoundingBox = true;        ///< Draw bounding boxes
        cv::Scalar boundingBoxColor = cv::Scalar(0, 255, 0);  ///< Default: green
        int boundingBoxThickness = 2;

        // Track ID Options
        bool showTrackId = true;            ///< Display track IDs
        cv::Scalar trackIdColor = cv::Scalar(0, 255, 255);    ///< Default: yellow
        double trackIdFontScale = 0.6;
        int trackIdThickness = 2;

        // Trajectory Options
        bool showTrajectory = false;        ///< Draw object paths
        int trajectoryLength = 30;          ///< Number of points to keep
        cv::Scalar trajectoryColor = cv::Scalar(255, 0, 0);   ///< Default: blue
        int trajectoryThickness = 2;

        // Velocity/Direction Options
        bool showVelocity = false;          ///< Draw velocity vectors
        float velocityScale = 3.0f;         ///< Scale factor for velocity arrows
        cv::Scalar velocityColor = cv::Scalar(0, 0, 255);     ///< Default: red
        int velocityThickness = 2;

        // Center Point Options
        bool showCenter = false;            ///< Draw center points
        cv::Scalar centerColor = cv::Scalar(255, 255, 0);     ///< Default: cyan
        int centerRadius = 4;

        // Confidence Options
        bool showConfidence = false;        ///< Display confidence scores
        cv::Scalar confidenceColor = cv::Scalar(255, 255, 255); ///< Default: white

        // State-based Coloring
        bool colorByState = true;          ///< Different colors for different states
        cv::Scalar trackingColor = cv::Scalar(0, 255, 0);     ///< TRACKING state: green
        cv::Scalar tentativeColor = cv::Scalar(255, 165, 0);  ///< TENTATIVE state: orange
        cv::Scalar lostColor = cv::Scalar(0, 0, 255);         ///< LOST state: red

        // Label Options
        bool showClassLabel = false;        ///< Show class name (requires class names map)
        bool showAge = true;               ///< Show track age
        bool showMissedFrames = true;      ///< Show missed frame count

        // State filtering
        bool showLostTracks = true;        ///< Draw LOST-state tracks (red boxes)

        // ReID score overlay
        bool showReidScore = false;        ///< Overlay cosine similarity "R:87%" on bbox

        VisualizationConfig() = default;
    };

    virtual ~IVisualTracker() = default;

    /**
     * @brief Draw tracking results on frame
     * @param frame Input/output frame to draw on (will be modified)
     * @param config Visualization configuration
     *
     * This method draws all active tracks according to the configuration.
     * It should be called after update() to visualize current state.
     */
    virtual void draw(cv::Mat& frame, const VisualizationConfig& config) = 0;

    /**
     * @brief Get trajectories of all tracked objects
     * @return Vector of trajectories (each trajectory is a vector of points)
     *
     * Useful for custom visualization or trajectory analysis.
     * Returns empty vector if trajectory tracking is not supported.
     */
    virtual std::vector<std::vector<cv::Point>> getTrajectories() const = 0;

    /**
     * @brief Get trajectory for a specific track
     * @param trackId Track ID to query
     * @return Trajectory points or empty vector if not available
     */
    virtual std::vector<cv::Point> getTrajectory(int trackId) const = 0;

    /**
     * @brief Clear all trajectory history
     * Useful for long-running applications to prevent memory growth.
     */
    virtual void clearTrajectories() = 0;

    /**
     * @brief Set maximum trajectory length
     * @param length Maximum number of points to keep per track
     */
    virtual void setTrajectoryLength(int length) = 0;
};

#endif // I_VISUAL_TRACKER_H
