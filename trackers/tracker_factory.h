#ifndef TRACKER_FACTORY_H
#define TRACKER_FACTORY_H

#include <memory>
#include <stdexcept>
#include "../i_object_tracker.h"
#include "../i_configurable_tracker.h"
#include "../i_tracker_telemetry.h"
#include "kalman_iou/kalman_iou_tracker_6state.h"
#include "kalman_iou/kalman_iou_tracker.h"
#include "kalman_iou/kalman_iou_byte_track.h"

/**
 * @file tracker_factory.h
 * @brief Factory for creating different Kalman+IoU tracker variants
 * @author Halil Erden
 * @date 2025-01-22
 *
 * This factory provides a centralized way to create and switch between
 * different tracker implementations at compile-time using enum selection.
 *
 * Benefits:
 * - Eliminates TrackedObject redefinition errors
 * - Clean compile-time tracker selection
 * - Type-safe enum-based interface
 * - Easy integration with main.cpp and benchmarks
 *
 * Usage:
 * @code
 *   // Create with default config
 *   auto tracker = TrackerFactory::create(TrackerType::BYTETRACK);
 *
 *   // Create with custom config
 *   KalmanIoUConfig config;
 *   config.iouThreshold = 0.3f;
 *   auto tracker = TrackerFactory::create(TrackerType::KALMAN_8STATE, config);
 *
 *   // Get tracker info
 *   std::cout << TrackerFactory::getDescription(TrackerType::BYTETRACK);
 * @endcode
 */

/**
 * @brief Available tracker types
 */
enum class TrackerType {
    /**
     * @brief 6-state Kalman filter [x, y, w, h, vx, vy]
     *
     * Baseline implementation with standard IoU matching.
     * - State: position + size + velocity (position only)
     * - Config: iouThreshold=0.2, maxLostFrames=20
     * - Use case: Stable baseline, good for comparison
     */
    KALMAN_6STATE,

    /**
     * @brief 8-state Kalman filter [x, y, w, h, vx, vy, vw, vh]
     *
     * Advanced implementation with adaptive noise covariance (BoT-SORT style).
     * - State: position + size + velocity (position + size)
     * - Adaptive Q/R matrices based on bounding box dimensions
     * - Config: iouThreshold=0.25, maxLostFrames=30
     * - Use case: Better size prediction, adaptive tracking
     */
    KALMAN_8STATE,

    /**
     * @brief ByteTrack: 8-state + two-stage matching
     *
     * Implements ByteTrack algorithm with high/low confidence detection matching.
     * - State: 8-state (same as KALMAN_8STATE)
     * - Stage 1: High confidence (>0.6) → normal Hungarian matching
     * - Stage 2: Low confidence (0.1-0.6) → recover LOST trackers only
     * - Config: iouThreshold=0.25, highConfThreshold=0.6, lowConfFloor=0.1
     * - Use case: Best for occluded/low-confidence detections
     */
    BYTETRACK
};

/**
 * @brief Factory class for creating tracker instances
 */
class TrackerFactory {
public:
    /**
     * @brief Create a tracker instance with default configuration
     *
     * @param type Tracker type to create
     * @return Unique pointer to IObjectTracker interface
     * @throws std::invalid_argument if type is invalid
     */
    static std::unique_ptr<IObjectTracker> create(TrackerType type) {
        return create(type, KalmanIoUConfig());
    }

    /**
     * @brief Create a tracker instance with custom configuration
     *
     * @param type Tracker type to create
     * @param config Configuration parameters
     * @return Unique pointer to IObjectTracker interface
     * @throws std::invalid_argument if type is invalid
     *
     * @note The config parameter is KalmanIoUConfig which is shared
     *       across all tracker types. ByteTrack-specific parameters
     *       (highConfThreshold, lowConfFloor) are hardcoded in
     *       KalmanIoUByteTrack implementation.
     */
    static std::unique_ptr<IObjectTracker> create(
        TrackerType type,
        const KalmanIoUConfig& config
    ) {
        switch (type) {
            case TrackerType::KALMAN_6STATE: {
                auto tracker = std::make_unique<KalmanIoUTracker6State>(config);
                return tracker;
            }

            case TrackerType::KALMAN_8STATE: {
                auto tracker = std::make_unique<KalmanIoUTracker>(config);
                return tracker;
            }

            case TrackerType::BYTETRACK: {
                auto tracker = std::make_unique<KalmanIoUByteTrack>(config);
                return tracker;
            }

            default:
                throw std::invalid_argument("Invalid tracker type");
        }
    }

    /**
     * @brief Get human-readable tracker type name
     *
     * @param type Tracker type
     * @return Short name (e.g., "6state", "8state", "bytetrack")
     */
    static const char* getTypeName(TrackerType type) {
        switch (type) {
            case TrackerType::KALMAN_6STATE:
                return "6state";
            case TrackerType::KALMAN_8STATE:
                return "8state";
            case TrackerType::BYTETRACK:
                return "bytetrack";
            default:
                return "unknown";
        }
    }

    /**
     * @brief Get detailed tracker description
     *
     * @param type Tracker type
     * @return Description string with implementation details
     */
    static const char* getDescription(TrackerType type) {
        switch (type) {
            case TrackerType::KALMAN_6STATE:
                return "Kalman 6-state [x,y,w,h,vx,vy] - Baseline tracker";
            case TrackerType::KALMAN_8STATE:
                return "Kalman 8-state [x,y,w,h,vx,vy,vw,vh] - Adaptive noise (BoT-SORT)";
            case TrackerType::BYTETRACK:
                return "ByteTrack 8-state - Two-stage matching (high/low confidence)";
            default:
                return "Unknown tracker type";
        }
    }

    /**
     * @brief Get default configuration for a tracker type
     *
     * @param type Tracker type
     * @return Default configuration optimized for the tracker type
     */
    static KalmanIoUConfig getDefaultConfig(TrackerType type) {
        KalmanIoUConfig config;

        switch (type) {
            case TrackerType::KALMAN_6STATE:
                config.iouThreshold = 0.2f;
                config.maxLostFrames = 20;
                break;

            case TrackerType::KALMAN_8STATE:
                config.iouThreshold = 0.25f;
                config.maxLostFrames = 30;
                break;

            case TrackerType::BYTETRACK:
                config.iouThreshold = 0.25f;
                config.maxLostFrames = 30;
                // Note: ByteTrack-specific thresholds (0.6, 0.1) are
                // hardcoded in KalmanIoUByteTrack::update()
                break;
        }

        return config;
    }

    /**
     * @brief Apply KalmanIoUConfig to any tracker created by this factory.
     *
     * Uses IConfigurableTracker<KalmanIoUConfig> dynamic_cast so the caller
     * does not need to know the concrete tracker type.
     *
     * @param tracker Raw pointer from TrackerFactory::create().get()
     * @param config  Configuration to apply
     */
    static void applyConfig(IObjectTracker* tracker, const KalmanIoUConfig& config) {
        if (auto* c = dynamic_cast<IConfigurableTracker<KalmanIoUConfig>*>(tracker))
            c->setConfig(config);
    }

    /**
     * @brief Set telemetry logger and frame size hint on any factory-created tracker.
     *
     * Centralises the type-specific dynamic_cast required because setTelemetry()
     * and setFrameSizeHint() are not part of any shared interface.
     *
     * @param tracker       Raw pointer from TrackerFactory::create().get()
     * @param telemetry     Telemetry logger instance
     * @param sourceSize    Original frame resolution
     * @param processedSize Processed/scaled frame resolution
     */
    static void applyTelemetry(IObjectTracker* tracker,
                               std::shared_ptr<ITrackerTelemetry> telemetry,
                               cv::Size sourceSize,
                               cv::Size processedSize) {
        if (auto* t = dynamic_cast<KalmanIoUTracker6State*>(tracker)) {
            t->setTelemetry(telemetry);
            t->setFrameSizeHint(sourceSize, processedSize);
        } else if (auto* t = dynamic_cast<KalmanIoUTracker*>(tracker)) {
            t->setTelemetry(telemetry);
            t->setFrameSizeHint(sourceSize, processedSize);
        } else if (auto* t = dynamic_cast<KalmanIoUByteTrack*>(tracker)) {
            t->setTelemetry(telemetry);
            t->setFrameSizeHint(sourceSize, processedSize);
        }
    }
};

/**
 * @brief Compile-time default tracker selection
 *
 * Define DEFAULT_TRACKER_TYPE before including this header to override.
 * Example:
 * @code
 *   #define DEFAULT_TRACKER_TYPE TrackerType::BYTETRACK
 *   #include "trackers/tracker_factory.h"
 * @endcode
 */
#ifndef DEFAULT_TRACKER_TYPE
#define DEFAULT_TRACKER_TYPE TrackerType::BYTETRACK
#endif

#endif // TRACKER_FACTORY_H