#ifndef I_CONFIGURABLE_TRACKER_H
#define I_CONFIGURABLE_TRACKER_H

#include <any>
#include <string>
#include <stdexcept>

/**
 * @file i_configurable_tracker.h
 * @brief Optional interface for tracker configuration
 * @author Halil Erden
 * @date 17.10.2025
 *
 * This interface allows trackers to expose their configuration parameters
 * in a type-safe way while maintaining algorithm independence.
 *
 * Design Philosophy:
 * - Type-safe: Each tracker has its own strongly-typed config struct
 * - Optional: Not all trackers need runtime configuration
 * - Flexible: Allows both batch config updates and individual parameter changes
 * - Generic: Template-based design works with any config type
 *
 * Usage Example:
 * @code
 * struct KalmanIoUConfig {
 *     float iouThreshold = 0.3f;
 *     int maxLostFrames = 20;
 * };
 *
 * class KalmanIoUTracker : public IObjectTracker,
 *                           public IConfigurableTracker<KalmanIoUConfig> {
 *     // Implementation...
 * };
 * @endcode
 */

template<typename ConfigType>
class IConfigurableTracker {
public:
    virtual ~IConfigurableTracker() = default;

    /**
     * @brief Get current configuration
     * @return Copy of current configuration struct
     */
    virtual ConfigType getConfig() const = 0;

    /**
     * @brief Set new configuration
     * @param config New configuration to apply
     *
     * This performs batch update of all configuration parameters.
     */
    virtual void setConfig(const ConfigType& config) = 0;

    /**
     * @brief Set individual parameter by name
     * @param key Parameter name (as string)
     * @param value Parameter value (type-erased with std::any)
     *
     * This allows runtime parameter modification without knowing
     * the full config struct at compile time. Useful for:
     * - GUI configuration panels
     * - Command-line parameter parsing
     * - Dynamic tuning during runtime
     *
     * @throws std::invalid_argument if key is unknown
     * @throws std::bad_any_cast if value type doesn't match
     *
     * @code
     * tracker->setParameter("iouThreshold", 0.5f);
     * tracker->setParameter("maxLostFrames", 30);
     * @endcode
     */
    virtual void setParameter(const std::string& key, const std::any& value) = 0;

    /**
     * @brief Get individual parameter by name
     * @param key Parameter name
     * @return Parameter value as std::any
     * @throws std::invalid_argument if key is unknown
     */
    virtual std::any getParameter(const std::string& key) const = 0;

    /**
     * @brief Validate configuration
     * @param config Configuration to validate
     * @return true if valid, false otherwise
     *
     * Override this to implement config validation logic.
     * Default implementation always returns true.
     */
    virtual bool validateConfig(const ConfigType& config) const {
        return true;
    }

    /**
     * @brief Reset configuration to defaults
     * Restores default configuration values.
     */
    virtual void resetConfig() {
        setConfig(ConfigType{});
    }
};

#endif // I_CONFIGURABLE_TRACKER_H
