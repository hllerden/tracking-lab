# Kalman IoU Tracker Quick Reference

This tracker implements an 8-state Kalman filter coupled with IoU-based data association (Hungarian algorithm). Use this page to refresh the key design points before making changes.

## State Model

- **State vector (8x1):** `[x, y, w, h, vx, vy, vw, vh]`
- **Measurement (4x1):** `[x, y, w, h]` from detections
- **Transition:** constant-velocity (dt = 1) for both position and size
- **Process noise:** higher variance on velocities to allow adaptation, lower on size

## Workflow Overview

1. **Prediction:** Kalman filter predicts bbox and velocities.
2. **Association:** Costs = `1 - clamp(IoU)` (supports CIOU/GIOU/DIOU/SIOU/AIOU). Hungarian solver assigns detections.
3. **Correction:** Matched trackers correct with measurements; statePost updates velocity/sizeVelocity.
4. **Lifecycle:** Unmatched trackers increment `missedFrames`, optionally use predictions when lost, removed after `maxLostFrames`.
5. **Spawning:** New trackers init with zero velocities; telemetry state stored for visualization/logging.

## Key Config Knobs (`KalmanIoUConfig`)

- `IoUType`, `iouThreshold` control association strictness.
- `usePredictionInLost`, `maxLostFrames` manage lost-track handling.
- `processNoise`, `measurementNoise` tune Kalman responsiveness.
- `removeOutOfBounds` and `cameraBounds` guard against stray tracks.

## Telemetry & Visualization

- `TrackedObject` keeps `velocity`, `sizeVelocity`, centers, and trajectory.
- Results exposed via `IObjectTracker::TrackedResult` for downstream consumers.
- Optional telemetry logger (`ITrackerTelemetry`) receives frame/track callbacks.

## File Map

- `kalman_iou_tracker.h/.cpp` – vanilla Kalman+IoU implementation
- `kalman_iou_byte_track.h/.cpp` – ByteTrack-inspired two-stage association version
- `../helpers/HungarianAlgorithm.*` – assignment solver
- Interfaces in project root (`i_object_tracker.h`, etc.)

## Quick Tips

- When adding new IoU metrics, extend `computeIoU` and update configuration enum.
- For stability issues, revisit noise covariance matrices or velocity clamps.
- To debug association, instrument the Hungarian cost matrix before solving.
