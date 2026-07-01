# MOT17 Dataset Usage Guide

This document explains how to use the MOT17 dataset with the tracking system and evaluate results with TrackEval.

## Overview

The project now includes `MotSequenceReader` class that reads MOT17 sequences frame-by-frame (similar to video playback) and generates tracking results in MOTChallenge format.

## Directory Structure

```
opencv-yolo/
├── MOT17-Data/           # Symbolic link to MOT17 dataset
│   ├── MOT17/
│   │   ├── train/        # Training sequences with ground truth
│   │   │   ├── MOT17-02-DPM/
│   │   │   │   ├── img1/          # Frame images (000001.jpg, 000002.jpg, ...)
│   │   │   │   ├── gt/gt.txt      # Ground truth annotations
│   │   │   │   ├── det/det.txt    # Detection results
│   │   │   │   └── seqinfo.ini    # Sequence metadata
│   │   │   ├── MOT17-04-DPM/
│   │   │   └── ...
│   │   └── test/         # Test sequences (no ground truth)
│   └── MOT17Labels/      # Alternative label format
├── output/               # Generated tracking results
└── models/               # YOLO models
```

## MOTChallenge Format

### Ground Truth Format (gt/gt.txt)
```
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
```

Example:
```
1,1,912,484,97,109,0,7,1
2,1,912,484,97,109,0,7,1
```

### Tracker Output Format
The `MotChallengeLogger` class automatically generates output in the same format:
```
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, -1, -1, -1
```

## Detection Modes

The project supports two detection modes:

### Mode 1: YOLO + Tracking (End-to-End)
- Uses YOLO for detection + KalmanIoU for tracking
- Tests complete detection & tracking system
- Slower (~50ms/frame with GPU)
- Use this to evaluate your YOLO model

### Mode 2: Tracker-Only (det.txt)
- Uses pre-computed detections from det.txt
- Tests **only** tracking algorithm
- **~10x faster** (~5ms/frame)
- Required for official MOTChallenge submissions
- Compare tracker performance across different detectors (DPM, FRCNN, SDP)

## Usage

### 1. YOLO Mode: Single Sequence

Edit [main.cpp](main.cpp) and uncomment MODE 1:

```cpp
return processMOT17Sequence(
    projectBasePath + "/MOT17-Data/MOT17/train/MOT17-02-DPM",
    projectBasePath + "/output/MOT17-02-DPM-YOLO.txt",
    projectBasePath + "/models/yolov9s.onnx",
    true,  // showVisualization
    runOnGPU
);
```

Build and run:
```bash
cd build
cmake --build .
./yoloExercise
```

### 2. Tracker-Only Mode: Single Sequence (NEW!)

Edit [main.cpp](main.cpp) and uncomment MODE 2:

```cpp
return processMOT17SequenceTrackerOnly(
    projectBasePath + "/MOT17-Data/MOT17/train/MOT17-02-DPM",
    projectBasePath + "/output/MOT17-02-DPM-tracker.txt",
    true  // showVisualization
);
```

This reads detections from `MOT17-02-DPM/det/det.txt` and only runs tracking.

### 3. Compare Different Detectors (MODE 3)

Test tracker with DPM, FRCNN, and SDP detections:

```cpp
std::vector<std::string> detectors = {"DPM", "FRCNN", "SDP"};
for (const auto& det : detectors) {
    std::string seqPath = projectBasePath + "/MOT17-Data/MOT17/train/MOT17-02-" + det;
    std::string outPath = projectBasePath + "/output/MOT17-02-" + det + "-tracker.txt";
    processMOT17SequenceTrackerOnly(seqPath, outPath, false);
}
return 0;
```

Then evaluate with TrackEval to see tracker performance across detection qualities.

### 4. Batch Processing (MODE 4)

Uncomment MODE 4 in [main.cpp](main.cpp):

```cpp
std::vector<std::string> sequences = {
    "MOT17-02-DPM", "MOT17-04-DPM", "MOT17-05-DPM",
    "MOT17-09-DPM", "MOT17-10-DPM", "MOT17-11-DPM", "MOT17-13-DPM"
};
for (const auto& seqName : sequences) {
    std::string seqPath = projectBasePath + "/MOT17-Data/MOT17/train/" + seqName;
    std::string outPath = projectBasePath + "/output/" + seqName + ".txt";
    processMOT17Sequence(seqPath, outPath, projectBasePath + "/models/yolov9s.onnx", false, runOnGPU);
}
```

### 3. Using MotSequenceReader Directly

```cpp
#include "mot_sequence_reader.h"
#include "TrackerTelemtryLogger/MotChallengeLogger.h"

// Initialize reader
MotSequenceReader reader("MOT17-Data/MOT17/train/MOT17-02-DPM");

// Get sequence info
std::cout << "Sequence: " << reader.getSequenceName() << std::endl;
std::cout << "Total frames: " << reader.getTotalFrames() << std::endl;
std::cout << "FPS: " << reader.getFPS() << std::endl;

// Initialize logger
auto logger = std::make_shared<MotChallengeLogger>("output/results.txt");

// Process frames
cv::Mat frame;
while (reader.nextFrame(frame)) {
    // Your tracking code here
    // Results are automatically logged via telemetry
}

logger->flush();
```

## MotSequenceReader API

### Constructor
```cpp
MotSequenceReader(const std::string& sequencePath)
```
- Loads sequence metadata from `seqinfo.ini`
- Validates image directory exists
- Optionally loads ground truth data

### Methods

```cpp
bool nextFrame(cv::Mat& frame)
```
Reads the next frame in sequence. Returns `false` when sequence ends.

```cpp
bool hasNext() const
```
Check if more frames are available.

```cpp
int getCurrentFrameIndex() const
```
Get current frame index (0-based).

```cpp
int getTotalFrames() const
```
Get total number of frames from `seqinfo.ini`.

```cpp
double getFPS() const
```
Get frames per second.

```cpp
cv::Size getFrameSize() const
```
Get frame dimensions (width × height).

```cpp
std::string getSequenceName() const
```
Get sequence name.

```cpp
void reset()
```
Reset reader to beginning of sequence.

```cpp
std::vector<GroundTruthBox> getGroundTruthForFrame(int frameIndex) const
```
Get ground truth annotations for a specific frame (1-based index).

## TrackEval Integration

### Output Format
The tracking results are automatically saved in MOTChallenge format compatible with TrackEval.

### Directory Structure for TrackEval
```
trackers/
└── opencv-yolo/
    └── MOT17-train/
        ├── MOT17-02-DPM.txt
        ├── MOT17-04-DPM.txt
        └── ...
```

### Running TrackEval

1. Clone TrackEval repository:
```bash
git clone https://github.com/JonathonLuiten/TrackEval.git
cd TrackEval
```

2. Copy your results:
```bash
mkdir -p data/trackers/mot_challenge/MOT17-train/opencv-yolo/data
cp /path/to/opencv-yolo/output/*.txt data/trackers/mot_challenge/MOT17-train/opencv-yolo/data/
```

3. Run evaluation:
```bash
python scripts/run_mot_challenge.py \
    --BENCHMARK MOT17 \
    --SPLIT_TO_EVAL train \
    --TRACKERS_TO_EVAL opencv-yolo \
    --METRICS HOTA CLEAR Identity \
    --USE_PARALLEL False \
    --NUM_PARALLEL_CORES 1
```

## Available MOT17 Training Sequences

- MOT17-02-DPM (600 frames, 1920×1080, 30 FPS)
- MOT17-04-DPM (1050 frames)
- MOT17-05-DPM (837 frames)
- MOT17-09-DPM (525 frames)
- MOT17-10-DPM (654 frames)
- MOT17-11-DPM (900 frames)
- MOT17-13-DPM (750 frames)

Each sequence has three detector variants: DPM, FRCNN, SDP.

## Tracking Settings

Configure tracking behavior via `ImpressionSettings`:

```cpp
ImpressionSettings settings;
settings.usePredict = true;          // Use Kalman predictions when detection fails
settings.predictFrameLimit = 20;     // Max frames to track without detection
settings.printTrajectory = true;     // Display object paths
settings.printCenter = true;         // Display center points
impression.setSettingsParams(settings);
```

## Performance Notes

- **GPU Acceleration**: Set `runOnGPU = true` for significant speedup
- **Visualization**: Disable `showVisualization` for batch processing
- **Frame Processing**: ~30-100 ms/frame depending on GPU/CPU
- **Expected FPS**: 10-30 FPS for real-time processing

## Troubleshooting

### "Sequence path does not exist"
Verify the MOT17-Data symbolic link exists:
```bash
ls -la MOT17-Data
```

### "Cannot open seqinfo.ini"
Check that the sequence directory contains required files:
```bash
ls MOT17-Data/MOT17/train/MOT17-02-DPM/
```

### "Failed to load frame"
Verify image files exist:
```bash
ls MOT17-Data/MOT17/train/MOT17-02-DPM/img1/ | head -5
```

## MOT17 Detector Types (DPM, FRCNN, SDP)

Each MOT17 sequence has 3 detector variants with different detection qualities:

| Detector | Quality | Speed | Recall | Precision | Use Case |
|----------|---------|-------|--------|-----------|----------|
| **DPM** | ⭐ Low | ⚡⚡⚡ Fast | Low | Low | Baseline, challenging test |
| **FRCNN** | ⭐⭐⭐ Good | ⚡⚡ Medium | Good | Good | General purpose |
| **SDP** | ⭐⭐⭐⭐ Best | ⚡ Slow | High | High | Best results |

**When using Tracker-Only mode**, test with all 3 detectors to see how your tracker performs across different detection qualities:

```cpp
// Compare tracker performance
processMOT17SequenceTrackerOnly("MOT17-02-DPM", "output/MOT17-02-DPM.txt");
processMOT17SequenceTrackerOnly("MOT17-02-FRCNN", "output/MOT17-02-FRCNN.txt");
processMOT17SequenceTrackerOnly("MOT17-02-SDP", "output/MOT17-02-SDP.txt");
```

Expected tracker performance:
- DPM: Lower scores (poor detections)
- FRCNN: Medium scores
- SDP: Highest scores (best detections)

**When using YOLO mode**, you're creating your own detections, so detector type doesn't matter (just use any sequence like MOT17-02-DPM for frames).

## Performance Comparison

| Mode | Detection | Tracking | Speed | Use Case |
|------|-----------|----------|-------|----------|
| **YOLO + Tracking** | YOLO inference | KalmanIoU | ~50ms/frame | Test complete system |
| **Tracker-Only** | Read det.txt | KalmanIoU | ~5ms/frame | Test tracker algorithm |

**Recommendation**:
1. Start with Tracker-Only mode for fast iteration on tracking algorithm
2. Once tracker is good, test with YOLO mode for end-to-end evaluation

## Files

- [mot_sequence_reader.h](mot_sequence_reader.h) - Frame sequence reader
- [mot_sequence_reader.cpp](mot_sequence_reader.cpp) - Implementation
- [mot_detection_reader.h](mot_detection_reader.h) - Detection file reader (NEW!)
- [mot_detection_reader.cpp](mot_detection_reader.cpp) - Implementation (NEW!)
- [TrackerTelemtryLogger/MotChallengeLogger.h](TrackerTelemtryLogger/MotChallengeLogger.h) - Output logger
- [TrackerTelemtryLogger/MotChallengeLogger.cpp](TrackerTelemtryLogger/MotChallengeLogger.cpp) - Logger implementation
- [main.cpp](main.cpp) - Processing examples (lines 138-268)

## API Reference

### MotSequenceReader
```cpp
MotSequenceReader reader("MOT17-Data/MOT17/train/MOT17-02-DPM");
while (reader.nextFrame(frame)) {
    // Process frame
}
```

### MotDetectionReader (NEW!)
```cpp
MotDetectionReader detReader("MOT17-02-DPM/det/det.txt");
auto detections = detReader.getDetectionsForFrame(frameIndex);
// Returns vector<IObjectTracker::Detection>
```

### processMOT17Sequence
```cpp
processMOT17Sequence(
    sequencePath,    // Path to sequence directory
    outputPath,      // Output file path
    modelPath,       // YOLO model path
    showViz,         // Visualization on/off
    runOnGPU         // GPU acceleration
);
```

### processMOT17SequenceTrackerOnly (NEW!)
```cpp
processMOT17SequenceTrackerOnly(
    sequencePath,    // Path to sequence directory
    outputPath,      // Output file path
    showViz          // Visualization on/off
);
```

## References

- [MOTChallenge Dataset](https://motchallenge.net/)
- [TrackEval Repository](https://github.com/JonathonLuiten/TrackEval)
- [MOT17 Format Specification](https://arxiv.org/abs/1603.00831)
- [HOTA Metrics Paper](https://link.springer.com/article/10.1007/s11263-020-01375-2)
