# tracking-lab

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A C++ real-time object detection and multi-object tracking system. Started this as a learning project to understand how tracking pipelines actually work under the hood, ended up building something I actually use for benchmarking and testing new ideas.

It runs YOLO models for detection and handles the tracking side with a custom Kalman filter + Hungarian algorithm. Nothing fancy on the surface but there's a lot going on inside.

---

## Demo

### Detections

<video src="https://github.com/hllerden/tracking-lab/raw/refs/heads/master/readme_tracking_detections.mp4" width="540" controls></video>

[Watch raw video](https://github.com/hllerden/tracking-lab/raw/refs/heads/master/readme_tracking_detections.mp4)

### Full Debug

<video src="https://github.com/hllerden/tracking-lab/raw/refs/heads/master/readme_tracking_full_debug.mp4" width="540" controls></video>

[Watch raw video](https://github.com/hllerden/tracking-lab/raw/refs/heads/master/readme_tracking_full_debug.mp4)

### Output

<video src="https://github.com/hllerden/tracking-lab/raw/refs/heads/master/readme_tracking_tracker_output.mp4" width="540" controls></video>

[Watch raw video](https://github.com/hllerden/tracking-lab/raw/refs/heads/master/readme_tracking_tracker_output.mp4)

---

## What it does

- YOLO inference via OpenCV DNN (supports v5, v8, v9, v10, v11, v26 in ONNX format)
- Custom Kalman-based tracker with multiple IoU variants for assignment
- Re-identification (ReID) support to recover lost tracks
- Full MOT17 benchmark intregration for proper evaluation
- Headless benchmark mode - no visualization, just raw numbers
- GPU acceleration with CUDA (highly recommended)

The core idea is simple: detect objects, predict where they'll be next frame using Kalman, match predictions to detections with Hungarian, repeat. The IoU variants (GIOU, DIOU, CIOU, SIOU, AIOU) are there so you can experiment with what works best for a given scenario.

---

## Dependencies

- OpenCV 5.x (with DNN and CUDA modules)
- Eigen3 3.3+
- CUDA 11+ (optional but makes a huge difference)

---

## Build

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

Executable ends up in `build/yoloExercise`. For benchmark mode:

```bash
cd build/benchmark
./mot17_benchmark
```

---

## IoU Types

The tracker supports 6 different matching metrics. Default is CIOU.

| Type | Notes |
|------|-------|
| IOU  | standard, works fine for most cases |
| GIOU | better with low overlap |
| DIOU | penalizes center distance |
| CIOU | also considers aspect ratio, generally best |
| SIOU | smoothed for low overlap situations |
| AIOU | tunable sensitivity via alpha |

---

## Benchmark Results - MOT17-04-FRCNN

Evaluated on MOT17 train set using FRCNN pre-computed detections. TrackEval (HOTA + CLEAR + Identity). Sequence: 1050 frames, 83 unique pedestrian identities.

### HOTA

| Metric | Tracker | Tracker + ReID |
|--------|--------:|---------------:|
| HOTA | 57.236 | 57.258 |
| DetA | 51.945 | 51.718 |
| AssA | 63.238 | 63.549 |
| DetRe | 55.652 | 55.704 |
| DetPr | 85.137 | 84.421 |
| AssRe | 68.215 | 68.608 |
| AssPr | 84.099 | 85.146 |
| LocA | 89.676 | 89.697 |
| OWTA | 59.310 | 59.486 |
| HOTA(0) | 66.781 | 67.012 |
| LocA(0) | 84.938 | 84.798 |
| HOTALocA(0) | 56.723 | 56.824 |

### CLEAR

| Metric | Tracker | Tracker + ReID |
|--------|--------:|---------------:|
| MOTA | 54.087 | 53.645 |
| MOTP | 89.188 | 89.166 |
| MODA | 54.143 | 53.704 |
| CLR_Re | 59.756 | 59.844 |
| CLR_Pr | 91.414 | 90.695 |
| MTR | 32.530 | 32.530 |
| PTR | 38.554 | 38.554 |
| MLR | 28.916 | 28.916 |
| sMOTA | 47.626 | 47.162 |
| CLR_TP | 28418 | 28460 |
| CLR_FN | 19139 | 19097 |
| CLR_FP | 2669 | 2920 |
| IDSW | 27 | 28 |
| MT | 27 | 27 |
| PT | 32 | 32 |
| ML | 24 | 24 |
| Frag | 71 | 68 |

### Identity

| Metric | Tracker | Tracker + ReID |
|--------|--------:|---------------:|
| IDF1 | 65.129 | 65.566 |
| IDR | 53.851 | 54.415 |
| IDP | 82.382 | 82.467 |
| IDTP | 25610 | 25878 |
| IDFN | 21947 | 21679 |
| IDFP | 5477 | 5502 |

### Count

| Metric | Tracker | Tracker + ReID |
|--------|--------:|---------------:|
| Dets | 31087 | 31380 |
| GT_Dets | 47557 | 47557 |
| IDs | 79 | 80 |
| GT_IDs | 83 | 83 |

Honest take: the difference on MOT17-04 is tiny. This sequence is a static camera with predictable pedestrian paths so Kalman + IoU already handles reassociation well. ReID didn't reduce ID switches here (actually went up by 1), the main gain is marginal AssA improvement (+0.3). For sequences with heavy occulsion and crowd scenarios ReID should make a bigger difference - this was just the sequence I had results for at the time.

---

## Project Structure

```
tracking-lab/
├── trackers/
│   ├── kalman_iou/          # main tracker implementation
│   └── reid_extractor.*     # ReID feature extraction
├── benchmark/               # MOT17 benchmark system
├── thirdParty/
│   ├── TrackEval/           # evaluation toolkit
│   └── deepsort/            # alternative tracker (not active)
├── models/                  # YOLO .onnx files go here
├── output/                  # benchmark results and reports
│   └── reports/             # TrackEval output per version
├── scripts/
│   ├── compare_versions.sh
│   └── compare_versions_minimal/   # Python comparison tool
├── impression.*             # high-level detection+tracking API
├── inference.*              # YOLO inference wrapper
└── CMakeLists.txt
```

---

## Running Benchmarks

```bash
# Run full benchmark (uses FRCNN pre-computed detections, fast)
cd build/benchmark && ./mot17_benchmark

# Evaluate with TrackEval
cd output && ./run_trackeval.sh --minimal

# Compare two versions
python3 -m scripts.compare_versions_minimal v1.0.9-minimal v1.0.10-minimal \
    --groups primary detection counts \
    --output output/comparison/ --html --json
```

The versioned output system (`output/v1.0.x/`) lets you track progress across code changes. The comparison tool generates HTML reports with charts.

---

## Notes

- Place YOLO model files in `models/` directory (`.onnx` format)
- MOT17 dataset should be at `MOT17-Data/MOT17/train/`
- Default IoU type is CIOU but you can change it per run in the benchmark config
- The 6-state Kalman filter uses constant velocity model (x, y, vx, vy, w, h)
- Hungarian algorithm is O(n³) so if you're tracking hundreds of objects per frame you'll feel it

Current version: 1.0.10

## Author
---
### ***Halil Erden***

halilxerden@gmail.com 

**Embedded Software Engineer** | C++ / Qt / Linux / Yocto / Halı / Kilim / Time travel
