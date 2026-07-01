#include "i_visual_tracker.h"
#include "mot/mot_detection_reader.h"
#include "mot/mot_sequence_reader.h"
#include "trackers/tracker_factory.h"

#include <opencv2/opencv.hpp>

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Options {
    std::string sequencePath = std::string(PROJECT_ROOT_DIR) + "/MOT17-Data/MOT17/train/MOT17-04-FRCNN";
    std::string outputVideo = std::string(PROJECT_ROOT_DIR) + "/output/readme_tracking.mp4";
    int startFrame = 1;
    int endFrame = 520;
    int maxRenderedFrames = 0;
    int stride = 1;
    int fps = 0;
    int panelWidth = 0;
    bool showWindow = false;
};

std::string quoteShell(const std::string& value) {
    std::string quoted = "'";
    for (char c : value) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

void printUsage(const char* exe) {
    std::cout
        << "Usage: " << exe << " [options]\n\n"
        << "Options:\n"
        << "  --sequence <dir>       MOT sequence dir (default: MOT17-04-FRCNN)\n"
        << "  --output <mp4>         Output MP4 path (default: output/readme_tracking.mp4)\n"
        << "  --start-frame <n>      First frame to render, tracker is warmed up before it (default: 1)\n"
        << "  --end-frame <n>        Last processed frame to include (default: 520)\n"
        << "  --max-frames <n>       Rendered frame cap, 0 means no cap (default: 0)\n"
        << "  --stride <n>           Render one frame every n processed frames (default: 1)\n"
        << "  --fps <n>              MP4 frame rate, 0 uses sequence FPS (default: 0)\n"
        << "  --panel-width <px>     Width of small left panels (default: source width / 2)\n"
        << "  --show-window          Show the two-column preview with cv::imshow while rendering\n"
        << "  --help                 Show this help\n";
}

int parseInt(const std::string& value, const std::string& name) {
    try {
        return std::stoi(value);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid integer for " + name + ": " + value);
    }
}

Options parseArgs(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto requireValue = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
            return argv[++i];
        };

        if (arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (arg == "--sequence") {
            options.sequencePath = requireValue(arg);
        } else if (arg == "--output") {
            options.outputVideo = requireValue(arg);
        } else if (arg == "--start-frame") {
            options.startFrame = parseInt(requireValue(arg), arg);
        } else if (arg == "--end-frame") {
            options.endFrame = parseInt(requireValue(arg), arg);
        } else if (arg == "--max-frames") {
            options.maxRenderedFrames = parseInt(requireValue(arg), arg);
        } else if (arg == "--stride") {
            options.stride = parseInt(requireValue(arg), arg);
        } else if (arg == "--fps") {
            options.fps = parseInt(requireValue(arg), arg);
        } else if (arg == "--panel-width") {
            options.panelWidth = parseInt(requireValue(arg), arg);
        } else if (arg == "--show-window") {
            options.showWindow = true;
        } else {
            throw std::runtime_error("Unknown option: " + arg);
        }
    }

    if (options.startFrame < 1 || options.endFrame < options.startFrame ||
        options.maxRenderedFrames < 0 || options.stride < 1 ||
        options.fps < 0 || options.panelWidth < 0) {
        throw std::runtime_error("Invalid options. Frame range, stride, FPS and sizes must be valid.");
    }
    return options;
}

cv::Mat resizeToPanel(const cv::Mat& frame, int panelWidth) {
    const double scale = static_cast<double>(panelWidth) / static_cast<double>(frame.cols);
    const int panelHeight = std::max(1, static_cast<int>(std::round(frame.rows * scale)));

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(panelWidth, panelHeight), 0.0, 0.0, cv::INTER_AREA);
    return resized;
}

cv::Mat resizeToHeight(const cv::Mat& frame, int panelHeight) {
    const double scale = static_cast<double>(panelHeight) / static_cast<double>(frame.rows);
    const int panelWidth = std::max(1, static_cast<int>(std::round(frame.cols * scale)));

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(panelWidth, panelHeight), 0.0, 0.0, cv::INTER_AREA);
    return resized;
}

void drawDetections(cv::Mat& frame, const std::vector<IObjectTracker::Detection>& detections) {
    const cv::Scalar boxColor(210, 210, 210);
    const cv::Scalar textColor(245, 245, 245);
    const cv::Scalar textShadow(20, 20, 20);

    for (const auto& detection : detections) {
        const cv::Rect box = detection.bbox & cv::Rect(0, 0, frame.cols, frame.rows);
        if (box.empty()) {
            continue;
        }
        cv::rectangle(frame, box, boxColor, 3);

        std::ostringstream label;
        label << std::fixed << std::setprecision(2) << detection.confidence;
        const cv::Point textPos(box.x, std::max(24, box.y - 6));
        cv::putText(frame, label.str(), textPos, cv::FONT_HERSHEY_SIMPLEX,
                    0.78, textShadow, 5, cv::LINE_AA);
        cv::putText(frame, label.str(), textPos, cv::FONT_HERSHEY_SIMPLEX,
                    0.78, textColor, 2, cv::LINE_AA);
    }
}

void drawColumnTitle(cv::Mat& panel, const std::string& title) {
    constexpr int titleHeight = 46;
    cv::rectangle(panel, cv::Rect(0, 0, panel.cols, titleHeight), cv::Scalar(18, 18, 18), cv::FILLED);
    cv::putText(panel, title, cv::Point(14, 32), cv::FONT_HERSHEY_SIMPLEX, 0.92,
                cv::Scalar(0, 0, 0), 5, cv::LINE_AA);
    cv::putText(panel, title, cv::Point(14, 32), cv::FONT_HERSHEY_SIMPLEX, 0.92,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
}

cv::Mat makeCanvas(const cv::Mat& detectionsPanel,
                   const cv::Mat& debugPanel,
                   const cv::Mat& cleanPanel,
                   int frameIndex) {
    cv::Mat det = detectionsPanel;
    cv::Mat clean = cleanPanel;

    if (det.cols != clean.cols) {
        const int maxWidth = std::max(det.cols, clean.cols);
        if (det.cols < maxWidth) {
            cv::copyMakeBorder(det, det, 0, 0, 0, maxWidth - det.cols,
                               cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        }
        if (clean.cols < maxWidth) {
            cv::copyMakeBorder(clean, clean, 0, 0, 0, maxWidth - clean.cols,
                               cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        }
    }

    cv::Mat leftColumn;
    cv::vconcat(std::vector<cv::Mat>{det, clean}, leftColumn);

    cv::Mat debug = debugPanel;
    if (debug.rows != leftColumn.rows) {
        debug = resizeToHeight(debug, leftColumn.rows);
    }
    if (debug.rows < leftColumn.rows) {
        cv::copyMakeBorder(debug, debug, 0, leftColumn.rows - debug.rows, 0, 0,
                           cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    } else if (leftColumn.rows < debug.rows) {
        cv::copyMakeBorder(leftColumn, leftColumn, 0, debug.rows - leftColumn.rows, 0, 0,
                           cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    }

    cv::Mat canvas;
    cv::hconcat(std::vector<cv::Mat>{leftColumn, debug}, canvas);
    const std::string frameText = "Frame " + std::to_string(frameIndex);
    int baseline = 0;
    const cv::Size frameTextSize = cv::getTextSize(frameText, cv::FONT_HERSHEY_SIMPLEX,
                                                   0.72, 2, &baseline);
    const cv::Point frameTextPos(canvas.cols - frameTextSize.width - 16, canvas.rows - 14);
    cv::putText(canvas, frameText, frameTextPos, cv::FONT_HERSHEY_SIMPLEX,
                0.72, cv::Scalar(0, 0, 0), 5, cv::LINE_AA);
    cv::putText(canvas, frameText, frameTextPos, cv::FONT_HERSHEY_SIMPLEX,
                0.72, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    return canvas;
}

bool commandOk(const std::string& command) {
    return std::system(command.c_str()) == 0;
}

bool hasFfmpeg() {
    return commandOk("ffmpeg -version >/dev/null 2>&1");
}

bool encodeMp4(const fs::path& framesDir, const fs::path& outputVideo, int fps) {
    if (!hasFfmpeg()) {
        std::cout << "ffmpeg not found. PNG frames are available at: " << framesDir << "\n";
        return false;
    }

    if (!outputVideo.parent_path().empty()) {
        fs::create_directories(outputVideo.parent_path());
    }

    const std::string inputPattern = (framesDir / "frame_%05d.png").string();
    const std::string mp4Cmd =
        "ffmpeg -y -v warning -framerate " + std::to_string(fps) +
        " -i " + quoteShell(inputPattern) +
        " -c:v libx264 -preset slow -crf 23 -pix_fmt yuv420p -movflags +faststart " +
        quoteShell(outputVideo.string());

    if (!commandOk(mp4Cmd)) {
        return false;
    }

    const auto size = fs::file_size(outputVideo);
    std::cout << "MP4 written: " << outputVideo << " (" << (size / 1024) << " KB)\n";
    return true;
}

int main(int argc, char** argv) {
    try {
        const Options options = parseArgs(argc, argv);
        const fs::path outputVideo(options.outputVideo);
        if (!outputVideo.parent_path().empty()) {
            fs::create_directories(outputVideo.parent_path());
        }
        const fs::path framesBaseDir = outputVideo.parent_path().empty() ? fs::path(".") : outputVideo.parent_path();
        const fs::path framesDir = framesBaseDir / (outputVideo.stem().string() + "_frames");

        fs::create_directories(framesDir);
        for (const auto& entry : fs::directory_iterator(framesDir)) {
            if (entry.path().extension() == ".png") {
                fs::remove(entry.path());
            }
        }

        MotSequenceReader reader(options.sequencePath);
        MotDetectionReader detReader(options.sequencePath + "/det/det.txt");
        const int panelWidth = options.panelWidth > 0 ? options.panelWidth : reader.getFrameSize().width / 2;
        const int outputFps = options.fps > 0 ? options.fps : std::max(1, static_cast<int>(std::round(reader.getFPS())));

        KalmanIoUConfig trackerConfig = TrackerFactory::getDefaultConfig(TrackerType::BYTETRACK);
        trackerConfig.cameraBounds = cv::Rect(0, 0, reader.getFrameSize().width, reader.getFrameSize().height);
        trackerConfig.removeOutOfBounds = true;
        trackerConfig.usePredictionInLost = true;
        trackerConfig.maxLostFrames = 30 * 2;
        trackerConfig.iouThreshold = 0.10f;
        trackerConfig.lowConfIouThreshold = 0.05f;
        trackerConfig.reidAlpha = 0.65f;
        trackerConfig.featureEmaDecay = 0.9f;
        trackerConfig.highConfThreshold = 0.55f;
        trackerConfig.lostStateThreshold = 3;

        auto tracker = TrackerFactory::create(TrackerType::BYTETRACK, trackerConfig);
        auto* visualTracker = dynamic_cast<IVisualTracker*>(tracker.get());
        if (!visualTracker) {
            throw std::runtime_error("Selected tracker does not implement IVisualTracker.");
        }
        visualTracker->setTrajectoryLength(30);

        IVisualTracker::VisualizationConfig fullDebug;
        fullDebug.showBoundingBox = true;
        fullDebug.showTrackId = true;
        fullDebug.showTrajectory = true;
        fullDebug.showVelocity = true;
        fullDebug.showMissedFrames = true;
        fullDebug.showLostTracks = true;
        fullDebug.showReidScore = true;

        IVisualTracker::VisualizationConfig clean;
        clean.showBoundingBox = true;
        clean.showTrackId = true;
        clean.showTrajectory = false;
        clean.showVelocity = false;
        clean.showMissedFrames = false;
        clean.showLostTracks = false;
        clean.showReidScore = false;

        cv::Mat frame;
        int processedFrame = 0;
        int renderedFrame = 0;

        if (options.showWindow) {
            cv::namedWindow("MOT tracking MP4 preview", cv::WINDOW_AUTOSIZE);
        }

        while (reader.nextFrame(frame) &&
               (options.maxRenderedFrames == 0 || renderedFrame < options.maxRenderedFrames)) {
            ++processedFrame;
            const auto detections = detReader.getDetectionsForFrame(processedFrame);
            tracker->update(detections);

            if (processedFrame > options.endFrame) {
                break;
            }

            const bool reachedStart = processedFrame >= options.startFrame;
            const bool renderThisFrame = ((processedFrame - options.startFrame) % options.stride) == 0 ||
                                         processedFrame == options.endFrame;
            if (!reachedStart || !renderThisFrame) {
                continue;
            }

            cv::Mat detView = frame.clone();
            cv::Mat debugView = frame.clone();
            cv::Mat cleanView = frame.clone();

            drawDetections(detView, detections);
            visualTracker->draw(debugView, fullDebug);
            visualTracker->draw(cleanView, clean);

            detView = resizeToPanel(detView, panelWidth);
            cleanView = resizeToPanel(cleanView, panelWidth);

            drawColumnTitle(detView, "DETECTIONS: det.txt");
            drawColumnTitle(debugView, "FULL DEBUG VIEW");
            drawColumnTitle(cleanView, "TRACKER OUTPUT");

            const cv::Mat canvas = makeCanvas(detView, debugView, cleanView, processedFrame);

            std::ostringstream filename;
            filename << "frame_" << std::setw(5) << std::setfill('0') << renderedFrame << ".png";
            cv::imwrite((framesDir / filename.str()).string(), canvas);

            if (options.showWindow) {
                cv::imshow("MOT tracking MP4 preview", canvas);
                if (cv::waitKey(1) == 27) {
                    break;
                }
            }

            ++renderedFrame;
        }

        if (renderedFrame == 0) {
            throw std::runtime_error("No frames were rendered. Check --start-frame and sequence path.");
        }

        std::cout << "Rendered " << renderedFrame << " PNG frames to " << framesDir << "\n";
        encodeMp4(framesDir, outputVideo, outputFps);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
