#include "mot_sequence_reader.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

MotSequenceReader::MotSequenceReader(const std::string& sequencePath)
    : sequencePath_(sequencePath)
    , currentFrameIndex_(0)
    , totalFrames_(0)
    , frameRate_(30.0)
    , groundTruthLoaded_(false) {

    // Verify sequence path exists
    if (!fs::exists(sequencePath_)) {
        throw std::runtime_error("Sequence path does not exist: " + sequencePath_);
    }

    // Parse seqinfo.ini
    auto seqInfo = parseSeqInfo();

    // Extract metadata
    sequenceName_ = seqInfo["name"];
    imageDir_ = seqInfo["imDir"];
    imageExt_ = seqInfo["imExt"];
    totalFrames_ = std::stoi(seqInfo["seqLength"]);
    frameRate_ = std::stod(seqInfo["frameRate"]);

    int width = std::stoi(seqInfo["imWidth"]);
    int height = std::stoi(seqInfo["imHeight"]);
    frameSize_ = cv::Size(width, height);

    // Verify image directory exists
    std::string imgPath = sequencePath_ + "/" + imageDir_;
    if (!fs::exists(imgPath)) {
        throw std::runtime_error("Image directory does not exist: " + imgPath);
    }

    // Optionally load ground truth
    loadGroundTruth();
}

bool MotSequenceReader::nextFrame(cv::Mat& frame) {
    if (!hasNext()) {
        return false;
    }

    std::string framePath = getFramePath(currentFrameIndex_);
    frame = cv::imread(framePath);

    if (frame.empty()) {
        throw std::runtime_error("Failed to load frame: " + framePath);
    }

    currentFrameIndex_++;
    return true;
}

bool MotSequenceReader::hasNext() const {
    return currentFrameIndex_ < totalFrames_;
}

int MotSequenceReader::getCurrentFrameIndex() const {
    return currentFrameIndex_;
}

int MotSequenceReader::getTotalFrames() const {
    return totalFrames_;
}

double MotSequenceReader::getFPS() const {
    return frameRate_;
}

cv::Size MotSequenceReader::getFrameSize() const {
    return frameSize_;
}

std::string MotSequenceReader::getSequenceName() const {
    return sequenceName_;
}

void MotSequenceReader::reset() {
    currentFrameIndex_ = 0;
}

std::vector<MotSequenceReader::GroundTruthBox>
MotSequenceReader::getGroundTruthForFrame(int frameIndex) const {
    auto it = groundTruth_.find(frameIndex);
    if (it != groundTruth_.end()) {
        return it->second;
    }
    return {};
}

std::map<std::string, std::string> MotSequenceReader::parseSeqInfo() {
    std::string seqInfoPath = sequencePath_ + "/seqinfo.ini";
    std::ifstream file(seqInfoPath);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open seqinfo.ini: " + seqInfoPath);
    }

    std::map<std::string, std::string> seqInfo;
    std::string line;
    bool inSequenceSection = false;

    while (std::getline(file, line)) {
        // Trim whitespace
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        // Skip empty lines and comments
        if (line.empty() || line[0] == ';' || line[0] == '#') {
            continue;
        }

        // Check for [Sequence] section
        if (line == "[Sequence]") {
            inSequenceSection = true;
            continue;
        }

        // Parse key=value pairs in [Sequence] section
        if (inSequenceSection) {
            size_t equalPos = line.find('=');
            if (equalPos != std::string::npos) {
                std::string key = line.substr(0, equalPos);
                std::string value = line.substr(equalPos + 1);

                // Trim key and value
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);

                seqInfo[key] = value;
            }
        }
    }

    file.close();

    // Validate required fields
    std::vector<std::string> requiredFields = {
        "name", "imDir", "frameRate", "seqLength", "imWidth", "imHeight", "imExt"
    };

    for (const auto& field : requiredFields) {
        if (seqInfo.find(field) == seqInfo.end()) {
            throw std::runtime_error("Missing required field in seqinfo.ini: " + field);
        }
    }

    return seqInfo;
}

void MotSequenceReader::loadGroundTruth() {
    std::string gtPath = sequencePath_ + "/gt/gt.txt";

    if (!fs::exists(gtPath)) {
        // Ground truth not available (e.g., test sequences)
        groundTruthLoaded_ = false;
        return;
    }

    std::ifstream file(gtPath);
    if (!file.is_open()) {
        groundTruthLoaded_ = false;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;

        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }

        if (tokens.size() < 10) {
            continue; // Invalid line
        }

        // MOT format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        int frameIndex = std::stoi(tokens[0]);  // 1-based
        int trackId = std::stoi(tokens[1]);
        float x = std::stof(tokens[2]);
        float y = std::stof(tokens[3]);
        float w = std::stof(tokens[4]);
        float h = std::stof(tokens[5]);
        float conf = std::stof(tokens[6]);
        int classId = std::stoi(tokens[7]);

        GroundTruthBox box;
        box.trackId = trackId;
        box.bbox = cv::Rect2f(x, y, w, h);
        box.confidence = conf;
        box.classId = classId;

        groundTruth_[frameIndex].push_back(box);
    }

    file.close();
    groundTruthLoaded_ = true;
}

std::string MotSequenceReader::getFramePath(int frameIndex) const {
    // MOT format uses 1-based indexing for frames, but we use 0-based internally
    int frameNumber = frameIndex + 1;

    std::ostringstream oss;
    oss << sequencePath_ << "/" << imageDir_ << "/"
        << std::setw(6) << std::setfill('0') << frameNumber
        << imageExt_;

    return oss.str();
}
