/**
 * @brief Simple test program for MotSequenceReader
 *
 * This program tests the MOT sequence reader with a small sequence
 * to verify that frames can be read correctly.
 */

#include <iostream>
#include "mot/mot_sequence_reader.h"
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    std::string sequencePath = std::string(PROJECT_ROOT_DIR) + "/MOT17-Data/MOT17/train/MOT17-02-DPM";

    if (argc > 1) {
        sequencePath = argv[1];
    }

    std::cout << "=== MOT Sequence Reader Test ===" << std::endl;
    std::cout << "Sequence path: " << sequencePath << std::endl;

    try {
        MotSequenceReader reader(sequencePath);

        std::cout << "\n=== Sequence Information ===" << std::endl;
        std::cout << "Name: " << reader.getSequenceName() << std::endl;
        std::cout << "Total frames: " << reader.getTotalFrames() << std::endl;
        std::cout << "FPS: " << reader.getFPS() << std::endl;
        std::cout << "Frame size: " << reader.getFrameSize() << std::endl;

        std::cout << "\n=== Reading first 10 frames ===" << std::endl;
        cv::Mat frame;
        int count = 0;

        while (reader.hasNext() && count < 10) {
            if (!reader.nextFrame(frame)) {
                std::cerr << "Failed to read frame " << count << std::endl;
                break;
            }

            std::cout << "Frame " << count << ": "
                     << frame.cols << "x" << frame.rows
                     << " (type: " << frame.type() << ")" << std::endl;

            // Display frame
            cv::imshow("MOT Frame", frame);
            cv::waitKey(100); // 100ms delay

            count++;
        }

        std::cout << "\n=== Test Complete ===" << std::endl;
        std::cout << "Successfully read " << count << " frames" << std::endl;

        cv::waitKey(0);
        cv::destroyAllWindows();

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}
