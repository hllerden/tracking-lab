#ifndef LOWLIGHTENHANCER_H
#define LOWLIGHTENHANCER_H

#include <opencv2/opencv.hpp>
#include <utility>
#include <iostream>
#include <string>
class LowLightEnhancer {
public:

    LowLightEnhancer(float tmin = 0.1, int kernelSize = 15, float alpha = 0.4, float omega = 0.75, float p = 0.1, double eps = 1e-3);
    // Enhance the image
    cv::Mat enhance(const cv::Mat& inputImage);

private:
    float tmin;
    int kernelSize;
    float alpha;
    float omega;
    float p;
    double eps;

    std::pair<cv::Mat, cv::Mat> getIlluminationChannel(const cv::Mat& image, float w);
    cv::Mat getAtmosphere(const cv::Mat& image, const cv::Mat& brightChannel, float p);
    cv::Mat getInitialTransmission(const cv::Mat& atmosphere, const cv::Mat& brightChannel);
    cv::Mat reduceInitialTransmission(const cv::Mat& initTransmission);
    cv::Mat getCorrectedTransmission(const cv::Mat& image, const cv::Mat& atmosphere, const cv::Mat& darkChannel,
                                     const cv::Mat& brightChannel, const cv::Mat& initialTransmission);
    cv::Mat getFinalImage(const cv::Mat& image, const cv::Mat& atmosphere, const cv::Mat& refinedTransmission, float tmin);
};

#endif // LOWLIGHTENHANCER_H
