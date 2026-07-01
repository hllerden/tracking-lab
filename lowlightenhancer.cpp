#include "lowlightenhancer.h"
#include <numeric>
#include <opencv2/ximgproc/edge_filter.hpp>
#include <algorithm>

// Constructor
LowLightEnhancer::LowLightEnhancer(float tmin, int kernelSize, float alpha, float omega, float p, double eps)
    : tmin(tmin), kernelSize(kernelSize), alpha(alpha), omega(omega), p(p), eps(eps) {}

// Main enhancement method
cv::Mat LowLightEnhancer::enhance(const cv::Mat& inputImage) {
    cv::Mat floatImage;
    inputImage.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);

    auto [darkChannel, brightChannel] = getIlluminationChannel(floatImage, kernelSize);
    cv::Mat atmosphere = getAtmosphere(floatImage, brightChannel, p);
    cv::Mat initialTransmission = getInitialTransmission(atmosphere, brightChannel);

    if (tmin > 0.0f) {
        initialTransmission = reduceInitialTransmission(initialTransmission);
    }

    cv::Mat correctedTransmission = getCorrectedTransmission(floatImage, atmosphere, darkChannel, brightChannel, initialTransmission);
    cv::Mat refinedTransmission;
    cv::ximgproc::guidedFilter(floatImage, correctedTransmission, refinedTransmission, kernelSize, eps);

    return getFinalImage(floatImage, atmosphere, refinedTransmission, tmin);
}

// Illumination channel calculation
std::pair<cv::Mat, cv::Mat> LowLightEnhancer::getIlluminationChannel(const cv::Mat& image, float w) {
    int N = image.rows;
    int M = image.cols;
    cv::Mat darkch = cv::Mat::zeros(cv::Size(M, N), CV_32FC1);
    cv::Mat brightch = cv::Mat::zeros(cv::Size(M, N), CV_32FC1);

    int padding = int(w / 2);
    cv::Mat padded;
    cv::copyMakeBorder(image, padded, padding, padding, padding, padding, cv::BORDER_REPLICATE);

    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < M; ++x) {
            cv::Rect region(x, y, w, w);
            cv::Mat patch = padded(region);
            double minVal, maxVal;
            cv::minMaxLoc(patch, &minVal, &maxVal, nullptr, nullptr);
            darkch.at<float>(y, x) = static_cast<float>(minVal);
            brightch.at<float>(y, x) = static_cast<float>(maxVal);
        }
    }

    return {darkch, brightch};
}

// Atmospheric light estimation
cv::Mat LowLightEnhancer::getAtmosphere(const cv::Mat& image, const cv::Mat& brightChannel, float p) {
    int totalPixels = brightChannel.rows * brightChannel.cols;
    int numTopPixels = static_cast<int>(totalPixels * p);

    cv::Mat flatBright = brightChannel.reshape(1, totalPixels);
    std::vector<int> indices(totalPixels);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + numTopPixels, indices.end(),
                      [&](int a, int b) { return flatBright.at<float>(a) > flatBright.at<float>(b); });

    cv::Vec3f atmosphere(0, 0, 0);
    for (int i = 0; i < numTopPixels; ++i) {
        int idx = indices[i];
        int y = idx / brightChannel.cols;
        int x = idx % brightChannel.cols;
        atmosphere += image.at<cv::Vec3f>(y, x);
    }

    return cv::Mat(1, 1, CV_32FC3, atmosphere / static_cast<float>(numTopPixels));
}

// Initial transmission map
cv::Mat LowLightEnhancer::getInitialTransmission(const cv::Mat& atmosphere, const cv::Mat& brightChannel) {
    cv::Vec3f atmosphereVec = atmosphere.at<cv::Vec3f>(0, 0);
    float maxAtmosphere = std::max({atmosphereVec[0], atmosphereVec[1], atmosphereVec[2]});

    cv::Mat initTransmission = (brightChannel - maxAtmosphere) / (1.0f - maxAtmosphere);
    cv::normalize(initTransmission, initTransmission, 0.0, 1.0, cv::NORM_MINMAX);
    return initTransmission;
}

// Corrected transmission map
cv::Mat LowLightEnhancer::getCorrectedTransmission(const cv::Mat& image, const cv::Mat& atmosphere, const cv::Mat& darkChannel,
                                                   const cv::Mat& brightChannel, const cv::Mat& initialTransmission) {
    cv::Vec3f atmosphereVec = atmosphere.at<cv::Vec3f>(0, 0);
    cv::Mat correctedTransmission = initialTransmission.clone();

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            float darkValue = darkChannel.at<float>(y, x);
            float brightValue = brightChannel.at<float>(y, x);
            if (brightValue - darkValue < alpha) {
                correctedTransmission.at<float>(y, x) *= (1 - omega * darkValue);
            }
        }
    }

    return correctedTransmission;
}

// Final enhanced image
cv::Mat LowLightEnhancer::getFinalImage(const cv::Mat& image, const cv::Mat& atmosphere, const cv::Mat& refinedTransmission, float tmin) {
    cv::Vec3f atmosphereVec = atmosphere.at<cv::Vec3f>(0, 0);
    cv::Mat J(image.size(), CV_32FC3);

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            cv::Vec3f pixel = image.at<cv::Vec3f>(y, x);
            float t = std::max(refinedTransmission.at<float>(y, x), tmin);
            for (int c = 0; c < 3; ++c) {
                J.at<cv::Vec3f>(y, x)[c] = (pixel[c] - atmosphereVec[c]) / t + atmosphereVec[c];
            }
        }
    }

    cv::normalize(J, J, 0.0, 1.0, cv::NORM_MINMAX);
    return J;
}
cv::Mat LowLightEnhancer::reduceInitialTransmission(const cv::Mat& initTransmission) {
    cv::Mat mod_init_t(initTransmission.size(), CV_8UC1);

    for (int i = 0; i < initTransmission.rows; ++i) {
        for (int j = 0; j < initTransmission.cols; ++j) {
            mod_init_t.at<uchar>(i, j) = std::min(static_cast<int>(initTransmission.at<float>(i, j) * 255), 255);
        }
    }

    int x[3] = {0, 32, 255};
    int f[3] = {0, 32, 48};

    // Creating array [0,...,255]
    cv::Mat table(cv::Size(1, 256), CV_8UC1);

    // Linear Interpolation
    int l = 0;
    for (int k = 0; k < 256; ++k) {
        if (k > x[l + 1]) {
            l = l + 1;
        }

        float m = static_cast<float>(f[l + 1] - f[l]) / (x[l + 1] - x[l]);
        table.at<uchar>(k, 0) = static_cast<uchar>(f[l] + m * (k - x[l]));
    }

    // Lookup table
    cv::LUT(mod_init_t, table, mod_init_t);

    cv::Mat reducedTransmission = initTransmission.clone();
    for (int i = 0; i < reducedTransmission.rows; ++i) {
        for (int j = 0; j < reducedTransmission.cols; ++j) {
            reducedTransmission.at<float>(i, j) = static_cast<float>(mod_init_t.at<uchar>(i, j)) / 255.0f;
        }
    }

    return reducedTransmission;
}
