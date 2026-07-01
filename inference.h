#ifndef INFERENCE_H
#define INFERENCE_H

// Cpp native
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <iostream>

// OpenCV / DNN / Inference
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// OpenCV DNN output tensor layout (batch boyutu 4.x vs 5.x davranışı)
enum class OpenCVDNNTensorLayout {
    LAYOUT_3D,  // OpenCV 4.x: [batch, dim, anchors]
    LAYOUT_2D   // OpenCV 5.x: batch drop edilmiş, [dim, anchors]
};

// YOLO model çıktı formatı (versiyon mimarisine göre parse stratejisi)
enum class YOLOOutputFormat {
    YOLOV5,   // [batch, 25200, 85]: objectness + per-class scores, center+size koordinatları
    YOLOV8,   // [batch, 84, 8400]:  no objectness, per-class scores, center+size (transposed)
    YOLOV10,  // [batch, N, 6]:      NMS-free, conf+class_id, köşe koordinatları (x1,y1,x2,y2)
    YOLOV11,  // YOLOv8 ile aynı ONNX çıktı formatı, [batch, 4+cls, 8400]
    YOLOV12,  // YOLOv8/v11 ile aynı ONNX çıktı formatı (R-YOLO, A2C2f head), [batch, 4+cls, 8400]
    YOLOV26,  // Rezerv — format belirsiz, varsayılan V8 uyumlu; [N,6] ise YOLOV10 kullan
    AUTO      // Tensor boyutlarına göre otomatik tespit (varsayılan, geriye dönük uyumlu)
};

// OpenCV 4.x ve 5.x her ikisi de [1, dim, anchors] 3D tensor döndürür
static constexpr OpenCVDNNTensorLayout kDefaultTensorLayout = OpenCVDNNTensorLayout::LAYOUT_3D;

struct Detection
{
    int class_id{0};
    std::string className{};
    float confidence{0.0};
    cv::Scalar color{};
    cv::Rect box{};
};

class Inference
{
public:
    Inference(const std::string &onnxModelPath, const cv::Size &modelInputShape = {640, 640}, const std::string &classesTxtFile = "", const bool &runWithCuda = true);
    std::vector<Detection> runInference(const cv::Mat &input);

    void setOpenCVLayout(OpenCVDNNTensorLayout layout) { cvLayout = layout; }
    void setYOLOFormat(YOLOOutputFormat format)        { yoloFormat = format; }

private:
    void loadClassesFromFile();
    void loadOnnxNetwork();
    cv::Mat formatToSquare(const cv::Mat &source);

    std::string modelPath{};
    std::string classesPath{};
    bool cudaEnabled{};

    OpenCVDNNTensorLayout cvLayout    { kDefaultTensorLayout };
    YOLOOutputFormat      yoloFormat  { YOLOOutputFormat::AUTO };

    // std::vector<std::string> classes{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
    // std::vector<std::string> classes{"person", "bicycle", "car", "motor", "bus", "train", "truck", "light", "dog", "scooter", "other vehicle"};
    // std::vector<std::string> classes{"face"};
    std::vector<std::string> classes{"person"};
    // std::vector<std::string> classes{"person", "bicycle", "car"};

    cv::Size2f modelShape{};

    float modelConfidenceThreshold {0.25};
    float modelScoreThreshold      {0.45};
    float modelNMSThreshold        {0.50};

    bool letterBoxForSquare = true;

    cv::dnn::Net net;
};

#endif // INFERENCE_H
