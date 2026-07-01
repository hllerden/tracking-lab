#include "inference.h"

Inference::Inference(const std::string &onnxModelPath, const cv::Size &modelInputShape, const std::string &classesTxtFile, const bool &runWithCuda)
{
    modelPath = onnxModelPath;
    modelShape = modelInputShape;
    classesPath = classesTxtFile;
    cudaEnabled = runWithCuda;

    loadOnnxNetwork();
    // loadClassesFromFile(); The classes are hard-coded for this example
}

std::vector<Detection> Inference::runInference(const cv::Mat &input)
{
    cv::Mat modelInput = input;
    if (modelInput.empty()) {
        std::cerr << "HATA: modelInput boş!" << std::endl;

    }

    if (letterBoxForSquare && modelShape.width == modelShape.height)
        modelInput = formatToSquare(modelInput);

    cv::Mat blob;
    cv::dnn::blobFromImage(modelInput, blob, 1.0/255.0, modelShape, cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    // --- Bölüm 1: OpenCV versiyonuna göre tensor boyutlarını oku ---
    int rawRows, rawDims;
    if (cvLayout == OpenCVDNNTensorLayout::LAYOUT_3D) {
        // OpenCV 4.x: [1, dim, anchors]
        rawRows = outputs[0].size[1];
        rawDims = outputs[0].size[2];
    } else {
        // OpenCV 5.x: batch drop edilmiş [dim, anchors]
        rawRows = outputs[0].size[0];
        rawDims = outputs[0].size[1];
    }

    // --- Bölüm 2: YOLO format tespiti ve transpose ---
    YOLOOutputFormat resolvedFormat = yoloFormat;

    // AUTO ve YOLOV26 (format belirsiz rezerv): tensor boyutlarından tespit et
    if (resolvedFormat == YOLOOutputFormat::AUTO ||
        resolvedFormat == YOLOOutputFormat::YOLOV26) {
        if (rawDims == 6) {
            resolvedFormat = YOLOOutputFormat::YOLOV10;
        } else if (rawDims > rawRows) {
            resolvedFormat = YOLOOutputFormat::YOLOV8;
        } else {
            resolvedFormat = YOLOOutputFormat::YOLOV5;
        }
    }

    // V8 uyumlu formatlar: V8, V11, V12 → transpose gerektirir
    auto isV8Compatible = [](YOLOOutputFormat f) {
        return f == YOLOOutputFormat::YOLOV8  ||
               f == YOLOOutputFormat::YOLOV11 ||
               f == YOLOOutputFormat::YOLOV12;
    };

    int rows = rawRows, dimensions = rawDims;
    if (isV8Compatible(resolvedFormat)) {
        rows       = rawDims;
        dimensions = rawRows;
        outputs[0] = outputs[0].reshape(1, rawRows);
        cv::transpose(outputs[0], outputs[0]);
    }

    float *data = (float *)outputs[0].data;

    float x_factor = modelInput.cols / modelShape.width;
    float y_factor = modelInput.rows / modelShape.height;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // --- Bölüm 3: Format-spesifik detection döngüsü ---
    for (int i = 0; i < rows; ++i)
    {
        if (resolvedFormat == YOLOOutputFormat::YOLOV10)
        {
            // [x1, y1, x2, y2, conf, class_id] — köşe koordinatları, NMS modelde yapılmış
            float conf   = data[4];
            int   cls_id = static_cast<int>(data[5]);
            if (conf >= modelConfidenceThreshold && cls_id >= 0 && cls_id < static_cast<int>(classes.size()))
            {
                float x1 = data[0] * x_factor;
                float y1 = data[1] * y_factor;
                float x2 = data[2] * x_factor;
                float y2 = data[3] * y_factor;
                boxes.push_back(cv::Rect(
                    static_cast<int>(x1), static_cast<int>(y1),
                    static_cast<int>(x2 - x1), static_cast<int>(y2 - y1)
                ));
                confidences.push_back(conf);
                class_ids.push_back(cls_id);
            }
        }
        else if (isV8Compatible(resolvedFormat))
        {
            // V8 / V11 / V12 / V26: [cx, cy, w, h, cls0_score, cls1_score, ...]
            float *classes_scores = data + 4;

            cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double maxClassScore;

            minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

            if (maxClassScore > modelScoreThreshold)
            {
                confidences.push_back(maxClassScore);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * x_factor);
                int top  = int((y - 0.5 * h) * y_factor);

                int width  = int(w * x_factor);
                int height = int(h * y_factor);

                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        else // YOLOV5: [cx, cy, w, h, objectness, cls0_score, cls1_score, ...]
        {
            float confidence = data[4];

            if (confidence >= modelConfidenceThreshold)
            {
                float *classes_scores = data + 5;

                cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
                cv::Point class_id;
                double max_class_score;

                minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

                if (max_class_score > modelScoreThreshold)
                {
                    confidences.push_back(confidence);
                    class_ids.push_back(class_id.x);

                    float x = data[0];
                    float y = data[1];
                    float w = data[2];
                    float h = data[3];

                    int left = int((x - 0.5 * w) * x_factor);
                    int top  = int((y - 0.5 * h) * y_factor);

                    int width  = int(w * x_factor);
                    int height = int(h * y_factor);

                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }

        data += dimensions;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);

    std::vector<Detection> detections{};
    for (unsigned long i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];

        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(100, 255);
        result.color = cv::Scalar(dis(gen),
                                  dis(gen),
                                  dis(gen));

        result.className = classes[result.class_id];
        result.box = boxes[idx];

        detections.push_back(result);
    }

    return detections;
}

void Inference::loadClassesFromFile()
{
    std::ifstream inputFile(classesPath);
    if (inputFile.is_open())
    {
        std::string classLine;
        while (std::getline(inputFile, classLine))
            classes.push_back(classLine);
        inputFile.close();
    }
}

void Inference::loadOnnxNetwork()
{
#if CV_VERSION_MAJOR >= 5
    // OpenCV 5.0: yeni engine (ENGINE_AUTO) sadece CPU destekler.
    // CUDA backend için ENGINE_CLASSIC zorunlu; CPU için ENGINE_AUTO
    // (yeni engine daha iyi ONNX operator coverage sağlar).
    if (cudaEnabled) {
        net = cv::dnn::readNetFromONNX(modelPath, cv::dnn::ENGINE_CLASSIC);
    } else {
        net = cv::dnn::readNetFromONNX(modelPath, cv::dnn::ENGINE_AUTO);
    }
#else
    net = cv::dnn::readNetFromONNX(modelPath);
#endif

    if (cudaEnabled)
    {
        std::cout << "\nRunning on CUDA" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    else
    {
        std::cout << "\nRunning on CPU" << std::endl;
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
}

cv::Mat Inference::formatToSquare(const cv::Mat &source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}
