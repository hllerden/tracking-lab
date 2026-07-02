cmake_minimum_required(VERSION 3.16)

# ==================== PROJECT VERSION ====================
set(PROJECT_VERSION_MAJOR 1)
set(PROJECT_VERSION_MINOR 1)
set(PROJECT_VERSION_PATCH 0)
set(PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")

project(yoloExercise VERSION ${PROJECT_VERSION} LANGUAGES CXX)

# Make version and project root available at compile time
add_compile_definitions(PROJECT_VERSION="${PROJECT_VERSION}")
add_compile_definitions(PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}")

message(STATUS "Project version: ${PROJECT_VERSION}")
# =========================================================

# set(CMAKE_AUTOUIC ON)
# set(CMAKE_AUTOMOC ON)
# set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(OpenCV_DIR "/opt/opencv-5.0.0-cuda-contrib/lib/cmake/opencv5")
# set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")

find_package(OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS})

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

# CUDA
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(CUDA 12 REQUIRED)

# ==================== REID BACKEND OPTION ====================
# OFF (default): OpenCV DNN — sequential batch=1 calls, no extra dependency
# ON           : ONNX Runtime GPU — true batch inference, ~10x faster for N detections
#   Requires: pip install onnxruntime-gpu  AND  headers in thirdParty/onnxruntime/include/
#   Model: use yolo26n-reid.onnx (ORT handles dynamic shapes, no onnxsim needed)
option(REID_BACKEND_ORT "Use ONNX Runtime for ReID GPU batch inference" ON)

if(REID_BACKEND_ORT)
    set(ORT_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/thirdParty/onnxruntime/include")

    # Auto-discover library from pip-installed onnxruntime-gpu
    execute_process(
        COMMAND python3 -c
            "import onnxruntime, os; print(os.path.join(os.path.dirname(onnxruntime.__file__), 'capi'))"
        OUTPUT_VARIABLE _ORT_CAPI_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    find_file(ORT_SHARED_LIB
        NAMES "libonnxruntime.so.1.23.2" "libonnxruntime.so"
        HINTS "${_ORT_CAPI_DIR}"
        NO_DEFAULT_PATH
    )

    if(ORT_SHARED_LIB AND EXISTS "${ORT_INCLUDE_DIR}/onnxruntime_cxx_api.h")
        set(_ORT_AVAILABLE TRUE)
        message(STATUS "ReID backend: ONNX Runtime  lib=${ORT_SHARED_LIB}")
        # Ensure the SONAME symlink (libonnxruntime.so.1) exists next to the versioned file.
        # The dynamic linker resolves the SONAME recorded in the binary at runtime.
        get_filename_component(_ORT_LIB_DIR "${ORT_SHARED_LIB}" DIRECTORY)
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E create_symlink
                "${ORT_SHARED_LIB}"
                "${_ORT_LIB_DIR}/libonnxruntime.so.1"
            ERROR_QUIET
        )
    else()
        set(_ORT_AVAILABLE FALSE)
        message(WARNING "REID_BACKEND_ORT=ON but library/headers missing — falling back to OpenCV DNN")
    endif()
endif()
# ============================================================

set(CMAKE_CUDA_STANDARD 12)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

# add_subdirectory(YOLOv8-CPP-Inference)
add_executable(yoloExercise
  main.cpp
  inference.h inference.cpp
  trackers/tracker_factory.h
  trackers/kalman_iou/kalman_iou_tracker.h trackers/kalman_iou/kalman_iou_tracker.cpp
  trackers/kalman_iou/kalman_iou_byte_track.h trackers/kalman_iou/kalman_iou_byte_track.cpp
  i_tracker_telemetry.h
  i_object_tracker.h
  i_visual_tracker.h
  i_configurable_tracker.h
  TrackerTelemtryLogger/MotChallengeLogger.h
  TrackerTelemtryLogger/MotChallengeLogger.cpp
  trackers/helpers/HungarianAlgorithm.h trackers/helpers/HungarianAlgorithm.cpp
  mot/mot_sequence_reader.h mot/mot_sequence_reader.cpp
  mot/mot_detection_reader.h mot/mot_detection_reader.cpp

  lowlightenhancer.h lowlightenhancer.cpp

  trackers/opencv_tracker_integration.h trackers/opencv_tracker_integration.cpp
  trackers/reid_extractor.h trackers/reid_extractor.cpp

  # thirdParty/deepsort/dataType.h
  # # thirdParty/deepsort/FeatureTensor.h
  # thirdParty/deepsort/hungarianoper.h
  # thirdParty/deepsort/kalmanfilter.h
  # thirdParty/deepsort/linear_assignment.h
  # thirdParty/deepsort/matrix.h
  # thirdParty/deepsort/model.h
  # thirdParty/deepsort/munkres.h
  # thirdParty/deepsort/nn_matching.h
  # thirdParty/deepsort/track.h
  # thirdParty/deepsort/tracker.h

  # # thirdParty/deepsort/FeatureTensor.cpp
  # thirdParty/deepsort/hungarianoper.cpp
  # thirdParty/deepsort/kalmanfilter.cpp
  # thirdParty/deepsort/linear_assignment.cpp
  # thirdParty/deepsort/model.cpp
  # thirdParty/deepsort/munkres.cpp
  # thirdParty/deepsort/nn_matching.cpp
  # thirdParty/deepsort/track.cpp
  # thirdParty/deepsort/tracker.cpp
  impression.h impression.cpp
  trackers/kalman_iou/kalman_iou_tracker_6state.h
  trackers/kalman_iou/kalman_iou_tracker_6state.cpp

  trackers/bot_sort/bot_sort_config.h
  trackers/bot_sort/cmc_module.h trackers/bot_sort/cmc_module.cpp
  trackers/bot_sort/bot_sort_tracker.h trackers/bot_sort/bot_sort_tracker.cpp

)
target_link_libraries(yoloExercise ${OpenCV_LIBS} Eigen3::Eigen)

if(REID_BACKEND_ORT AND _ORT_AVAILABLE)
    target_compile_definitions(yoloExercise PRIVATE USE_ORT_REID)
    target_include_directories(yoloExercise PRIVATE "${ORT_INCLUDE_DIR}")
    target_link_libraries(yoloExercise "${ORT_SHARED_LIB}")
    set_target_properties(yoloExercise PROPERTIES
        BUILD_RPATH  "${_ORT_CAPI_DIR}"
        INSTALL_RPATH "${_ORT_CAPI_DIR}")
endif()

# Test executable for MOT sequence reader
add_executable(test_mot_reader
  test_mot_reader.cpp
  mot/mot_sequence_reader.h mot/mot_sequence_reader.cpp
)
target_link_libraries(test_mot_reader ${OpenCV_LIBS})

add_executable(mot17_tracking_mp4
  tools/mot17_tracking_mp4.cpp
  mot/mot_sequence_reader.h mot/mot_sequence_reader.cpp
  mot/mot_detection_reader.h mot/mot_detection_reader.cpp
  trackers/tracker_factory.h
  trackers/kalman_iou/kalman_iou_tracker.h trackers/kalman_iou/kalman_iou_tracker.cpp
  trackers/kalman_iou/kalman_iou_byte_track.h trackers/kalman_iou/kalman_iou_byte_track.cpp
  trackers/kalman_iou/kalman_iou_tracker_6state.h trackers/kalman_iou/kalman_iou_tracker_6state.cpp
  trackers/helpers/HungarianAlgorithm.h trackers/helpers/HungarianAlgorithm.cpp
  trackers/bot_sort/bot_sort_config.h
  trackers/bot_sort/cmc_module.h trackers/bot_sort/cmc_module.cpp
  trackers/bot_sort/bot_sort_tracker.h trackers/bot_sort/bot_sort_tracker.cpp
  i_object_tracker.h
  i_visual_tracker.h
  i_configurable_tracker.h
  i_tracker_telemetry.h
)
target_link_libraries(mot17_tracking_mp4 ${OpenCV_LIBS} Eigen3::Eigen)
target_include_directories(mot17_tracking_mp4 PRIVATE ${CMAKE_SOURCE_DIR})

# ==================== BENCHMARK SYSTEM SUBDIRECTORY ====================
# MOT17 Benchmark System (headless, versioned, comprehensive testing)
add_subdirectory(benchmark)
# =======================================================================



# target_link_libraries(MotionDetection Qt${QT_VERSION_MAJOR}::Core     ${OpenCV_LIBS}
# )

include(GNUInstallDirs)
install(TARGETS yoloExercise
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
