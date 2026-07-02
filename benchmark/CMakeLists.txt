# ==================== MOT17 BENCHMARK SYSTEM ====================
# Headless benchmark executable for systematic tracking algorithm testing
# Version: 1.0.0
# ================================================================

cmake_minimum_required(VERSION 3.16)

# Find parent project headers
include_directories(${CMAKE_SOURCE_DIR})

# Benchmark executable
add_executable(mot17_benchmark
  mot17_benchmark.cpp
  benchmark_config.h

  # MOT17 readers (from parent)
  ${CMAKE_SOURCE_DIR}/mot/mot_sequence_reader.h
  ${CMAKE_SOURCE_DIR}/mot/mot_sequence_reader.cpp
  ${CMAKE_SOURCE_DIR}/mot/mot_detection_reader.h
  ${CMAKE_SOURCE_DIR}/mot/mot_detection_reader.cpp

  # Tracking system (from parent)
  ${CMAKE_SOURCE_DIR}/trackers/tracker_factory.h
  ${CMAKE_SOURCE_DIR}/trackers/kalman_iou/kalman_iou_tracker.h
  ${CMAKE_SOURCE_DIR}/trackers/kalman_iou/kalman_iou_tracker.cpp
  ${CMAKE_SOURCE_DIR}/trackers/kalman_iou/kalman_iou_tracker_6state.h
  ${CMAKE_SOURCE_DIR}/trackers/kalman_iou/kalman_iou_tracker_6state.cpp
  ${CMAKE_SOURCE_DIR}/trackers/kalman_iou/kalman_iou_byte_track.h
  ${CMAKE_SOURCE_DIR}/trackers/kalman_iou/kalman_iou_byte_track.cpp
  ${CMAKE_SOURCE_DIR}/inference.h
  ${CMAKE_SOURCE_DIR}/inference.cpp
  ${CMAKE_SOURCE_DIR}/impression.h
  ${CMAKE_SOURCE_DIR}/impression.cpp
  ${CMAKE_SOURCE_DIR}/trackers/reid_extractor.h
  ${CMAKE_SOURCE_DIR}/trackers/reid_extractor.cpp

  # BoT-SORT tracker
  ${CMAKE_SOURCE_DIR}/trackers/bot_sort/bot_sort_config.h
  ${CMAKE_SOURCE_DIR}/trackers/bot_sort/cmc_module.h
  ${CMAKE_SOURCE_DIR}/trackers/bot_sort/cmc_module.cpp
  ${CMAKE_SOURCE_DIR}/trackers/bot_sort/bot_sort_tracker.h
  ${CMAKE_SOURCE_DIR}/trackers/bot_sort/bot_sort_tracker.cpp

  # Telemetry (from parent)
  ${CMAKE_SOURCE_DIR}/TrackerTelemtryLogger/MotChallengeLogger.h
  ${CMAKE_SOURCE_DIR}/TrackerTelemtryLogger/MotChallengeLogger.cpp

  # Utilities (from parent)
  ${CMAKE_SOURCE_DIR}/trackers/helpers/HungarianAlgorithm.h
  ${CMAKE_SOURCE_DIR}/trackers/helpers/HungarianAlgorithm.cpp

  # Interfaces (from parent)
  ${CMAKE_SOURCE_DIR}/i_tracker_telemetry.h
  ${CMAKE_SOURCE_DIR}/i_object_tracker.h
  ${CMAKE_SOURCE_DIR}/i_visual_tracker.h
  ${CMAKE_SOURCE_DIR}/i_configurable_tracker.h
)

# Link libraries
target_link_libraries(mot17_benchmark ${OpenCV_LIBS} Eigen3::Eigen)

if(REID_BACKEND_ORT AND _ORT_AVAILABLE)
    target_compile_definitions(mot17_benchmark PRIVATE USE_ORT_REID)
    target_include_directories(mot17_benchmark PRIVATE "${ORT_INCLUDE_DIR}")
    target_link_libraries(mot17_benchmark "${ORT_SHARED_LIB}")
    set_target_properties(mot17_benchmark PROPERTIES
        BUILD_RPATH  "${_ORT_CAPI_DIR}"
        INSTALL_RPATH "${_ORT_CAPI_DIR}")
endif()



# Minimal benchmark executable (fast testing with fixed IoU)
add_executable(mot17_minimal_benchmark
  mot17_minimal_benchmark.cpp
  benchmark_config.h

  ${CMAKE_SOURCE_DIR}/mot/mot_sequence_reader.h
  ${CMAKE_SOURCE_DIR}/mot/mot_sequence_reader.cpp
  ${CMAKE_SOURCE_DIR}/mot/mot_detection_reader.h
  ${CMAKE_SOURCE_DIR}/mot/mot_detection_reader.cpp
  ${CMAKE_SOURCE_DIR}/trackers/tracker_factory.h
  ${CMAKE_SOURCE_DIR}/trackers/kalman_iou/kalman_iou_tracker.h
  ${CMAKE_SOURCE_DIR}/trackers/kalman_iou/kalman_iou_tracker.cpp
  ${CMAKE_SOURCE_DIR}/trackers/kalman_iou/kalman_iou_tracker_6state.h
  ${CMAKE_SOURCE_DIR}/trackers/kalman_iou/kalman_iou_tracker_6state.cpp
  ${CMAKE_SOURCE_DIR}/trackers/kalman_iou/kalman_iou_byte_track.h
  ${CMAKE_SOURCE_DIR}/trackers/kalman_iou/kalman_iou_byte_track.cpp
  ${CMAKE_SOURCE_DIR}/trackers/reid_extractor.h
  ${CMAKE_SOURCE_DIR}/trackers/reid_extractor.cpp
  ${CMAKE_SOURCE_DIR}/trackers/bot_sort/bot_sort_config.h
  ${CMAKE_SOURCE_DIR}/trackers/bot_sort/cmc_module.h
  ${CMAKE_SOURCE_DIR}/trackers/bot_sort/cmc_module.cpp
  ${CMAKE_SOURCE_DIR}/trackers/bot_sort/bot_sort_tracker.h
  ${CMAKE_SOURCE_DIR}/trackers/bot_sort/bot_sort_tracker.cpp
  ${CMAKE_SOURCE_DIR}/TrackerTelemtryLogger/MotChallengeLogger.h
  ${CMAKE_SOURCE_DIR}/TrackerTelemtryLogger/MotChallengeLogger.cpp
  ${CMAKE_SOURCE_DIR}/trackers/helpers/HungarianAlgorithm.h
  ${CMAKE_SOURCE_DIR}/trackers/helpers/HungarianAlgorithm.cpp
  ${CMAKE_SOURCE_DIR}/i_tracker_telemetry.h
  ${CMAKE_SOURCE_DIR}/i_object_tracker.h
  ${CMAKE_SOURCE_DIR}/i_visual_tracker.h
  ${CMAKE_SOURCE_DIR}/i_configurable_tracker.h
)

target_link_libraries(mot17_minimal_benchmark
  ${OpenCV_LIBS}
  Eigen3::Eigen
)

add_executable(mot17_compare_trackers_benchmark
  mot17_compare_trackers_benchmark.cpp
  benchmark_config.h

  ${CMAKE_SOURCE_DIR}/mot/mot_sequence_reader.h
  ${CMAKE_SOURCE_DIR}/mot/mot_sequence_reader.cpp
  ${CMAKE_SOURCE_DIR}/mot/mot_detection_reader.h
  ${CMAKE_SOURCE_DIR}/mot/mot_detection_reader.cpp
  ${CMAKE_SOURCE_DIR}/trackers/tracker_factory.h
  ${CMAKE_SOURCE_DIR}/trackers/kalman_iou/kalman_iou_tracker.h
  ${CMAKE_SOURCE_DIR}/trackers/kalman_iou/kalman_iou_tracker.cpp
  ${CMAKE_SOURCE_DIR}/trackers/kalman_iou/kalman_iou_tracker_6state.h
  ${CMAKE_SOURCE_DIR}/trackers/kalman_iou/kalman_iou_tracker_6state.cpp
  ${CMAKE_SOURCE_DIR}/trackers/kalman_iou/kalman_iou_byte_track.h
  ${CMAKE_SOURCE_DIR}/trackers/kalman_iou/kalman_iou_byte_track.cpp
  ${CMAKE_SOURCE_DIR}/trackers/reid_extractor.h
  ${CMAKE_SOURCE_DIR}/trackers/reid_extractor.cpp
  ${CMAKE_SOURCE_DIR}/trackers/bot_sort/bot_sort_config.h
  ${CMAKE_SOURCE_DIR}/trackers/bot_sort/cmc_module.h
  ${CMAKE_SOURCE_DIR}/trackers/bot_sort/cmc_module.cpp
  ${CMAKE_SOURCE_DIR}/trackers/bot_sort/bot_sort_tracker.h
  ${CMAKE_SOURCE_DIR}/trackers/bot_sort/bot_sort_tracker.cpp
  ${CMAKE_SOURCE_DIR}/TrackerTelemtryLogger/MotChallengeLogger.h
  ${CMAKE_SOURCE_DIR}/TrackerTelemtryLogger/MotChallengeLogger.cpp
  ${CMAKE_SOURCE_DIR}/trackers/helpers/HungarianAlgorithm.h
  ${CMAKE_SOURCE_DIR}/trackers/helpers/HungarianAlgorithm.cpp
  ${CMAKE_SOURCE_DIR}/i_tracker_telemetry.h
  ${CMAKE_SOURCE_DIR}/i_object_tracker.h
  ${CMAKE_SOURCE_DIR}/i_visual_tracker.h
  ${CMAKE_SOURCE_DIR}/i_configurable_tracker.h
)

target_link_libraries(mot17_compare_trackers_benchmark
  ${OpenCV_LIBS}
  Eigen3::Eigen
)

if(REID_BACKEND_ORT AND _ORT_AVAILABLE)
    target_compile_definitions(mot17_compare_trackers_benchmark PRIVATE USE_ORT_REID)
    target_include_directories(mot17_compare_trackers_benchmark PRIVATE "${ORT_INCLUDE_DIR}")
    target_link_libraries(mot17_compare_trackers_benchmark "${ORT_SHARED_LIB}")
    set_target_properties(mot17_compare_trackers_benchmark PROPERTIES
        BUILD_RPATH  "${_ORT_CAPI_DIR}"
        INSTALL_RPATH "${_ORT_CAPI_DIR}")
endif()

message(STATUS "MOT17 Benchmark System configured (v${PROJECT_VERSION})")
message(STATUS "  - mot17_benchmark (full: 168 tests)")
message(STATUS "  - mot17_minimal_benchmark (minimal: 21 tests, configurable IoU)")
message(STATUS "  - mot17_compare_trackers_benchmark (4 tracker comparison: 6state, 8state, bytetrack, botsort)")
