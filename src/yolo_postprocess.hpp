/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
#pragma once
#include "hailo_objects.hpp"
#include "hailo_common.hpp"
#include "yolo_output.hpp"
#include "coco_eighty.hpp"

__BEGIN_DECLS

class YoloParams
{
public:
    float iou_threshold;
    float detection_threshold;
    std::map<std::uint8_t, std::string> labels;
    uint32_t num_classes;
    uint32_t max_boxes;
    std::vector<std::vector<int>> anchors_vec;
    std::string output_activation; // can be "none" or "sigmoid"
    int label_offset;
    YoloParams() : iou_threshold(0.45f), detection_threshold(0.3f), output_activation("none"), label_offset(1) {}
    void check_params_logic(uint32_t num_classes_tensors);
};

class Yolov5Params : public YoloParams
{
public:
    Yolov5Params()
    {
        labels = common::coco_eighty;
        max_boxes = 200;
        anchors_vec = {
            {116, 90, 156, 198, 373, 326},
            {30, 61, 62, 45, 59, 119},
            {10, 13, 16, 30, 33, 23}};
    }
};

YoloParams *init(std::string config_path, std::string func_name);
void free_resources(void *params_void_ptr);
void filter(HailoROIPtr roi, void *params_void_ptr);
void yolov5(HailoROIPtr roi, void *params_void_ptr);

__END_DECLS