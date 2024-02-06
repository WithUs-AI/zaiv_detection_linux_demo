/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
#include <cmath>
#include <vector>
#include <algorithm>
#include "yolo_output.hpp"

std::pair<uint32_t, float> YoloOutputLayer::get_class(uint32_t row, uint32_t col, uint32_t anchor)
{
    uint32_t cls_prob, prob_max = 0;
    uint32_t selected_class_id = 1;
    for (uint32_t class_id = label_offset; class_id <= _num_classes; class_id++)
    {
        cls_prob = get_class_prob(row, col, anchor, class_id);
        if (cls_prob > prob_max)
        {
            selected_class_id = class_id;
            prob_max = cls_prob;
        }
    }
    return std::pair<uint32_t, float>(selected_class_id, get_class_conf(prob_max));
}

float YoloOutputLayer::get_confidence(uint32_t row, uint32_t col, uint32_t anchor)
{
    uint32_t channel = _tensor->features() / NUM_ANCHORS * anchor + CONF_CHANNEL_OFFSET;
    float confidence = _tensor->get_full_percision(row, col, channel, _is_uint16);
    if (_perform_sigmoid)
        confidence = sigmoid(confidence);
    return confidence;
}

float YoloOutputLayer::sigmoid(float x)
{
    // returns the value of the sigmoid function f(x) = 1/(1 + e^-x)
    return 1.0f / (1.0f + expf(-x));
}

uint32_t YoloOutputLayer::get_class_prob(uint32_t row, uint32_t col, uint32_t anchor, uint32_t class_id)
{
    uint32_t channel = _tensor->features() / NUM_ANCHORS * anchor + CLASS_CHANNEL_OFFSET + class_id - 1;
    if (_is_uint16)
        return _tensor->get_uint16(row, col, channel);
    else
        return _tensor->get(row, col, channel);
}

float Yolov5OL::get_class_conf(uint32_t prob_max)
{
    float conf = _tensor->fix_scale(prob_max);
    if (_perform_sigmoid)
        conf = sigmoid(conf);
    return conf;
}

std::pair<float, float> Yolov5OL::get_center(uint32_t row, uint32_t col, uint32_t anchor)
{
    float x, y = 0.0f;
    uint32_t channel = _tensor->features() / NUM_ANCHORS * anchor;
    x = (_tensor->get_full_percision(row, col, channel, _is_uint16) * 2.0f - 0.5f + col) / _width;
    y = (_tensor->get_full_percision(row, col, channel + 1, _is_uint16) * 2.0f - 0.5f + row) / _height;
    return std::pair<float, float>(x, y);
}

std::pair<float, float> Yolov5OL::get_shape(uint32_t row, uint32_t col, uint32_t anchor, uint32_t image_width, uint32_t image_height)
{
    float w, h = 0.0f;
    uint32_t channel = _tensor->features() / NUM_ANCHORS * anchor + NUM_CENTERS;
    w = pow(2.0f * _tensor->get_full_percision(row, col, channel, _is_uint16), 2.0f) * _anchors[anchor * 2] / image_width;
    h = pow(2.0f * _tensor->get_full_percision(row, col, channel + 1, _is_uint16), 2.0f) * _anchors[anchor * 2 + 1] / image_height;
    return std::pair<float, float>(w, h);
}