#pragma once

#include <string>
#include "hailo_objects.hpp"
#include <opencv2/opencv.hpp>

void zaiv_input_frame_for_inference(cv::Mat *frame);
void zaiv_terminate_inference_thread();
void zaiv_start_inference_thread();
void zaiv_wait_for_inference_thread();
void zaiv_set_eth_name(std::string s);
void zaiv_set_hef_file(std::string s);
void zaiv_set_hef_labels(std::map<uint8_t, std::string> &labels);
bool zaiv_inference_thread_alive();
void zaiv_draw_all(cv::Mat *frame, HailoROIPtr roi);
bool zaiv_timer_timeout_33ms();
