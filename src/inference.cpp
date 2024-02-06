
#include <iostream>
#include "hailo/hailort.hpp"
#include "hailo_objects.hpp"
#include "hailo_tensors.hpp"
#include "hailo_common.hpp"
#include "yolo_postprocess.hpp"

#include <algorithm>
#include <future>
#include <cstdint>
#include <iomanip>
#include <array>
#include <queue>
#include <time.h>
#include <opencv2/opencv.hpp>





#define QUEUE_MAX 5 // timing mismatch fix -> 1




#define ETH_HEF_FILE "./hef/coco_nonms_eth_nonms_eth_hailo0913.hef"
#define PCIE_HEF_FILE "./hef/yolov5m_wo_spp_60p.hef"

#define CONFIG_FILE ("./resources/yolov5.json")




#define ACCELERATOR_SCAN_TIMEOUT_DEFAULT (5000)
#define ACCELERATOR_SCAN_MAX_DEVICE_DEFAULT (5)

#define INPUT_COUNT (1)
#define OUTPUT_COUNT (3)

#define REQUIRE_ACTION(cond, action, label, ...)                \
	do {                                                        \
		if (!(cond)) {                                          \
			std::cout << (__VA_ARGS__) << std::endl;            \
			action;                                             \
			goto label;                                         \
		}                                                       \
	} while(0)

#define REQUIRE_SUCCESS(status, label, ...) REQUIRE_ACTION((HAILO_SUCCESS == (status)), , label, __VA_ARGS__)



#define NO_GLOBAL_ID_COLOR (cv::Scalar(255, 0, 0))
#define GLOBAL_ID_COLOR (cv::Scalar(0, 255, 0))
#define SPACE " "

std::string confidence_to_string(float confidence)
{
	int confidence_percentage = (confidence * 100);

	return std::to_string(confidence_percentage) + "%";
}

static std::string get_detection_text(HailoDetectionPtr detection, bool show_confidence = true)
{
	std::string text;
	std::string label = detection->get_label();
	std::string confidence = confidence_to_string(detection->get_confidence());
	if (!show_confidence)
	text = label;
	else if (!label.empty())
	{
		text = label + SPACE + confidence;
	}
	else
	{
		text = confidence;
	}
	return text;
}

#define NULL_COLOR_ID ((size_t)NULL_CLASS_ID)
#define DEFAULT_COLOR (cv::Scalar(255, 255, 255))

static const std::vector<cv::Scalar> color_table = {
	cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 255),
	cv::Scalar(255, 0, 255), cv::Scalar(255, 170, 0), cv::Scalar(255, 0, 170), cv::Scalar(0, 255, 170), cv::Scalar(170, 255, 0),
	cv::Scalar(170, 0, 255), cv::Scalar(0, 170, 255), cv::Scalar(255, 85, 0), cv::Scalar(85, 255, 0), cv::Scalar(0, 255, 85),
	cv::Scalar(0, 85, 255), cv::Scalar(85, 0, 255), cv::Scalar(255, 0, 85), cv::Scalar(255, 255, 255) };


cv::Scalar indexToColor(size_t index)
{
	return color_table[index % color_table.size()];
}

static cv::Scalar get_color(size_t color_id)
{
	cv::Scalar color;
	if (NULL_COLOR_ID == color_id)
	color = DEFAULT_COLOR;
	else
	color = indexToColor(color_id);

	return color;
}


#define TEXT_FONT_FACTOR (0.12f)

bool show_confidence = true;

void zaiv_draw_all(cv::Mat *frame, HailoROIPtr roi)
{
	
	int detections = 0;

	for (HailoObjectPtr obj : roi->get_objects())
	{
		if (obj->get_type() == HAILO_DETECTION)
		{
			HailoDetectionPtr detection = std::dynamic_pointer_cast<HailoDetection>(obj);

			cv::Scalar color = NO_GLOBAL_ID_COLOR;
			std::string text = "";

			color = get_color((size_t)detection->get_class_id());
			text = get_detection_text(detection, show_confidence);

			HailoBBox roi_bbox = hailo_common::create_flattened_bbox(roi->get_bbox(), roi->get_scaling_bbox());
			auto detection_bbox = detection->get_bbox();

			auto bbox_min = cv::Point(
			((detection_bbox.xmin() * roi_bbox.width()) + roi_bbox.xmin()) * frame->size().width,
			((detection_bbox.ymin() * roi_bbox.height()) + roi_bbox.ymin()) * frame->size().height);
			auto bbox_max = cv::Point(
			((detection_bbox.xmax() * roi_bbox.width()) + roi_bbox.xmin()) * frame->size().width,
			((detection_bbox.ymax() * roi_bbox.height()) + roi_bbox.ymin()) * frame->size().height);

			cv::Rect rect = cv::Rect(bbox_min, bbox_max);
			cv::rectangle(*frame, rect, color);

			// Draw text
			cv::Point text_position = cv::Point(rect.x - log(rect.width), rect.y - log(rect.width));
			float font_scale = TEXT_FONT_FACTOR * log(rect.width);
			cv::putText(*frame, text, text_position, 0, font_scale, color);

			detections++;
		}
	}
	
	
	
	// std::cout << "detections : " << detections << std::endl;
}

YoloParams* init_params;


unsigned long GetTick()
{
	struct timespec tp;

	clock_gettime(CLOCK_MONOTONIC,&tp);
	return (unsigned long)(tp.tv_sec *1000 + tp.tv_nsec / 1000000);
}


bool _terminate = false;

void zaiv_terminate_inference_thread()
{
	_terminate = true;
}

bool zaiv_inference_thread_alive()
{
	return !_terminate;
}


bool thread_terminate = false;

int fps=0;

std::array<std::queue<std::vector<uint8_t>>, OUTPUT_COUNT> featuresbuffer;
std::mutex featuresbuffer_lock;

std::queue<cv::Mat> postprocessFrameQueue;
std::queue<HailoROIPtr> FrameDetections;

bool thread_terminated[OUTPUT_COUNT] = {false,};

void vstream_read_thread_runner(int index, hailo_output_vstream output_vstream, size_t output_vstream_frame_sizes)
{
	std::cout << "Starting vstream_read_thread_runner - index:" << index << "\n";
	
	std::vector<uint8_t> dummydata(output_vstream_frame_sizes);

	while(!thread_terminate)
	{
		hailo_status status = hailo_vstream_read_raw_buffer(output_vstream, dummydata.data(), output_vstream_frame_sizes);
		
		if (HAILO_SUCCESS != status) {
			std::cerr << "Failed reading with status = " << status << std::endl;
			_terminate = thread_terminate = true;
			break;
		}

		featuresbuffer_lock.lock();
		featuresbuffer[index].push(dummydata);
		featuresbuffer_lock.unlock();
		
	}
	
	thread_terminated[index] = true;
	std::cout << "vstream_read_thread_runner index:" << index << " Exited\n";
}



void yolo_output_collecter_thread_runner(hailo_vstream_info_t *output_vstream_info)
{
	std::cout << "Starting yolo_output_collecter_thread_runner\n";

	std::array<std::vector<uint8_t>, OUTPUT_COUNT> featuresbuffer_pop;
	bool all_streams_read_valid;

	while(!thread_terminate)
	{
		
		all_streams_read_valid = true;
		featuresbuffer_lock.lock();
		for(int i=0;i<OUTPUT_COUNT;i++)
		{
			if(featuresbuffer[i].size() == 0)
			{
				all_streams_read_valid = false;
				break;
			}
		}
		if(all_streams_read_valid)
		{
			for(int i=0;i<OUTPUT_COUNT;i++)
			{
				featuresbuffer_pop[i] = featuresbuffer[i].front();
				featuresbuffer[i].pop();
			}
		}
		featuresbuffer_lock.unlock();


		if(all_streams_read_valid)
		{
			fps++;
			
			HailoROIPtr roi = std::make_shared<HailoROI>(HailoROI(HailoBBox(0.0f, 0.0f, 1.0f, 1.0f)));

			roi->add_tensor(std::make_shared<HailoTensor>(reinterpret_cast<uint8_t*>(featuresbuffer_pop[2].data()), output_vstream_info[2]));
			roi->add_tensor(std::make_shared<HailoTensor>(reinterpret_cast<uint8_t*>(featuresbuffer_pop[1].data()), output_vstream_info[1]));
			roi->add_tensor(std::make_shared<HailoTensor>(reinterpret_cast<uint8_t*>(featuresbuffer_pop[0].data()), output_vstream_info[0]));

			yolov5(roi, init_params);
			
			FrameDetections.push(roi);
		}
		else
		{
			usleep(1000);
		}

	}
	
	std::cout << "yolo_output_collecter_thread_runner Terminated\n";
}

bool timer_timeout(int period)
{
	static unsigned long tick_past = 0;
	if(tick_past == 0)
	{
		tick_past = GetTick();
	}
	else if(tick_past + period < GetTick())
	{
		tick_past = GetTick();
		
		return true;
	}
	
	return false;
}


bool zaiv_timer_timeout_33ms()
{
	static unsigned long tick_past = 0;
	if(tick_past == 0)
	{
		tick_past = GetTick();
	}
	else if(tick_past + 33 < GetTick())
	{
		tick_past = GetTick();
		
		return true;
	}
	
	return false;
}

bool reset_hailo_some_attemps(hailo_device *device)
{
	static int reset_attemps = 0;
	std::cout << "HAILO RESET !!!!!!!! - " << reset_attemps << std::endl;
	hailo_reset_device(*device, HAILO_RESET_DEVICE_MODE_CHIP);
	sleep(5);
	if(reset_attemps++<=3) return true;
	else return false;
}




// bool wait_for_finish = false;

cv::Mat scaledframe;
cv::Mat rgbframe;

hailo_status status = HAILO_UNINITIALIZED;
hailo_device device = NULL;
hailo_hef hef = NULL;
hailo_configure_params_t config_params = { 0 };
hailo_configured_network_group network_group = NULL;
size_t network_group_size = 1;
hailo_input_vstream_params_by_name_t input_vstream_params[INPUT_COUNT] = { 0 };
hailo_output_vstream_params_by_name_t output_vstream_params[OUTPUT_COUNT] = { 0 };
size_t input_vstreams_size = INPUT_COUNT;
size_t output_vstreams_size = OUTPUT_COUNT;
hailo_activated_network_group activated_network_group = NULL;
hailo_input_vstream input_vstreams[INPUT_COUNT] = { NULL };
hailo_output_vstream output_vstreams[OUTPUT_COUNT] = { NULL };
size_t input_vstream_frame_sizes[INPUT_COUNT], output_vstream_frame_sizes[OUTPUT_COUNT];

hailo_vstream_info_t input_vstream_info[INPUT_COUNT];
hailo_vstream_info_t output_vstream_info[OUTPUT_COUNT];
std::vector<uint8_t> input_dummydata;

char hailo_eth_name[100];

std::thread vstream_read_thread[OUTPUT_COUNT];
std::thread yolo_output_collect_thread;

int hef_input_width;
int hef_input_height;


extern void Inference_input_request_cb();
extern void Inferenced_Frame_cb(cv::Mat showframe, HailoROIPtr roi);


void zaiv_set_eth_name(char *s)
{
	strcpy(hailo_eth_name, s);
}

	
void zaiv_input_frame_for_inference(cv::Mat *frame)
{
	cv::Mat *mat;
	
	if (frame->size().width == hef_input_width && frame->size().height == hef_input_height)
	{
		mat = frame;
	}
	else
	{
		cv::resize(*frame, scaledframe, cv::Size(hef_input_width, hef_input_height), 0, 0);
		mat = &scaledframe;
	}
	
	cv::cvtColor(*mat, rgbframe, cv::COLOR_BGR2RGB);
	
	usleep(1000); // this fix drawing glitches in VM
	
	hailo_status status = hailo_vstream_write_raw_buffer(input_vstreams[0], rgbframe.data, input_vstream_frame_sizes[0]);
	if (HAILO_SUCCESS != status) {
		std::cerr << "Failed writing to device data of image. Got status = " << status << std::endl;
		_terminate = thread_terminate = true;
		return;
	}
	
	
	postprocessFrameQueue.push((*mat).clone());
}

int inference_runner()
{
	std::cout << "Starting inference_runner\n";

	
	init_params = init(CONFIG_FILE, "yolov5");

	
	

	hailo_eth_device_info_t device_infos[ACCELERATOR_SCAN_MAX_DEVICE_DEFAULT];
	size_t num_of_devices = 0;
	
	std::cout << "scanning eth hailo ! - " << hailo_eth_name << "\n";
	
	status = hailo_scan_ethernet_devices(
		hailo_eth_name,
		device_infos,
		ACCELERATOR_SCAN_MAX_DEVICE_DEFAULT,
		&num_of_devices,
		ACCELERATOR_SCAN_TIMEOUT_DEFAULT);
	
	if(status == HAILO_SUCCESS && num_of_devices > 0) // ETH Board Configure
	{
		std::cout << "ETH Hailo Go !\n";
		
		std::cout << "num_of_devices : " << num_of_devices << std::endl;
		
		if(num_of_devices != 1)
		{
			std::cout << "ETH Hailo Detected on more devices\n";
			_terminate = true;
			return 1;
		}
		
		
		std::string cmd = "";
		
		cmd = "sudo ./resources/configure_ethernet_buffers.sh ";
		cmd += hailo_eth_name;
		
		std::cout << "run Command - " << cmd << "\n";
		system(cmd.c_str());
		
		sleep(3);
		
		const char *s_device_address = inet_ntoa(device_infos[0].device_address.sin_addr);
		std::cout << "device_address: " << s_device_address << "\n";
		
		do {
			status = hailo_create_ethernet_device(&device_infos[0], &device);
			// REQUIRE_SUCCESS(status, l_exit, "Failed to create eth_device");
			if(status != HAILO_SUCCESS)
			{
				std::cout << "Failed to create eth_device" << std::endl;
				if(reset_hailo_some_attemps(&device)) continue;
				else 
				{
					_terminate = true;
					return 1;
				}
			}
			
			status = hailo_create_hef_file(&hef, ETH_HEF_FILE);
			// REQUIRE_SUCCESS(status, l_release_device, "Failed reading hef file");
			if(status != HAILO_SUCCESS)
			{
				std::cout << "Failed reading hef file" << std::endl;
				(void)hailo_release_device(device);
				if(reset_hailo_some_attemps(&device)) continue;
				else 
				{
					_terminate = true;
					return 1;
				}
			}
			
			status = hailo_init_configure_params(hef, HAILO_STREAM_INTERFACE_ETH, &config_params);
			// REQUIRE_SUCCESS(status, l_release_hef, "Failed initializing configure parameters");
			if(status != HAILO_SUCCESS)
			{
				std::cout << "Failed initializing configure parameters" << std::endl;
				(void)hailo_release_hef(hef);
				if(reset_hailo_some_attemps(&device)) continue;
				else 
				{
					_terminate = true;
					return 1;
				}
			}
			
			status = hailo_configure_device(device, hef, &config_params, &network_group, &network_group_size);
			// REQUIRE_SUCCESS(status, l_release_hef, "Failed configure devcie from hef");
			if(status != HAILO_SUCCESS)
			{
				std::cout << "Failed configure devcie from hef" << std::endl;
				(void)hailo_release_hef(hef);
				if(reset_hailo_some_attemps(&device)) continue;
				else 
				{
					_terminate = true;
					return 1;
				}
			}
			
		}
		while(status != HAILO_SUCCESS);
		
		
	}
	else // PCI Board Configure
	{
		std::cout << "PCIE Hailo Go !\n";
		
		status = hailo_create_pcie_device(NULL, &device);
		REQUIRE_SUCCESS(status, l_exit, "Failed to create pcie_device");

		status = hailo_create_hef_file(&hef, PCIE_HEF_FILE);
		REQUIRE_SUCCESS(status, l_release_device, "Failed reading hef file");

		status = hailo_init_configure_params(hef, HAILO_STREAM_INTERFACE_PCIE, &config_params);
		REQUIRE_SUCCESS(status, l_release_hef, "Failed initializing configure parameters");
		
		status = hailo_configure_device(device, hef, &config_params, &network_group, &network_group_size);
		REQUIRE_SUCCESS(status, l_release_hef, "Failed configure devcie from hef");
	}
	



	
	
	

	REQUIRE_ACTION(network_group_size == 1, status = HAILO_INVALID_ARGUMENT, l_release_hef, "Invalid network group size");

	status = hailo_make_input_vstream_params(network_group, true, HAILO_FORMAT_TYPE_AUTO, input_vstream_params, &input_vstreams_size);
	REQUIRE_SUCCESS(status, l_release_hef, "Failed making input virtual stream params");

	status = hailo_make_output_vstream_params(network_group, true, HAILO_FORMAT_TYPE_AUTO, output_vstream_params, &output_vstreams_size);
	REQUIRE_SUCCESS(status, l_release_hef, "Failed making output virtual stream params");

	std::cout << "input_vstreams_size : " << input_vstreams_size << std::endl;
	std::cout << "output_vstreams_size : " << output_vstreams_size << std::endl;

	REQUIRE_ACTION(((input_vstreams_size == INPUT_COUNT) || (output_vstreams_size == OUTPUT_COUNT)), status = HAILO_INVALID_OPERATION, l_release_hef, "Expected one input vstream and three outputs vstreams");

	status = hailo_create_input_vstreams(network_group, input_vstream_params, input_vstreams_size, input_vstreams);
	REQUIRE_SUCCESS(status, l_release_hef, "Failed creating input virtual streams");

	status = hailo_create_output_vstreams(network_group, output_vstream_params, output_vstreams_size, output_vstreams);
	REQUIRE_SUCCESS(status, l_release_input_vstream, "Failed creating output virtual streams");

	status = hailo_activate_network_group(network_group, NULL, &activated_network_group);
	REQUIRE_SUCCESS(status, l_release_output_vstream, "Failed activating network group");
	
	
	for (size_t i = 0; i < input_vstreams_size; i++)
	{
		status = hailo_get_input_vstream_frame_size(input_vstreams[i], &input_vstream_frame_sizes[i]);
		REQUIRE_SUCCESS(status, l_deactivate_network_group, "Failed getting input virtual stream frame size");
		
		std::cout << "input_vstream_frame_sizes[" << i << "] : " << input_vstream_frame_sizes[i] << std::endl;
		
		status = hailo_get_input_vstream_info(input_vstreams[i], &input_vstream_info[i]);
		REQUIRE_SUCCESS(status, l_deactivate_network_group, "Failed to get input vstream info");

	}  

	hef_input_width = input_vstream_info[0].shape.width;
	hef_input_height = input_vstream_info[0].shape.height;

	for (int i = 0; i < (int)output_vstreams_size; i++)
	{
		status = hailo_get_output_vstream_frame_size(output_vstreams[i], &output_vstream_frame_sizes[i]);
		REQUIRE_SUCCESS(status, l_deactivate_network_group, "Failed getting output virtual stream frame size");
		
		std::cout << "output_vstream_frame_sizes[" << i << "] : " << output_vstream_frame_sizes[i] << std::endl;
		
		status = hailo_get_output_vstream_info(output_vstreams[i], &output_vstream_info[i]);
		REQUIRE_SUCCESS(status, l_deactivate_network_group, "Failed to get output vstream info");
	}
	
	input_dummydata = std::vector<uint8_t>(input_vstream_frame_sizes[0]);
	
	for(int i=0;i<(int)output_vstreams_size;i++)
	{
		vstream_read_thread[i] = std::thread(vstream_read_thread_runner, i, output_vstreams[i], output_vstream_frame_sizes[i]);
	}
	yolo_output_collect_thread = std::thread(yolo_output_collecter_thread_runner, output_vstream_info);
	
	std::cout << "Entering Main Loop\n";
	while (true)
	{
		// if(!wait_for_finish)
		{			
			if(postprocessFrameQueue.size() < QUEUE_MAX && FrameDetections.size() < QUEUE_MAX)
			{
				std::future<void> result = std::async(std::launch::async, Inference_input_request_cb);
				result.get();
			}
			
		}
		
		
		
		
		
		// char c = cv::waitKey(1);
		// if (c == 'q') wait_for_finish = true;
		
		static int fps_mon = 0;
		
		if(timer_timeout(1000))
		{
			fps_mon = fps;
			fps = 0;
			
			std::cout << "fps_mon : " << fps_mon << std::endl;
			// std::cout << "postprocessFrameQueue: " << postprocessFrameQueue.size() << std::endl;
			// std::cout << "FrameDetections: " << FrameDetections.size() << std::endl;
		}
		
		
		
		if(!FrameDetections.empty())
		{
			cv::Mat &showframe = postprocessFrameQueue.front();
			HailoROIPtr roi = FrameDetections.front();
			
			Inferenced_Frame_cb(showframe.clone(), roi);
			
			FrameDetections.pop();
			postprocessFrameQueue.pop();
		}
		
		
		// if(wait_for_finish)
		// {
			// if(FrameDetections.empty() && postprocessFrameQueue.empty())
			// {
				// break;
			// }
		// }
		

		if (_terminate) break;
		
		usleep(1000);
	}
	std::cout << "Main Loop Exited\n";
	
	thread_terminate = true;
	
	std::cout << "yolo_output_collect_thread waiting\n";
	yolo_output_collect_thread.join();
	
	for(int i=0;i<10;i++)
	{
		usleep(1000 * 100); // 100 ms 
		
		bool all_dead = true;
		for(int n=0;n<OUTPUT_COUNT;n++)
		{
			if(!thread_terminated[n])
			{
				all_dead = false;
				break;
			}
		}
		
		if(all_dead) break;
		
		std::cout << "Write dummy buffer for Exiting\n";
		hailo_vstream_write_raw_buffer(input_vstreams[0], input_dummydata.data(), input_vstream_frame_sizes[0]);	
	}
	
	
	for(int i=0;i<(int)output_vstreams_size;i++)
	{
		std::cout << "vstream_read_thread " << i << " waiting\n";
		vstream_read_thread[i].join();
	}
	

	std::cout << "Releasing Hailo Resources\n";

	status = HAILO_SUCCESS;
	l_deactivate_network_group:
	(void)hailo_deactivate_network_group(activated_network_group);
	l_release_output_vstream:
	(void)hailo_release_output_vstreams(output_vstreams, output_vstreams_size);
	l_release_input_vstream:
	(void)hailo_release_input_vstreams(input_vstreams, input_vstreams_size);
	l_release_hef:
	(void)hailo_release_hef(hef);
	l_release_device:
	(void)hailo_release_device(device);
	l_exit:
	
	_terminate = true;
	std::cout << "inference_runner Terminated\n";
	return status;
}


std::thread inference_runner_thread;

void zaiv_start_inference_thread()
{
	inference_runner_thread = std::thread(inference_runner);
}

void zaiv_wait_for_inference_thread()
{
	inference_runner_thread.join();
}