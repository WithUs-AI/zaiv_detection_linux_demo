
#include <unistd.h>
#include <signal.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <opencv2/opencv.hpp>
#include "inference.hpp"

#define ETH_HAILO_INTERFACE_NAME 		"eth1"
#define INPUT_CAMERA_INDEX 				0
#define INPUT_CAMERA_WIDTH 				640
#define INPUT_CAMERA_HEIGHT 			480
#define INPUT_VIDEO_FILE 				"./resources/detection.mp4"


#define NMS_HEF_FILE 					"./hef/yolov5_bdd.hef" // 6 labels

static std::map<uint8_t, std::string> yolov5_bdd_labels = {
    {0, "unlabeled"},
    {1, "1"},
    {2, "2"},
    {3, "3"},
    {4, "4"},
    {5, "5"},
    {6, "6"},
};
	

cv::VideoCapture video;
cv::Mat frame;

/*
* opencv 코드 기반으로 비디오 소스 가져오는 함수
* 카메라 가 인식되지 않을시 자체 비디오 를 재생
*/
bool input_source_get(cv::VideoCapture *v)
{
	*v = cv::VideoCapture(INPUT_CAMERA_INDEX);
	v->set(cv::CAP_PROP_FRAME_WIDTH, INPUT_CAMERA_WIDTH);
	v->set(cv::CAP_PROP_FRAME_HEIGHT, INPUT_CAMERA_HEIGHT);
	
	if(!v->isOpened())
	{
		std::cout << "camera 0 is not vaild\n";
		
		*v = cv::VideoCapture(INPUT_VIDEO_FILE);
		if(!v->isOpened())
		{
			std::cout << "input source not valid\n";
			return false;
		}
	}
	
	return true;
}

/*
* inference.cpp 코드에서 자동으로 호출되는 콜백함수, 입력 비디오 소스에서 프레임을 읽어 `zaiv_input_frame_for_inference` 함수로 cv::Mat 객체의 포인터를 전달해야한다
* 오류 발생시 `zaiv_terminate_inference_thread` 함수로 inference 동작을 종료시킨다
*/
void Inference_input_request_cb()
{
	if(video.read(frame))
	{
		zaiv_input_frame_for_inference(&frame);
	}
	else
	{
		video.release();
		
		if(!input_source_get(&video)) zaiv_terminate_inference_thread();
	}
}


std::queue<std::pair<cv::Mat, HailoROIPtr>> InferencedFrames;

/*
* Inference 완료시 해당 프레임과 Detection 정보를 호출한다 `InferencedFrames` Queue 에 넣고 main loop 에서 추출해 사용한다
*/
void Inferenced_Frame_cb(cv::Mat showframe, HailoROIPtr roi)
{
	// queue 를 사용해 main loop 에서 작업을 진행, inference 코드 동작의 느려짐을 방지
	std::pair<cv::Mat, HailoROIPtr> *ptrInferencedframe = new std::pair<cv::Mat, HailoROIPtr>(std::make_pair(showframe, roi));
	InferencedFrames.push(*ptrInferencedframe);
	delete ptrInferencedframe;
}

/*
* Ctrl-C(SIGINT) 신호 종료 기능 대응
*/
void intHandler(int dummy)
{
	zaiv_terminate_inference_thread();
}


int main(int argc, char **argv)
{
	std::cout << argv[0] << " Started\n";
	
	signal(SIGINT, intHandler);
	
	std::string eth_name;
	
	// 이더넷 보드 대응 코드 (이더넷 인터페이스 이름 강제 지정 기능) , PCI 는 무관
	if(argc == 2) eth_name = std::string(argv[1]);
	else eth_name = ETH_HAILO_INTERFACE_NAME;
	zaiv_set_eth_name(eth_name);

	// // 커스텀 HEF 파일 지정 기능 NMS HEF 사용가능
	// zaiv_set_hef_file(NMS_HEF_FILE);
	// // NMS HEF 파일의 라벨 지정
	// zaiv_set_hef_labels(yolov5_bdd_labels);
	
	if(!input_source_get(&video)) return 1;
	
	// Hailo Main Thread 시작
	zaiv_start_inference_thread();
	
	// cv::namedWindow("detection", cv::WND_PROP_FULLSCREEN);
	
	while(zaiv_inference_thread_alive())
	{
		
		if(!InferencedFrames.empty())
		{
			std::pair<cv::Mat, HailoROIPtr> &frontElement = InferencedFrames.front();
			
			cv::Mat &showframe = frontElement.first;
			HailoROIPtr roi = frontElement.second;
			
			/// ================================== 사용자 작성 코드 시작 ==================================
			
			zaiv_draw_all(&showframe, roi); // 현재 프레임에 Detection 데이터를 그려줌
			
			if(zaiv_timer_timeout_33ms()) // imshow 프레임 제한
			{
				cv::imshow("detection", showframe); // Detection 그려진 프레임 화면 표시
				char c = cv::waitKey(1);
				if(c == 'q') zaiv_terminate_inference_thread();
			}
			
			/*
			std::vector<HailoDetectionPtr> detections = hailo_common::get_hailo_detections(roi);

			std::cout << "detection counts: " << detections.size() << "\n";

			for (auto &detection : detections)
			{
				std::cout << "class id    : " << detection->get_class_id() << "\n";
				std::cout << "class label : " << detection->get_label() << "\n";
				std::cout << "confidence  : " << detection->get_confidence() << "\n";

				HailoBBox detection_bbox = detection->get_bbox();
				std::cout << "xmin        : " << detection_bbox.xmin() << "\n";
				std::cout << "ymin        : " << detection_bbox.ymin() << "\n";
				std::cout << "width       : " << detection_bbox.width() << "\n";
				std::cout << "height      : " << detection_bbox.height() << "\n";
			}
			*/
			
			/// ================================== 사용자 작성 코드 종료 ==================================
			
			InferencedFrames.pop();
		}
		
		usleep(1000);
	}
	
	// Hailo Main Thread 종료
	zaiv_terminate_inference_thread();
	
	// Hailo Main Thread 종료 대기
	zaiv_wait_for_inference_thread();
	
	std::cout << argv[0] << " Terminated\n";
	return 0;
}