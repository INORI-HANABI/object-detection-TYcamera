#include "../common/common.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>

#define BRIGHTNESS_NUM 3
#define STATISTIC_NUM 5
#define BRIGHTNESS_ADJUSTMENT_NUM 8
#define Manual_Adjustment_MaxNum 2

using namespace cv;
using namespace std;


void eventCallback(TY_EVENT_INFO *event_info, void *userdata)
{
	if (event_info->eventId == TY_EVENT_DEVICE_OFFLINE) {
		LOGD("=== Event Callback: Device Offline!");
		*(bool*)userdata = true;
	}
	else if (event_info->eventId == TY_EVENT_LICENSE_ERROR) {
		LOGD("=== Event Callback: License Error!");
	}
}




int main(int argc, char* argv[])
{
	std::string ID, IP;

	for (int i = 1; i < argc; i++){
		if (strcmp(argv[i], "-id") == 0){
			ID = argv[++i];
		}
		else if (strcmp(argv[i], "-ip") == 0) {
			IP = argv[++i];
		}
		else if (strcmp(argv[i], "-h") == 0){
			LOGI("Usage: SimpleView_Callback [-h] [-id <ID>]");
			return 0;
		}
	}

	LOGD("=== Init lib");
	int loop_index = 1;
	bool loop_exit = false;

	ASSERT_OK(TYInitLib());
	TY_VERSION_INFO ver;
	ASSERT_OK(TYLibVersion(&ver));
	LOGD("     - lib version: %d.%d.%d", ver.major, ver.minor, ver.patch);

	while (!loop_exit) {
		LOGD("==========================");
		LOGD("========== loop %d", loop_index++);
		LOGD("==========================");


		TY_INTERFACE_HANDLE hIface;
		TY_DEV_HANDLE hDevice;
		std::vector<TY_DEVICE_BASE_INFO> selected;

		int ret = 0;
		ret = selectDevice(TY_INTERFACE_USB, ID, IP, 1, selected);
		if (ret == TY_STATUS_ERROR) {
			MSleep(2000);
			continue;
		}

		ASSERT(selected.size() > 0);
		TY_DEVICE_BASE_INFO& selectedDev = selected[0];

		LOGD("=== Open interface: %s", selectedDev.iface.id);
		ASSERT_OK(TYOpenInterface(selectedDev.iface.id, &hIface));

		LOGD("=== Open device: %s", selectedDev.id);
		ret = TYOpenDevice(hIface, selectedDev.id, &hDevice);
		if (ret == TY_STATUS_ERROR) {
			continue;
		}

		int32_t allComps;
		ret = TYGetComponentIDs(hDevice, &allComps);
		if (ret == TY_STATUS_ERROR) {
			continue;
		}

		if (allComps & TY_COMPONENT_RGB_CAM){
			LOGD("=== Has RGB camera, open RGB cam");
			ret = TYEnableComponents(hDevice, TY_COMPONENT_RGB_CAM);
			if (ret == TY_STATUS_ERROR) {
				continue;
			}
		}

		if (allComps & TY_COMPONENT_IR_CAM_LEFT) {
			LOGD("Has IR left camera, open IR left cam");
			ASSERT_OK(TYEnableComponents(hDevice, TY_COMPONENT_IR_CAM_LEFT));
		}

		if (allComps & TY_COMPONENT_IR_CAM_RIGHT) {
			LOGD("Has IR right camera, open IR right cam");
			ASSERT_OK(TYEnableComponents(hDevice, TY_COMPONENT_IR_CAM_RIGHT));
		}
		if (allComps & TY_COMPONENT_BRIGHT_HISTO) {
			LOGD("=== Has bright histo component");
			ASSERT_OK(TYEnableComponents(hDevice, TY_COMPONENT_BRIGHT_HISTO));
		}

		LOGD("=== Configure components, open depth cam");
		int32_t componentIDs = TY_COMPONENT_DEPTH_CAM;
		ret = TYEnableComponents(hDevice, componentIDs);
		if (ret == TY_STATUS_ERROR) {
			continue;
		}

		LOGD("=== Configure feature, set resolution to 640x480.");
		int err = TYSetEnum(hDevice, TY_COMPONENT_DEPTH_CAM, TY_ENUM_IMAGE_MODE, TY_IMAGE_MODE_DEPTH16_640x480);
		if (ret == TY_STATUS_ERROR) {
			continue;
		}

		LOGD("=== Prepare image buffer");
		uint32_t frameSize;
		ret = TYGetFrameBufferSize(hDevice, &frameSize);
		if (ret == TY_STATUS_ERROR) {
			continue;
		}

		LOGD("     - Allocate & enqueue buffers");
		char* frameBuffer[2];
		frameBuffer[0] = new char[frameSize];
		frameBuffer[1] = new char[frameSize];
		LOGD("     - Enqueue buffer (%p, %d)", frameBuffer[0], frameSize);
		ret = TYEnqueueBuffer(hDevice, frameBuffer[0], frameSize);
		if (ret == TY_STATUS_ERROR) {
			continue;
		}
		LOGD("     - Enqueue buffer (%p, %d)", frameBuffer[1], frameSize);
		ret = TYEnqueueBuffer(hDevice, frameBuffer[1], frameSize);
		if (ret == TY_STATUS_ERROR) {
			continue;
		}

		bool device_offline = false;;
		LOGD("=== Register event callback");
		LOGD("Note: Callback may block internal data receiving,");
		LOGD("      so that user should not do long time work in callback.");
		ret = TYRegisterEventCallback(hDevice, eventCallback, &device_offline);
		if (ret == TY_STATUS_ERROR) {
			continue;
		}

		LOGD("=== Disable trigger mode");
		TY_TRIGGER_PARAM trigger;
		trigger.mode = TY_TRIGGER_MODE_OFF;
		ret = TYSetStruct(hDevice, TY_COMPONENT_DEVICE, TY_STRUCT_TRIGGER_PARAM, &trigger, sizeof(trigger));
		if (ret == TY_STATUS_ERROR) {
			continue;
		}


		LOGD("=== Start capture");
		ret = TYStartCapture(hDevice);
		if (ret == TY_STATUS_ERROR) {
			continue;
		}

		bool saveFrame = false;
		int saveIdx = 0;
		cv::Mat depth;
		cv::Mat leftIR;
		cv::Mat rightIR;
		cv::Mat color;

		cv::Mat temp;
		cv::Mat _renderedDepthREDBLUE;
		cv::Mat _renderedDepthGRAY;
		cv::Mat gray;
		cv::Mat	binary;
		DepthRender _render;

		LOGD("=== Wait for callback");
		bool exit_main = false;
		int count = 0;
		TY_FRAME_DATA frame;

		int min_dist = 0;
		int max_dist = 3000;
		int min_binary = 175;
		int max_binary = 255;

		const int Global_Adjustment_IRGains[BRIGHTNESS_ADJUSTMENT_NUM] = { 0x20, 0x35, 0x50, 0x70, 0x90, 0xB0, 0xD0, 0xF0 };
		int current_ir_gain = 200;  //当前增益
		int last_current_ir_gain = 0x00;  //上一次调整增益，防止震荡
		int brightness_areas[BRIGHTNESS_NUM];
		int total_brightness_points = 0x00;
		int statistic_times = STATISTIC_NUM;
		int stat_brightness_areas[BRIGHTNESS_NUM];
		int adjustment_pointer = 7;
		bool autoDownAdjustLocker = false;
		int manualAdjustCounter = Manual_Adjustment_MaxNum;
		for (int i = 0; i<BRIGHTNESS_NUM; i++) {
			brightness_areas[i] = 0x00;  //测试出来的CAM亮度由低到高
			stat_brightness_areas[i] = 0x00;  //连续统计亮度，为了稳定调节
		}
		bool brightAdjustLock = false;

		cv::Mat	tempBackground;
		cv::Mat bgMask_KNN;
		cv::Mat background;
		Ptr<BackgroundSubtractor> ptrKNN = createBackgroundSubtractorKNN(500,(400.0),false);
		Ptr<BackgroundSubtractorMOG2> mog = createBackgroundSubtractorMOG2(100, 25, false);
				
		while (!exit_main){
			int err = TYFetchFrame(hDevice, &frame, 1000);
			if (ret < 0) {
				break;
			}
						
			
			if (err == TY_STATUS_OK) {
				LOGD("=== Get frame %d", ++count);
				parseFrame(frame, &depth, &leftIR, &rightIR, &color);

				if (TYSetInt(hDevice, TY_COMPONENT_IR_CAM_LEFT, TY_INT_GAIN, current_ir_gain) == TY_STATUS_OK
					&& TYSetInt(hDevice, TY_COMPONENT_IR_CAM_RIGHT, TY_INT_GAIN, current_ir_gain) == TY_STATUS_OK){
					LOGD("=== SET TY_COMPONENT_IR_CAM = %d", current_ir_gain);
				}
				
				
				for (int i = 0; i < frame.validCount; i++){
					if (frame.image[i].componentID == TY_COMPONENT_BRIGHT_HISTO) {
						int32_t *ir_left_his, *ir_right_his;
						ir_left_his = (int32_t *)frame.image[i].buffer;
						ir_right_his = (int32_t *)frame.image[i].buffer + 256;

						memset(&brightness_areas, 0x00, BRIGHTNESS_NUM*sizeof(int));
						int i;
						for (i = 0; i<256; i++) {
							if (i < 21)
								brightness_areas[0] += ir_left_his[i];
							else if (i < 253)
								brightness_areas[1] += ir_left_his[i];
							else
								brightness_areas[2] += ir_left_his[i];
						}

						total_brightness_points = (brightness_areas[0] + brightness_areas[1] + brightness_areas[2]);
						brightness_areas[0] = 100 * (float)brightness_areas[0] / (float)total_brightness_points;
						brightness_areas[1] = 100 * (float)brightness_areas[1] / (float)total_brightness_points;
						brightness_areas[2] = 100 * (float)brightness_areas[2] / (float)total_brightness_points;

						printf("area[0] = %u, area[1] = %u, area[2] = %u\n", brightness_areas[0], brightness_areas[1], brightness_areas[2]);

						if (brightAdjustLock == false) {
							if (statistic_times > 0) //统计一次
							{
								statistic_times--;
								stat_brightness_areas[0] += brightness_areas[0];
								stat_brightness_areas[1] += brightness_areas[1];
								stat_brightness_areas[2] += brightness_areas[2];
							}
							else {
								stat_brightness_areas[0] = 0.5 + stat_brightness_areas[0] / STATISTIC_NUM;
								stat_brightness_areas[1] = 0.5 + stat_brightness_areas[1] / STATISTIC_NUM;
								stat_brightness_areas[2] = 0.5 + stat_brightness_areas[2] / STATISTIC_NUM;
								printf("***stat_area[0] = %u, stat_area[1] = %u, stat_area[2] = %u, current_ir_gain = %u\n",	stat_brightness_areas[0], stat_brightness_areas[1], stat_brightness_areas[2], current_ir_gain);
								brightAdjustLock = true;
							}
						}

						//自动调整增益，测试时放在这里，正式程序建议放到外面的while大循环里
						//auto adjust ir gains
						if (brightAdjustLock) {
							//测试代码，如果镜头过曝，逐渐调整亮度
							if (stat_brightness_areas[BRIGHTNESS_NUM - 1] > 20) {// || //最亮区域过大
								//stat_brightness_areas[0] < 5 || //最暗区域过小
								//(stat_brightness_areas[BRIGHTNESS_NUM - 1] > 5 && stat_brightness_areas[BRIGHTNESS_NUM - 2] < 40)) { //最亮区域过于集中，周围都比较暗
								if (adjustment_pointer > 0) {  //防止极端跳变,本行限定了只能降低亮度，不能升高亮度
									adjustment_pointer--;
									adjustment_pointer %= BRIGHTNESS_ADJUSTMENT_NUM; //调整亮度逐渐减小，直到最小
									autoDownAdjustLocker = true;  //自动下降调节标志置位，防止反复调上调下
								}
							}
							else if (stat_brightness_areas[0] > 95 || stat_brightness_areas[BRIGHTNESS_NUM - 1] < 1) { //如果画面太暗 0xf0
								if (autoDownAdjustLocker == false && manualAdjustCounter>0) { //如果当前增益值不是自动调下来的，则可以升高增益
									adjustment_pointer = BRIGHTNESS_ADJUSTMENT_NUM - 1; //直接调整亮度到最亮
									manualAdjustCounter--; //手工经验调节一次，计数器减1
								}
							}
							else if (stat_brightness_areas[0] > 90 || stat_brightness_areas[BRIGHTNESS_NUM - 1] < 1) { //如果画面太暗 0xD0
								if (autoDownAdjustLocker == false && manualAdjustCounter>0) { //如果当前增益值不是自动调下来的，则可以升高增益
									adjustment_pointer = BRIGHTNESS_ADJUSTMENT_NUM - 2; //直接调整亮度到次最亮
									manualAdjustCounter--; //手工经验调节一次，计数器减1
								}
							}
							else if (stat_brightness_areas[0] > 80 || stat_brightness_areas[BRIGHTNESS_NUM - 1] < 1) { //如果画面比较暗 0xB0
								if (autoDownAdjustLocker == false && manualAdjustCounter>0) { //如果当前增益值不是自动调下来的，则可以升高增益
									adjustment_pointer = BRIGHTNESS_ADJUSTMENT_NUM - 3; //直接调整亮度到次亮
									manualAdjustCounter--; //手工经验调节一次，计数器减1
								}
							}
							else if (stat_brightness_areas[0] > 70 || stat_brightness_areas[BRIGHTNESS_NUM - 1] < 1) { //如果画面次次暗 0x80
								if (autoDownAdjustLocker == false && manualAdjustCounter>0) { //如果当前增益值不是自动调下来的，则可以升高增益
									adjustment_pointer = BRIGHTNESS_ADJUSTMENT_NUM - 4; //直接调整亮度到次次亮
									manualAdjustCounter--; //手工经验调节一次，计数器减1
								}
							}
							else if (stat_brightness_areas[0] > 60 || stat_brightness_areas[BRIGHTNESS_NUM - 1] < 1) { //如果画面次次暗 0x40
								if (autoDownAdjustLocker == false && manualAdjustCounter>0) { //如果当前增益值不是自动调下来的，则可以升高增益
									adjustment_pointer = BRIGHTNESS_ADJUSTMENT_NUM - 5; //直接调整亮度到稍亮
									manualAdjustCounter--; //手工经验调节一次，计数器减1
								}
							}

							if (current_ir_gain != Global_Adjustment_IRGains[adjustment_pointer] && last_current_ir_gain != Global_Adjustment_IRGains[adjustment_pointer]) {  //防止来回切换亮度，没有必要
								if (TYSetInt(hDevice, TY_COMPONENT_IR_CAM_LEFT, TY_INT_GAIN, Global_Adjustment_IRGains[adjustment_pointer]) == TY_STATUS_OK
									&& TYSetInt(hDevice, TY_COMPONENT_IR_CAM_RIGHT, TY_INT_GAIN, Global_Adjustment_IRGains[adjustment_pointer]) == TY_STATUS_OK) {
									last_current_ir_gain = current_ir_gain;
									current_ir_gain = Global_Adjustment_IRGains[adjustment_pointer];
									LOGD("=== SET TY_COMPONENT_IR_CAM = %d", current_ir_gain);
								}
								else  {
									LOGD("=== SET TY_COMPONENT_IR_CAM failed");
								}
							}

							statistic_times = STATISTIC_NUM;
							memset(&stat_brightness_areas, 0x00, BRIGHTNESS_NUM*sizeof(int));
							brightAdjustLock = false;
						}
					}
					ASSERT_OK(TYEnqueueBuffer(hDevice, frame.userBuffer, frame.bufferSize));
				}
				

				if (!color.empty()){
					LOGI("Color format is %s", colorFormatName(TYImageInFrame(frame, TY_COMPONENT_RGB_CAM)->pixelFormat));
				}

				LOGD("=== Callback: Re-enqueue buffer(%p, %d)", frame.userBuffer, frame.bufferSize);
				ret = TYEnqueueBuffer(hDevice, frame.userBuffer, frame.bufferSize);
				if (ret == TY_STATUS_ERROR) {
					continue;
				}
				if (!leftIR.empty()){
					//imshow("L", leftIR);
				}
				if (!rightIR.empty()){
					//imshow("R", rightIR);
				}
				if (!depth.empty()){
					double t1 = cvGetTickCount();
					_render.SetColorRange(min_dist,max_dist);
					_render.SetColorTypeBLUERED();
					_renderedDepthREDBLUE = _render.Compute(depth);
					cv::imshow("_renderedDepthREDBLUE", _renderedDepthREDBLUE);
					_render.SetColorTypeGRAY();
					_renderedDepthGRAY = _render.Compute(depth);
					//cv::imshow("_renderedDepthGRAY", _renderedDepthGRAY);
					/*
					cvtColor(_renderedDepth, gray, COLOR_BGR2GRAY);
					threshold(gray, binary, min_binary, max_binary, THRESH_BINARY);
					Mat k = getStructuringElement(MORPH_ELLIPSE, Size(2, 2), Point(-1, -1));
					morphologyEx(binary, binary, MORPH_OPEN, k, Point(-1, -1), 3);
					morphologyEx(binary, binary, MORPH_CLOSE, k, Point(-1, -1), 3);
					double t2 = cvGetTickCount();

					mog->apply(binary, temp, 0.005);

					
					ptrKNN->apply(binary, bgMask_KNN);
					ptrKNN->getBackgroundImage(tempBackground);
					if (count % 10 == 0){
					background = tempBackground;
					}
					temp = binary - background;
					
					
					
					Mat labels = Mat::zeros(temp.size(), temp.type());
					Mat stats, centroids;
					int num_labels = connectedComponentsWithStats(temp, labels, stats, centroids, 8);
					for (int i = num_labels - 1; i > 0; i--) {
						Vec2d pt = centroids.at<Vec2d>(i, 0);
						int x = stats.at<int>(i, CC_STAT_LEFT);
						int y = stats.at<int>(i, CC_STAT_TOP);
						int width = stats.at<int>(i, CC_STAT_WIDTH);
						int height = stats.at<int>(i, CC_STAT_HEIGHT);
						int area = stats.at<int>(i, CC_STAT_AREA);

						if (width * height > 8000){
							rectangle(_renderedDepth, Rect(x, y, width, height), Scalar(255, 0, 255), 8, 8, 0);

							//Point center = Point(x + (width / 2), y + (height / 2));						
						}

					}
					double t3 = cvGetTickCount();
					cv::imshow("outPut", _renderedDepth);
					double t4 = cvGetTickCount();
					double dethrender_comput = (t2 - t1)*1000.0 / getTickFrequency();
					double forRect = (t3 - t2)*1000.0 / getTickFrequency();
					double withoutShow = (t3 - t1)*1000.0 / getTickFrequency();
					double withShow = (t4 - t1)*1000.0 / getTickFrequency();
					printf("\n");
					printf("dethrender_comput time = %gms\n", dethrender_comput);
					printf("forRect time = %gms\n", forRect);
					printf("withoutShow time = %gms\n", withoutShow);
					printf("withShow time = %gms\n", withShow);
					printf("\n");
					*/
					
					
				    

				}
				if (saveFrame && !depth.empty() && !leftIR.empty() && !rightIR.empty()){             
					LOGI(">>>> save frame %d", saveIdx);

					char f[32];
					sprintf(f, "outpic/%dL-%d.jpg", saveIdx, current_ir_gain);
					imwrite(f, leftIR);
					sprintf(f, "outpic/%dR-%d.jpg", saveIdx, current_ir_gain);
					imwrite(f, rightIR);
					sprintf(f, "outpic/%dDRB-%d.jpg", saveIdx, current_ir_gain);
					imwrite(f, _renderedDepthREDBLUE);
					sprintf(f, "outpic/%dDG-%d.jpg", saveIdx, current_ir_gain);
					imwrite(f, _renderedDepthGRAY);
					saveIdx++;
					saveFrame = false;
				}
			}

			if (device_offline){
				LOGI("Found device offline");
				break;
			}

			int key = cv::waitKey(10);
			switch (key & 0xff){
			case 0xff:
				break;
			case 'g':
				scanf("%d", &current_ir_gain);
				break;
			case 'q':
				exit_main = true;
				break;
			case 'd':
				printf("please input min_dist:");
				scanf("%d", &min_dist);
				printf("please input max_dist:");
				scanf("%d", &max_dist);
				break;
			case 'b':
				printf("please input min_binary:");
				scanf("%d", &min_binary);
				printf("please input max_binary:");
				scanf("%d", &max_binary);
				break;
			case 's':
				saveFrame = true;
				break;
			case 'x':
				exit_main = true;
				loop_exit = true;
				break;
			default:
				LOGD("Unmapped key %d", key);
			}
		}

		if (device_offline) {
			LOGI("device offline release resource");
		}
		else {
			LOGI("normal exit");
		}
		ret = TYStopCapture(hDevice);
		if (ret == TY_STATUS_ERROR) {
			continue;
		}
		ret = TYCloseDevice(hDevice);
		if (ret == TY_STATUS_ERROR) {
			continue;
		}
		ret = TYCloseInterface(hIface);
		if (ret == TY_STATUS_ERROR) {
			continue;
		}
		delete frameBuffer[0];
		delete frameBuffer[1];
	}

	LOGD("=== Deinit lib");
	ASSERT_OK(TYDeinitLib());
	LOGD("=== Main done!");
	return 0;
}


