#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <algorithm>
#include <math.h>
#include <photo\photo.hpp>



using namespace cv;
using namespace std;

struct Gain{
	int index;
	int value;
};

bool cmp(Gain a, Gain b){
	return a.value > b.value;
}




int main()
{
	vector<Gain> binary_gain;
	int current_binarycut[4] = { 0 };
	int last_current_binarycut[4] = { 0 };
	VideoCapture video1;
	VideoCapture video2;
	video1.open("videos/2DG.avi");	
	video2.open("videos/2DBR.avi");

	if (!video1.isOpened() || !video2.isOpened())    // 判断是否打开成功
	{
		printf("open video file failed.\n ");
		return -1;
	}
	
	Mat frame1;
	Mat frame2;
	Mat gray;
	Mat binary[4];

	Mat k = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	int frame_num = 0;
	int total_frame_num = video1.get(cv::CAP_PROP_FRAME_COUNT);
	printf("%d\n", total_frame_num);

	while (frame_num != total_frame_num){
		video1 >> frame1;
		video2 >> frame2;
		frame_num++;

		
		if (frame_num >= 1)
		{
			cvtColor(frame1, frame1, COLOR_BGR2GRAY);
			morphologyEx(frame1, frame1, MORPH_OPEN, k, Point(-1, -1), 3);
			morphologyEx(frame1, frame1, MORPH_CLOSE, k, Point(-1, -1), 3);
			
			int image_count = 1;//要计算直方图的图像的个数
			int channels[1] = { 0 };//图像的通道'
			Mat out;//计算所得直方图
			int dims = 1;//得到直方图的维数
			int histsize[1] = { 256 };//直方图横坐标的子区间数
			float hrange[2] = { 0, 255 };//区间的总范围
			const float *ranges[1] = { hrange };//指针数组
			calcHist(&frame1, image_count, channels, Mat(), out, dims, histsize, ranges);

			binary_gain.clear();
			int gray_index = 0;
			for (int i = 0; i < out.rows; i++){
				for (int j = 0; j < out.cols; j++){
					float gray_value = out.at<float>(i, j);
					//printf("%d : %d \n", gray_index, gray_value);
					gray_index += 1;
					if (gray_index < 100)
						continue;
					else{
						Gain forgray;
						forgray.index = gray_index;
						forgray.value = gray_value;
						binary_gain.push_back(forgray);
					}

				}
			}

			sort(binary_gain.begin(), binary_gain.end(), cmp);
			if (!binary_gain.empty()){
				if (abs(binary_gain[1].index - last_current_binarycut[1]) > 10){
					current_binarycut[1] = (binary_gain[1].index + last_current_binarycut[1])/2;
					last_current_binarycut[1] = current_binarycut[1];
				}
				else{
					current_binarycut[1] = last_current_binarycut[1];
				}
			}
			else{
				current_binarycut[1] = last_current_binarycut[1];
			}

			printf("%d : %d \n", binary_gain[1].index, binary_gain[1].value);
			threshold(frame1, binary[1], current_binarycut[1], 255, THRESH_BINARY);
			morphologyEx(binary[1], binary[1], MORPH_ERODE, k, Point(-1, -1), 3);

			
			Mat labels = Mat::zeros(binary[1].size(), binary[1].type());
			Mat stats, centroids;
			int num_labels = connectedComponentsWithStats(binary[1], labels, stats, centroids, 4);
			for (int i = 1; i < num_labels ; i++) {
				Vec2d pt = centroids.at<Vec2d>(i, 0);
				int x = stats.at<int>(i, CC_STAT_LEFT);
				int y = stats.at<int>(i, CC_STAT_TOP);
				int width = stats.at<int>(i, CC_STAT_WIDTH);
				int height = stats.at<int>(i, CC_STAT_HEIGHT);
				int area = stats.at<int>(i, CC_STAT_AREA);
				if (x + width > binary[1].cols || y + height > binary[1].rows){
					continue;
				}
				cout << "x:" << x << "y:" << y << "width:" << width << "height:" << height << "area:" << area << endl;
				//Point2d center = Point(x + (width / 2), y + (height / 2));
				if (width * height > 8000 ){
					rectangle(frame2, Rect(x, y, width, height), Scalar(255, 0, 255), 8, 8, 0);
				}
			}
			
			
			//显示图像  
			imshow("camera", frame2);
			imshow("moving area", binary[1]);
			

		}
		
		
		waitKey(100);
	}
	

	
	video1.release();
	video2.release();



	return 0;

}
