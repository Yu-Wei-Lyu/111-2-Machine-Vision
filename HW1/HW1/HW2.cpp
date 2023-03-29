#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat getGrayScaleImage(const Mat& image);
Mat getBinaryImage(const Mat& image, int threshold = 128);

int main() {
	cout << "[Main] Start to processing images, please wait..." << endl;
	vector<string> folderList = { "../Image/Source/", "../Image/Binary/" };
	vector<int> binaryThresholdList = {128, 240, 0, 0 };
	vector<string> imageList = { /*"1.png", "2.png",*/ "3.png"/*, "4.png"*/};
	for (int i = 0; i < imageList.size(); i++) {
		Mat image = imread(folderList.at(0) + imageList.at(i));
		//imshow("Source " + imageList.at(i), image);
		Mat binary = getBinaryImage(image, 85); // binaryThresholdList.at(i)
		imshow("Binary " + imageList.at(i), binary);
		imwrite(folderList.at(1) + imageList.at(i), binary);
	}
	cout << "[Main] All image processing complete." << endl;
	waitKey();
	destroyAllWindows();
	return 0;
}

Mat getGrayScaleImage(const Mat& image) {
	Mat grayImage = Mat(image.rows, image.cols, CV_8UC1);
	const uchar* imagePtr;
	uchar* gray;
	for (int row = 0; row < image.rows; row++) {
		imagePtr = image.ptr<uchar>(row);
		gray = grayImage.ptr<uchar>(row);
		for (int col = 0; col < image.cols; col++) {
			uchar blue = *imagePtr++, green = *imagePtr++, red = *imagePtr++;
			*gray++ = (0.3 * red) + (0.59 * green) + (0.11 * blue);
		}
	}
	imshow("Gray", grayImage);
	return grayImage;
}

Mat getBinaryImage(const Mat& image, int threshold) {
	Mat grayImage = image;
	if (image.channels() == 3)
		grayImage = getGrayScaleImage(image);
	Mat binaryImage = Mat(image.rows, image.cols, CV_8UC1);
	const uchar* imagePtr;
	uchar* binary;
	for (int row = 0; row < grayImage.rows; row++) {
		imagePtr = grayImage.ptr<uchar>(row);
		binary = binaryImage.ptr<uchar>(row);
		for (int col = 0; col < grayImage.cols; col++) {
			*binary++ = (*imagePtr++ >= threshold) ? 255 : 0;
		}
	}
	return binaryImage;
}