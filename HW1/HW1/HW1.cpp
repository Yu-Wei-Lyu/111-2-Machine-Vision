#include<opencv2/opencv.hpp>
#include<direct.h>

using namespace std;
using namespace cv;

void prepareFolder();
Mat getMyColorImage();
Mat getGrayScaleImage(const Mat& image);
Mat getBinaryImage(const Mat& image);
Mat getIndexColorImage(const Mat& image);

int main()
{
	const string SOURCE_FOLDER = "./Image/Source/";
	const string GRAYSCALE_FOLDER = "./Image/Grayscale/";
	const string BINARY_FOLDER = "./Image/Binary/";
	const string INDEX_COLOR_FOLDER = "./Image/Index color/";
	const string RESIZE_FOLDER = "./Image/Resize/";
	string RESIZE_INTERPOLATION_FOLDER = "./Image/Resize(interpolation)/";
	prepareFolder();
	vector<string> imageList = { "House256.png", "House512.png", "JellyBeans.png", "Lena.png", "Mandrill.png", "Peppers.png" };
	//vector<string> imageList = { "Lena.png" };
	Mat colorMap = getMyColorImage();
	for (string& imageName : imageList) {
		Mat image = imread(SOURCE_FOLDER + imageName);
		imshow("Source", image);
		Mat gray = getGrayScaleImage(image);
		imshow("Gray", gray);
		imwrite(GRAYSCALE_FOLDER + imageName, gray);
		Mat binary = getBinaryImage(gray);
		imshow("Binary", binary);
		imwrite(BINARY_FOLDER + imageName, binary);
		Mat indexColor = getIndexColorImage(image);
		imshow("Index-Color", indexColor);
		imwrite(INDEX_COLOR_FOLDER + imageName, indexColor);
	}
	waitKey();
	destroyAllWindows();
	return 0;
}

void prepareFolder() {
	_mkdir("./Image/");
	_mkdir("./Image/Source/");
	_mkdir("./Image/Grayscale/");
	_mkdir("./Image/Binary/");
	_mkdir("./Image/Index color/");
	_mkdir("./Image/Resize/");
	_mkdir("./Image/Resize(interpolation)/");
}

Mat getMyColorImage() {
	Mat myColorMap = Mat(16, 16, CV_8UC3);
	uchar* p = myColorMap.ptr<uchar>(0);
	for (int i = 0; i < 256; i += 51) {
		for (int j = 0; j < 256; j += 51) {
			for (int k = 0; k < 256; k += 51) {
				*p++ = i, *p++ = j, *p++ = k;
			}
		}
	}
	imwrite("./Image/myColorMap.png", myColorMap);
	return myColorMap;
}

Mat getGrayScaleImage(const Mat& image) {
	Mat grayImage = Mat(image.rows, image.cols, CV_8UC1);
	int rows = image.rows * image.cols;
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
	return grayImage;
}

Mat getBinaryImage(const Mat& image) {
	Mat grayImage = image;
	if (image.channels() == 3) {
		grayImage = getGrayScaleImage(image);
	}
	Mat binaryImage = Mat(image.rows, image.cols, CV_8UC1);
	const uchar* imagePtr = grayImage.ptr<uchar>(0);
	uchar* binary = binaryImage.ptr<uchar>(0);
	int rows = image.rows * image.cols;
	for (int row = 0; row < image.rows; row++) {
		*binary++ = (*imagePtr >= 128) ? 255 : 0;
		imagePtr++;
	}
	return binaryImage;
}

Mat getIndexColorImage(const Mat& image) {
	Mat resultImage = Mat(image.rows, image.cols, CV_8UC3);
	const uchar* imagePtr = image.ptr<uchar>(0);
	uchar* resultPtr = resultImage.ptr<uchar>(0);
	int rows = image.rows * image.cols * image.channels();
	for (int row = 0; row < image.rows; row++) {
		int remain = *imagePtr % 51;
		*resultPtr = *imagePtr - remain;
		if (remain >= 25)
			*resultPtr += 51;
		resultPtr++;
		imagePtr++;
	}
	return resultImage;
}
