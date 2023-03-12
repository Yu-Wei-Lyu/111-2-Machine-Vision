#include<opencv2/opencv.hpp>
#include<direct.h>
#define INDEX_COLOR_PATH  "./Image/Index color/"
#define RESIZE_PATH  "./Image/Resize/"
#define RESIZE_INTERPOLATION_PATH  "./Image/Resize(interpolation)/"

using namespace std;
using namespace cv;

void grayScaleImage(string fileName, bool showImage = true);
void binaryImage(string fileName, bool showImage = true);
void indexColorImage(string fileName, bool showImage = true);

int main()
{
	vector<string> imageList = { "House256.png", "House512.png", "JellyBeans.png", "Lena.png", "Mandrill.png", "Peppers.png" };
	//binaryImage(image_name.at(0), true);
	for (string& imageName : imageList) {
		indexColorImage(imageName, true);
	}
	waitKey();
	destroyAllWindows();
	return 0;
}

void grayScaleImage(string fileName, bool showImage) {
	const string SOURCE_FOLDER = "./Image/Source/";
	const string GRAYSCALE_FOLDER = "./Image/Grayscale/";
	Mat image = imread(SOURCE_FOLDER + fileName);
	Mat grayscaleImage = Mat(image.rows, image.cols, CV_8UC1);
	uchar* imagePtr = image.ptr<uchar>(0);
	uchar* gray = grayscaleImage.ptr<uchar>(0);
	int pixels = image.rows * image.cols;
	for (int pixel = 0; pixel < pixels; pixel++) {
		uchar blue = *imagePtr++, green = *imagePtr++, red = *imagePtr++;
		*gray++ = (0.3 * red) + (0.59 * green) + (0.11 * blue);
	}
	if (showImage) {
		imshow(SOURCE_FOLDER + fileName, image);
		imshow(GRAYSCALE_FOLDER + fileName, grayscaleImage);
	}
	imwrite(GRAYSCALE_FOLDER + fileName, grayscaleImage);
}

void binaryImage(string fileName, bool showImage) {
	const string GRAYSCALE_FOLDER = "./Image/Grayscale/";
	const string BINARY_FOLDER = "./Image/Binary/";
	Mat image = imread(GRAYSCALE_FOLDER + fileName);
	Mat binaryImage = Mat(image.rows, image.cols, CV_8UC1);
	uchar* imagePtr = image.ptr<uchar>(0);
	uchar* binary = binaryImage.ptr<uchar>(0);
	int pixels = image.rows * image.cols;
	for (int pixel = 0; pixel < pixels; pixel++) {
		*binary++ = (*imagePtr >= 128) ? 255 : 0;
		imagePtr = imagePtr + image.channels();
	}
	if (showImage) {
		imshow(GRAYSCALE_FOLDER + fileName, image);
		imshow(BINARY_FOLDER + fileName, binaryImage);
	}
	imwrite(BINARY_FOLDER + fileName, binaryImage);
}

void indexColorImage(string fileName, bool showImage) {
	const string SOURCE_FOLDER = "./Image/Source/";
	const string INDEX_COLOR_FOLDER = "./Image/Index color/";
	Mat image = imread(SOURCE_FOLDER + fileName);
	Mat myLUT = Mat(16, 16, CV_8UC1);
	Mat resultImage = Mat(image.rows, image.cols, CV_8UC1);
	uchar* myLUTPtr = myLUT.ptr<uchar>(0);
	uchar* imagePtr = image.ptr<uchar>(0);
	uchar* resultPtr = resultImage.ptr<uchar>(0);
	int pixels = image.rows * image.cols;
	for (int pixel = 0; pixel < pixels; pixel++) {
		uchar blue = *imagePtr++, green = *imagePtr++, red = *imagePtr++;
		*resultPtr++ = (*imagePtr >= 128) ? 255 : 0;
		imagePtr = imagePtr + image.channels();
	}
	if (showImage) {
		imshow(SOURCE_FOLDER + fileName, image);
		imshow(INDEX_COLOR_FOLDER + fileName, resultImage);
	}
	imwrite(INDEX_COLOR_FOLDER + fileName, resultImage);
}
