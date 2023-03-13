#include<opencv2/opencv.hpp>
#include<direct.h>

using namespace std;
using namespace cv;

Mat getGrayScaleImage(const Mat& image);
Mat getBinaryImage(const Mat& image);
Mat getIndexColorImage(const Mat& image, Mat& lookUpTable);

int main()
{
	const string SOURCE_FOLDER = "./Image/Source/";
	const string GRAYSCALE_FOLDER = "./Image/Grayscale/";
	const string INDEX_COLOR_FOLDER = "./Image/Index color/";
	const string RESIZE_FOLDER = "./Image/Resize/";
	const string RESIZE_INTERPOLATION_FOLDER = "./Image/Resize(interpolation)/";
	vector<string> imageList = { "House256.png", "House512.png", "JellyBeans.png", "Lena.png", "Mandrill.png", "Peppers.png" };
	Mat image = imread(SOURCE_FOLDER + "test.png");
	imshow("Source", image);
	Mat gray = getGrayScaleImage(image);
	imshow("Gray", gray);
	Mat binary = getBinaryImage(image);
	imshow("Binary", binary);
	Mat lookUpTable = Mat(16, 16, CV_8UC3);
	Mat indexColor = getIndexColorImage(image, lookUpTable);
	imshow("Index-Color", indexColor);
	imshow("LookUpTable", lookUpTable);
	//Mat indexColor = getIndexColorImage(image, lookUpTable);
	//imshow("Binary", binary);
	
	//grayScaleImage(image, imageList.at(0), true);
	//for (string& imageName : imageList) {
	//	Mat image = imread(SOURCE_FOLDER + imageName);
	//	Mat gray = grayScaleImage(image);
	//	imwrite("./Image/Grayscale/" + imageName, gray);
	//}
	waitKey();
	destroyAllWindows();
	return 0;
}

Mat getGrayScaleImage(const Mat& image) {
	Mat grayImage = Mat(image.rows, image.cols, CV_8UC1);
	const uchar* imagePtr = image.ptr<uchar>(0);
	uchar* gray = grayImage.ptr<uchar>(0);
	int pixels = image.rows * image.cols;
	for (int pixel = 0; pixel < pixels; pixel++) {
		uchar blue = *imagePtr++, green = *imagePtr++, red = *imagePtr++;
		*gray++ = (0.3 * red) + (0.59 * green) + (0.11 * blue);
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
	int pixels = image.rows * image.cols;
	for (int pixel = 0; pixel < pixels; pixel++) {
		*binary++ = (*imagePtr >= 128) ? 255 : 0;
		imagePtr++;
	}
	return binaryImage;
}

Mat getIndexColorImage(const Mat& image, Mat& lookUpTable) {
	Mat resultImage = Mat(image.rows, image.cols, CV_8UC1);
	const uchar* imagePtr = image.ptr<uchar>(0);
	uchar* resultPtr = resultImage.ptr<uchar>(0);
	int pixels = image.rows * image.cols;
	lookUpTable.at<Vec3b>(0, 0) = image.at<Vec3b>(0, 0);
	int LUTPixels = 1;
	for (int pixel = 0; pixel < pixels; pixel++) {
		uchar blue = *imagePtr++, green = *imagePtr++, red = *imagePtr++;
		//cout << pixel << " : " << LUTPixels << " | " << (int)blue << "," << (int)green << "," << (int)red << endl;
		uchar* lookUpPtr = lookUpTable.ptr<uchar>(0);
		bool IsAtLUT = false;
		for (int LUTPixel = 0; LUTPixel < LUTPixels; LUTPixel++) {
			uchar LUTBlue = *lookUpPtr++, LUTGreen = *lookUpPtr++, LUTRed = *lookUpPtr++;
			if (blue == LUTBlue && green == LUTGreen && red == LUTRed) {
				*resultPtr = LUTPixel;
				//cout << pixel << " : found " << LUTPixels << " | " << (int)LUTBlue << "," << (int)LUTGreen << "," << (int)LUTRed << endl;
				IsAtLUT = true;
			}
		}
		if (!IsAtLUT) {
			//cout << "[!] Not in LUT" << endl;
			LUTPixels+=1;
			if (LUTPixels == 256) {
				cout << "LUT up to 256 colors" << endl;
				return resultImage;
			}
			lookUpTable.at<Vec3b>(LUTPixels / 16, LUTPixels % 16)[0] = blue;
			lookUpTable.at<Vec3b>(LUTPixels / 16, LUTPixels % 16)[1] = green;
			lookUpTable.at<Vec3b>(LUTPixels / 16, LUTPixels % 16)[2] = red;
			
			*resultPtr = LUTPixels;
		}
		resultPtr++;
	}
	return resultImage;
}
