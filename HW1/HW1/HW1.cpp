#include<opencv2/opencv.hpp>
#include<direct.h>

using namespace std;
using namespace cv;

void prepareFolder(vector<string> folderList);
Mat getGrayScaleImage(const Mat& image);
Mat getBinaryImage(const Mat& image);
Vec3b getColorFromMap(Mat& colorMap, int& colorPixels, const Vec3b bgr, int& threshould);
Mat getIndexColorImage(const Mat& image, Mat& colorMap, int threshould);
Mat getDoubleSizeImage(const Mat& image);
Mat getHalfSizeImage(const Mat& image);
Mat getDoubleSizeRoundImage(const Mat& image);
Mat getHalfSizeRoundImage(const Mat& image);
uchar getBilinearValue(uchar valueClose, uchar valueFar);
vector<uchar> getBilinearList(vector<uchar> source);

int main()
{
	cout << "[Main] Start to processing images, please wait..." << endl;
	vector<string> folderList = { "../Image/Source/", "../Image/Grayscale/", "../Image/Binary/", "../Image/Index color/", "../Image/Resize/", "../Image/Resize(interpolation)/" };
	vector<int> thresholdList = { 14, 20, 13, 15, 25, 21 };
	vector<string> imageList = { "House256.png", "House512.png", "JellyBeans.png", "Lena.png", "Mandrill.png", "Peppers.png" };
	prepareFolder(folderList);
	for (int i = 0; i < imageList.size(); i++) {
		Mat image = imread(folderList.at(0) + imageList.at(i));
		imshow("Source " + imageList.at(i), image);
		Mat gray = getGrayScaleImage(image);
		imshow("Gray", gray);
		imwrite(folderList.at(1) + imageList.at(i), gray);
		Mat binary = getBinaryImage(gray);
		imshow("Binary", binary);
		imwrite(folderList.at(2) + imageList.at(i), binary);
		Mat myColorMap;
		Mat indexColor2 = getIndexColorImage(image, myColorMap, thresholdList.at(i));
		imshow("Index-Color", indexColor2);
		myColorMap = getDoubleSizeImage(getDoubleSizeImage(getDoubleSizeImage(myColorMap)));
		imwrite(folderList.at(3) + imageList.at(i), indexColor2);
		imshow("Color-Map-Limit", myColorMap);
		imwrite(folderList.at(3) + "color_map_" + imageList.at(i), myColorMap);
		Mat scaledHalfImage = getHalfSizeImage(image);
		imshow("Scale half size image " + imageList.at(i), scaledHalfImage);
		imwrite(folderList.at(4) + "half_size_" + imageList.at(i), scaledHalfImage);
		Mat scaledDoubleImage = getDoubleSizeImage(image);
		imshow("Scale double size image " + imageList.at(i), scaledDoubleImage);
		imwrite(folderList.at(4) + "double_size_" + imageList.at(i), scaledDoubleImage);
		Mat scaledHalfRoundImage = getHalfSizeRoundImage(image);
		imshow("Scale half size image (round) " + imageList.at(i), scaledHalfRoundImage);
		imwrite(folderList.at(5) + "half_size_" + imageList.at(i), scaledHalfRoundImage);
		Mat scaledDoubleRoundImage = getDoubleSizeRoundImage(image);
		imshow("Scale double size image (round) " + imageList.at(i), scaledDoubleRoundImage);
		imwrite(folderList.at(5) + "double_size_" + imageList.at(i), scaledDoubleRoundImage);
	}
	cout << "[Main] All image processing complete." << endl;
	waitKey();
	destroyAllWindows();
	return 0;
}

void prepareFolder(vector<string> folderList) {
	for (string& folder : folderList) {
		if (_mkdir(folder.c_str()) != 0) {
			cerr << "[Prepare folder] " << strerror(errno) << ", it won't cover (" << folder << ")" << endl;
		}
	}
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
	return grayImage;
}

Mat getBinaryImage(const Mat& image) {
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
			*binary++ = (*imagePtr++ >= 128) ? 255 : 0;
		}
	}
	return binaryImage;
}

Vec3b getColorFromMap(Mat& colorMap, int& colorPixels, const Vec3b bgr, int& threshould) {
	uchar sourceBlue = bgr[0], sourceGreen = bgr[1], sourceRed = bgr[2];
	uchar* colorMapPtr = colorMap.ptr<uchar>(0);
	bool isInColorMap = false;
	for (int i = 0; i < colorPixels; i++) {
		uchar colorMapBlue = *colorMapPtr++, colorMapGreen = *colorMapPtr++, colorMapRed = *colorMapPtr++;
		int colorGap = sqrt(pow((colorMapBlue - sourceBlue), 2) + pow((colorMapGreen - sourceGreen), 2) + pow((colorMapRed - sourceRed), 2));
		if (colorGap <= threshould) {
			isInColorMap = true;
			break;
		}
	}
	if (!isInColorMap) {
		*colorMapPtr++ = sourceBlue;
		*colorMapPtr++ = sourceGreen;
		*colorMapPtr++ = sourceRed;
		colorPixels++;
	}
	return Vec3b(*(colorMapPtr - 3), *(colorMapPtr - 2), *(colorMapPtr - 1));
}

Mat getIndexColorImage(const Mat& image, Mat& colorMap, int threshold) {
	colorMap = Mat(16, 16, CV_8UC3);
	Mat indexColorImage = Mat(image.rows, image.cols, CV_8UC3);
	int colorMapSize = 1;
	colorMap.at<Vec3b>(0, 0) = image.at<Vec3b>(0, 0);
	for (int row = 0; row < image.rows; row++) {
		for (int col = 0; col < image.cols; col++) {
			const Vec3b originPixel = image.at<Vec3b>(row, col);
			indexColorImage.at<Vec3b>(row, col) = getColorFromMap(colorMap, colorMapSize, originPixel, threshold);
			if (colorMapSize > 256) {
				cerr << "[Index color image] Threshold too low, the image process incomplete, please set higher threshold." << endl;
				return indexColorImage;
			}
		}
	}
	return indexColorImage;
}

Mat getDoubleSizeImage(const Mat& image) {
	int doubleRow = image.rows * 2, doubleCol = image.cols * 2;
	Mat scaledImage = Mat(doubleRow, doubleCol, CV_8UC3);
	for (int row = 0; row < doubleRow; row++) {
		const uchar* imagePtr = image.ptr<uchar>(row / 2);
		uchar* scaledPtr = scaledImage.ptr<uchar>(row);
		for (int col = 0; col < image.cols; col++) {
			uchar imageBlue = *imagePtr++, imageGreen = *imagePtr++, imageRed = *imagePtr++;
			for (int pixelColor = 0; pixelColor < 2; pixelColor++) {
				*scaledPtr++ = imageBlue;
				*scaledPtr++ = imageGreen;
				*scaledPtr++ = imageRed;
			}
		}
	}
	return scaledImage;
}

Mat getHalfSizeImage(const Mat& image) {
	int halfRow = image.rows / 2, halfCol = image.cols / 2;
	Mat scaledImage = Mat(halfRow, halfCol, CV_8UC3);
	for (int row = 0; row < halfRow; row++) {
		const uchar* imagePtr = image.ptr<uchar>(row * 2);
		uchar* scaledPtr = scaledImage.ptr<uchar>(row);
		for (int col = 0; col < halfCol; col++) {
			uchar imageBlue = *imagePtr++, imageGreen = *imagePtr++, imageRed = *imagePtr++;
			*scaledPtr++ = imageBlue;
			*scaledPtr++ = imageGreen;
			*scaledPtr++ = imageRed;
			imagePtr += 3;
		}
	}
	return scaledImage;
}

Mat getDoubleSizeRoundImage(const Mat& image) {
	int doubleRow = image.rows * 2, doubleCol = image.cols * 2;
	Mat scaledImage = Mat(doubleRow, doubleCol, CV_8UC3);
	for (int row = 0; row < image.rows / 2; row++) {
		const uchar* imagePtr1 = image.ptr<uchar>(row * 2);
		const uchar* imagePtr2 = image.ptr<uchar>(row * 2 + 1);
		uchar* scaledPtr1 = scaledImage.ptr<uchar>(row * 4);
		uchar* scaledPtr2 = scaledImage.ptr<uchar>(row * 4 + 1);
		uchar* scaledPtr3 = scaledImage.ptr<uchar>(row * 4 + 2);
		uchar* scaledPtr4 = scaledImage.ptr<uchar>(row * 4 + 3);
		for (int col = 0; col < image.cols / 2; col++) {
			uchar leftTopBlue = *imagePtr1++, leftTopGreen = *imagePtr1++, leftTopRed = *imagePtr1++;
			uchar rightTopBlue = *imagePtr1++, rightTopGreen = *imagePtr1++, rightTopRed = *imagePtr1++;
			uchar leftBottomBlue = *imagePtr2++, leftBottomGreen = *imagePtr2++, leftBottomRed = *imagePtr2++;
			uchar rightBottomBlue = *imagePtr2++, rightBottomGreen = *imagePtr2++, rightBottomRed = *imagePtr2++;
			vector<uchar> blueBilinear = getBilinearList(vector<uchar>{ leftTopBlue, rightTopBlue, leftBottomBlue, rightBottomBlue });
			vector<uchar> greenBilinear = getBilinearList(vector<uchar>{ leftTopGreen, rightTopGreen, leftBottomGreen, rightBottomGreen });
			vector<uchar> redBilinear = getBilinearList(vector<uchar>{ leftTopRed, rightTopRed, leftBottomRed, rightBottomRed });
			for (int i = 0; i < 4; i++) {
				*scaledPtr1++ = blueBilinear.at(i);
				*scaledPtr1++ = greenBilinear.at(i);
				*scaledPtr1++ = redBilinear.at(i);
			}
			for (int i = 4; i < 8; i++) {
				*scaledPtr2++ = blueBilinear.at(i);
				*scaledPtr2++ = greenBilinear.at(i);
				*scaledPtr2++ = redBilinear.at(i);
			}
			for (int i = 8; i < 12; i++) {
				*scaledPtr3++ = blueBilinear.at(i);
				*scaledPtr3++ = greenBilinear.at(i);
				*scaledPtr3++ = redBilinear.at(i);
			}
			for (int i = 12; i < 16; i++) {
				*scaledPtr4++ = blueBilinear.at(i);
				*scaledPtr4++ = greenBilinear.at(i);
				*scaledPtr4++ = redBilinear.at(i);
			}
		}
	}
	return scaledImage;
}

vector<uchar> getBilinearList(vector<uchar> source) {
	vector<uchar> result(16, -1);
	result.at(0) = source.at(0);
	result.at(3) = source.at(1);
	result.at(12) = source.at(2);
	result.at(15) = source.at(3);
	vector<int> sidePixel{ 1, 2, 13, 14 };
	for (int index = 0; index < sidePixel.size(); index++) {
		int pixelIndex = sidePixel.at(index);
		int closeIndex = pixelIndex + 1 * pow(-1, pixelIndex % 2);
		int farIndex = pixelIndex - 2 * pow(-1, pixelIndex % 2);
		result.at(pixelIndex) = getBilinearValue(result.at(closeIndex), result.at(farIndex));
	}
	for (int index = 4; index < 12; index++) {
		int closeIndex = index - 4 * pow(-1, index / 8);
		int farIndex = index + 8 * pow(-1, index / 8);
		result.at(index) = getBilinearValue(result.at(closeIndex), result.at(farIndex));
	}
	return result;
}

uchar getBilinearValue(uchar valueClose, uchar valueFar) {
	return (uchar)(valueClose * (2.0 / 3.0) + valueFar * (1.0 / 3.0));
}

Mat getHalfSizeRoundImage(const Mat& image) {
	int halfRow = image.rows / 2, halfCol = image.cols / 2;
	Mat scaledImage = Mat(halfRow, halfCol, CV_8UC3);
	for (int row = 0; row < halfCol; row++) {
		const uchar* imagePtr1 = image.ptr<uchar>(row * 2);
		const uchar* imagePtr2 = image.ptr<uchar>(row * 2 + 1);
		uchar* scaledPtr = scaledImage.ptr<uchar>(row);
		for (int col = 0; col < halfCol; col++) {
			uchar leftTopBlue = *imagePtr1++, leftTopGreen = *imagePtr1++, leftTopRed = *imagePtr1++;
			uchar rightTopBlue = *imagePtr1++, rightTopGreen = *imagePtr1++, rightTopRed = *imagePtr1++;
			uchar leftBottomBlue = *imagePtr2++, leftBottomGreen = *imagePtr2++, leftBottomRed = *imagePtr2++;
			uchar rightBottomBlue = *imagePtr2++, rightBottomGreen = *imagePtr2++, rightBottomRed = *imagePtr2++;
			*scaledPtr++ = (leftTopBlue + rightTopBlue + leftBottomBlue + rightBottomBlue) / 4;
			*scaledPtr++ = (leftTopGreen + rightTopGreen + leftBottomGreen + rightBottomGreen) / 4;
			*scaledPtr++ = (leftTopRed + rightTopRed + leftBottomRed + rightBottomRed) / 4;
		}
	}
	return scaledImage;
}

