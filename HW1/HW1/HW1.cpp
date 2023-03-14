#include<opencv2/opencv.hpp>
#include<direct.h>

using namespace std;
using namespace cv;

void prepareFolder(vector<string> folderList);
Mat getGrayScaleImage(const Mat& image);
Mat getBinaryImage(const Mat& image);

Vec3b getColorFromMap(Mat& colorMap, const Vec3b bgr, int& threshould, int& colorPixels);
Mat getIndexColorImage(const Mat& image, Mat& colorMap, int threshould);

int main()
{
	vector<string> folderList = { "./Image/Source/", "./Image/Grayscale/", "./Image/Binary/", "./Image/Index color/", "./Image/Resize/", "./Image/Resize(interpolation)/" };
	//vector<string> imageList = { "House256.png", "House512.png", "JellyBeans.png", "Lena.png", "Mandrill.png", "Peppers.png" };
	vector<string> imageList = { "House512.png" };
	prepareFolder(folderList);

	for (string& imageName : imageList) {
		Mat image = imread(folderList.at(0) + imageName);
		imshow("Source", image);
		/*Mat gray = getGrayScaleImage(image);
		imshow("Gray", gray);
		imwrite(folderList.at(1) + imageName, gray);
		Mat binary = getBinaryImage(gray);
		imshow("Binary", binary);
		imwrite(folderList.at(2) + imageName, binary);*/
		Mat myColorMap;
		Mat indexColor2 = getIndexColorImage(image, myColorMap, 20);
		imshow("Index-Color-Limit", indexColor2);
		imshow("Color-Map-Limit", myColorMap);
		imwrite(folderList.at(3) + imageName, indexColor2);
		imwrite(folderList.at(3) + "color_map_" + imageName, myColorMap);
	}
	waitKey();
	destroyAllWindows();
	return 0;
}

void prepareFolder(vector<string> folderList) {
	for (string& folder : folderList) {
		if (_mkdir(folder.c_str()) != 0) {
			cerr << strerror(errno) << ", file path " << folder.c_str() << endl;
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

Vec3b getColorFromMap(Mat& colorMap, const Vec3b bgr, int& threshould, int& colorPixels) {
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
			indexColorImage.at<Vec3b>(row, col) = getColorFromMap(colorMap, originPixel, threshold, colorMapSize);
			if (colorMapSize > 256) {
				cout << "Threshold too low, please set higher threshold." << endl;
				return indexColorImage;
			}
		}
	}
	return indexColorImage;
}

/*
Mat getMyColorMap();
Mat getIndexColorImage(const Mat& image);
Mat getMyColorMap() {
	Mat myColorMap = Mat(16, 16, CV_8UC3);
	uchar* p = myColorMap.ptr<uchar>(0);
	for (int i = 0; i < 256; i += 51) {
		for (int j = 0; j < 256; j += 51) {
			for (int k = 0; k < 256; k += 51) {
				*p++ = i, * p++ = j, * p++ = k;
			}
		}
	}
	imshow("./Image/myColorMap.png", myColorMap);
	imwrite("./Image/myColorMap.png", myColorMap);
	return myColorMap;
}
Mat getIndexColorImage(const Mat& image) {
	Mat indexColorImage = Mat(image.rows, image.cols, CV_8UC3);
	int nCols = image.channels() * image.cols;
	for (int row = 0; row < image.rows; row++) {
		const uchar* imagePtr = image.ptr<uchar>(row);
		uchar* indexColorPtr = indexColorImage.ptr<uchar>(row);
		for (int col = 0; col < nCols; col++) {
			uchar originValue = *imagePtr++;
			int remain = originValue % 51;
			*indexColorPtr = originValue - remain;
			if (remain >= 25)
				(*indexColorPtr) += 51;
			indexColorPtr++;
		}
	}
	return indexColorImage;
}*/