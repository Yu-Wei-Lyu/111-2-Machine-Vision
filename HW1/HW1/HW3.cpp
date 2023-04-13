#include<opencv2/opencv.hpp>
#include <ppl.h>
#include<time.h> 

#define COLOR_BLACK 0
#define COLOR_GRAY 128
#define COLOR_WHITE 255

using namespace std;
using namespace cv;

class QuadTreeImage {
private:
	const string BINARY_FOLDER = "../Image/Binary/";
	Mat _image, _grayImage, _binaryImage, _resultImage;
	string _name;
	int _layer;
	int _splitAreaRows;
	int _splitAreaCols;

public:
	// 類別初始化
	QuadTreeImage(Mat image, string name) {
		_name = name;
		_image = image;
		Initialize();
	}

	// 類別初始化
	void Initialize() {
		int height = _image.rows, width = _image.cols;
		_grayImage = Mat(height, width, CV_8UC1);
		_binaryImage = Mat(height, width, CV_8UC1);
		_resultImage = Mat(height, width, CV_8UC1, 255);
	}

	// 設定灰階化圖像
	void SetGrayScaleImage() {
		const uchar* imagePtr;
		uchar* gray;
		for (int row = 0; row < _image.rows; ++row) {
			imagePtr = _image.ptr<uchar>(row);
			gray = _grayImage.ptr<uchar>(row);
			for (int col = 0; col < _image.cols; ++col) {
				uchar blue = *imagePtr++, green = *imagePtr++, red = *imagePtr++;
				*gray++ = (0.3 * red) + (0.59 * green) + (0.11 * blue);
			}
		}
		//imshow(_name + " gray", _grayImage);
	}

	// 依門檻值設定二值化影像
	void GetBinaryImage(const int& threshold) {
		SetGrayScaleImage();
		const uchar* imagePtr;
		uchar* binary;
		for (int row = 0; row < _grayImage.rows; ++row) {
			imagePtr = _grayImage.ptr<uchar>(row);
			binary = _binaryImage.ptr<uchar>(row);
			for (int col = 0; col < _grayImage.cols; ++col) {
				*binary++ = (*imagePtr++ >= threshold) ? 255 : 0;
			}
		}
		imshow(_name + "original binary", _binaryImage);
		imwrite(BINARY_FOLDER + _name, _binaryImage);
	}

	void GetQuadTreeLayerBy(int value) {
		re(value, Point2i(0, 0), Point2i(_image.rows, _image.cols));
		imshow("result", _resultImage);
	}

	// 判斷2x2像素是否同色
	uchar GetMergeColor(const vector<uchar> pixels) {
		int average = 0;
		for (const uchar& pixel : pixels) {
			average += pixel;
		}
		average /= pixels.size();
		return (average == COLOR_BLACK || average == COLOR_WHITE) ? average : COLOR_GRAY;
	}

	void re(int count, Point2i pointBegin, Point2i pointEnd) {
		vector<uchar> pixels;
		const uchar* binaryPtr;
		for (int x = pointBegin.x; x < pointEnd.x; x++) {
			binaryPtr = _binaryImage.ptr<uchar>(x);
			for (int y = pointBegin.y; y < pointEnd.y; y++) {
				pixels.push_back(*binaryPtr++);
			}
		}
		uchar mergedColor = this->GetMergeColor(pixels);
		if (mergedColor == COLOR_BLACK || mergedColor == COLOR_WHITE) {
			uchar* resultPtr;
			for (int x = pointBegin.x; x < pointEnd.x; x++) {
				resultPtr = _resultImage.ptr<uchar>(x);
				for (int y = pointBegin.y; y < pointEnd.y; y++) {
					*resultPtr++ = mergedColor;
				}
			}
		}
		if (mergedColor == COLOR_GRAY && count != 0) {
			int midX = (pointBegin.x + pointEnd.x) / 2, midY = (pointBegin.y + pointEnd.y) / 2;
			re(count - 1, Point2i(pointBegin.x, pointBegin.y), Point2i(midX, midY));
			re(count - 1, Point2i(midX, pointBegin.y), Point2i(pointEnd.x, pointEnd.y));
			re(count - 1, Point2i(pointBegin.x, midY), Point2i(midX, midY));
			re(count - 1, Point2i(midX, midY), Point2i(pointEnd.x, pointEnd.y));
		}
	}

	// 一維至二維印射處理 _labelVector 所用
	int LabelVectorIndex(int i, int j) {
		return i * _image.cols + j;
	}
};

class ImageInfo {
public:
	string Name;
	int BinaryThreshold;

	ImageInfo(string name, int binaryThreshold) {
		this->Name = name;
		this->BinaryThreshold = binaryThreshold;
	}
};

int main() {
	cout << "[Main] Start to processing images, please wait..." << endl;

	// 設定各圖像處理參數
	//vector<ImageInfo> imageInfoList{ 
	//	ImageInfo("1.png", 135), 
	//	ImageInfo("2.png", 245), 
	//	ImageInfo("3.png", 155),
	//	ImageInfo("4.png", 254) 
	//};

	vector<ImageInfo> imageInfoList{ ImageInfo("1.png", 135) };
	// 執行各圖像處理
	for (ImageInfo& imageInfo : imageInfoList) {
		Mat image = imread("../Image/Source/" + imageInfo.Name);
		QuadTreeImage quadTreeImage = QuadTreeImage(image, imageInfo.Name);
		//quadTreeImage.GetBinaryImage(imageInfo.BinaryThreshold);
		quadTreeImage.GetQuadTreeLayerBy(1);
	}
	cout << "[Main] All image processing complete." << endl;
	cv::waitKey();
	cv::destroyAllWindows();
	return 0;
}