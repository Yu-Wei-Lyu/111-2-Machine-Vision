#include<opencv2/opencv.hpp>
#include <ppl.h>
#include<time.h> 

using namespace std;
using namespace cv;

class QuadTreeImage {
private:
	const string BINARY_FOLDER = "../Image/Binary/";
	Mat _image, _grayImage, _binaryImage;
	string _name;

public:
	// ���O��l��
	QuadTreeImage(Mat image, string name) {
		_name = name;
		_image = image;
		Initialize();
	}

	// ���O��l��
	void Initialize() {
		int height = _image.rows, width = _image.cols;
		_grayImage = Mat(height, width, CV_8UC1);
		_binaryImage = Mat(height, width, CV_8UC1);
	}

	// �]�w�Ƕ��ƹϹ�
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

	// �̪��e�ȳ]�w�G�ȤƼv��
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

	// �P�_ value �O�_�b vector ��
	bool isInVector(const vector<int>& v, int value) {
		for (const int& e : v) {
			if (e == value) return true;
		}
		return false;
	}

	// �@���ܤG���L�g�B�z _labelVector �ҥ�
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

	// �]�w�U�Ϲ��B�z�Ѽ�
	vector<ImageInfo> imageInfoList{ 
		ImageInfo("1.png", 135), 
		ImageInfo("2.png", 245), 
		ImageInfo("3.png", 155),
		ImageInfo("4.png", 254) 
	};

	// ����U�Ϲ��B�z
	for (ImageInfo& imageInfo : imageInfoList) {
		Mat image = imread("../Image/Source/" + imageInfo.Name);
		QuadTreeImage quadTreeImage = QuadTreeImage(image, imageInfo.Name);
		quadTreeImage.GetBinaryImage(imageInfo.BinaryThreshold);
	}
	cout << "[Main] All image processing complete." << endl;
	cv::waitKey();
	cv::destroyAllWindows();
	return 0;
}