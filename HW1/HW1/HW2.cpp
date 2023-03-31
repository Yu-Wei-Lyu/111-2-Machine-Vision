#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class LabelImage { // ���O�O�s���P���q���v�� �ä��|�л\���
public:
	// ���O��l��
	LabelImage(Mat image, string name) { // ���, �t���ɦW���Ϥ��W��
		_name = name;
		_image = image;
		_grayCount = vector<int>(256);
		_grayHistogram = Mat(256, 256, CV_8UC1, 255); // ���W�ƦǶ��Ȥ�� (256x256x1)
		_grayImage = Mat(image.rows, image.cols, CV_8UC1);
		SetGrayScaleImage();
		_binaryImage = Mat(image.rows, image.cols, CV_8UC1);
		_labelVector = vector<int>(image.rows * image.cols);
		_labelingImage = Mat(image.rows, image.cols, CV_8UC1);
		_component = 0;
		_maxGrayCount = 0;
	}

	// �]�w�Ƕ��ƹϹ�
	void SetGrayScaleImage() {
		const uchar* imagePtr;
		uchar* gray;
		for (int row = 0; row < _image.rows; row++) {
			imagePtr = _image.ptr<uchar>(row);
			gray = _grayImage.ptr<uchar>(row);
			for (int col = 0; col < _image.cols; col++) {
				uchar blue = *imagePtr++, green = *imagePtr++, red = *imagePtr++;
				*gray++ = (0.3 * red) + (0.59 * green) + (0.11 * blue);
			}
		}
	}

	// �]�w�Ƕ��Ȥ��
	void SetGrayHistogram() {
		uchar* grayPtr;
		int maxCount = 0;
		for (int row = 0; row < _grayImage.rows; row++) {
			grayPtr = _grayImage.ptr<uchar>(row);
			for (int col = 0; col < _grayImage.cols; col++) {
				_grayCount.at(*grayPtr) += 1;
				if (_grayCount.at(*grayPtr) > maxCount) maxCount = _grayCount.at(*grayPtr);
				*grayPtr++;
			}
		}

		for (int col = 0; col < 256; col++) {
			int grayValue = (int)(_grayCount.at(col) / (double)maxCount * 256);
			//cout << col << ": " << grayValue << endl; // �d���
			for (int row = 256 - grayValue; row < 256; row++)
				_grayHistogram.at<uchar>(row, col) = 0;
		}
	}

	// �Ϊ��e�ȳ]�w�G�ȤƼv��
	void SetBinaryImage(int threshold = 128) {
		const uchar* imagePtr;
		uchar* binary;
		for (int row = 0; row < _grayImage.rows; row++) {
			imagePtr = _grayImage.ptr<uchar>(row);
			binary = _binaryImage.ptr<uchar>(row);
			for (int col = 0; col < _grayImage.cols; col++) {
				*binary++ = (*imagePtr++ >= threshold) ? 255 : 0;
			}
		}
	}

	int LabelVectorIndex(int i, int j) {
		return i * _image.cols + j;
	}

	void InitLabelingVector() {
		uchar* binaryPtr;
		for (int row = 0; row < _binaryImage.rows; row++) {
			binaryPtr = _binaryImage.ptr<uchar>(row);
			for (int col = 0; col < _binaryImage.cols; col++) {
				_labelVector.at(LabelVectorIndex(row, col)) = (*binaryPtr++ == 0) ? -1 : 0;
				if (_labelVector.at(LabelVectorIndex(row, col)) == -1) cout << "-";
				else cout << 0;
			}
			cout << endl;
		}
	}



	// label �íp�⪫���
	void LabelingBy4() {
		InitLabelingVector();
		uchar* labelPtr;
		uchar pixelTop, pixelLeft;
		int labelNumber = 0;
		cout << "==" << endl;
		for (int row = 0; row < _labelingImage.rows; row++) {
			for (int col = 0; col < _labelingImage.cols; col++) {
				if (_labelVector.at(LabelVectorIndex(row, col)) != 0) {
					pixelTop = (row > 0) ? _labelVector.at(LabelVectorIndex(row - 1, col)) : 0;
					pixelLeft = (col > 0) ? _labelVector.at(LabelVectorIndex(row, col - 1)) : 0;
					if (pixelTop <= 0 && pixelLeft <= 0) _labelVector.at(LabelVectorIndex(row, col)) = ++labelNumber;
					else if (pixelTop > 0 && pixelLeft <= 0) _labelVector.at(LabelVectorIndex(row, col)) = pixelTop;
					else if (pixelTop <= 0 && pixelLeft > 0) _labelVector.at(LabelVectorIndex(row, col)) = pixelLeft;
					else {
						//if (pixelTop == pixelLeft) 
						_labelVector.at(LabelVectorIndex(row, col)) = pixelTop;
						
					}
					//cout << _labelVector.at(LabelVectorIndex(row, col)) << ":" << pixelTop << ":" << pixelLeft << endl;
				}
				cout << _labelVector.at(LabelVectorIndex(row, col));
			}
			cout << endl;
		}
	}

	// ��ܤG�ȤƹϹ�
	void ShowBinaryImage() {
		imshow(_name + "�G�ȹ�", _binaryImage);
	}

	// �x�s�G�ȤƹϹ�
	void SaveBinaryImage() {
		imwrite(BINARY_FOLDER + _name, _binaryImage);
	}

	// ��ܦǶ��Ȥ��
	void ShowGrayHistogram() {
		//imshow(HISTOGRAM_PREFIX + _name, _grayHistogram);
		imshow(_name + " �Ƕ��Ȥ��", _grayHistogram);
	}

	// �x�s�Ƕ��Ȥ��
	void SaveGrayHistogram() {
		imwrite(BINARY_FOLDER + HISTOGRAM_PREFIX + _name, _grayHistogram);
	}
private:
	const string BINARY_FOLDER = "../Image/Binary/";
	const string HISTOGRAM_PREFIX = "gray_histogram_";
	Mat _image;
	Mat _grayImage;
	Mat _binaryImage;
	Mat _grayHistogram;
	Mat _labelingImage;
	int _component;
	int _maxGrayCount;
	string _name;
	vector<int> _grayCount;
	vector<int> _labelVector;
};

int main() {
	cout << "[Main] Start to processing images, please wait..." << endl;
	vector<string> folderList = { "../Image/Source/", "../Image/Binary/" };
	//vector<int> binaryThresholdList = { 119, 245, 85, 254 };
	//vector<string> imageList = { "1.png", "2.png", "3.png", "4.png"};
	vector<int> binaryThresholdList = { 254 };
	vector<string> imageList = { "test2.png" };
	for (int i = 0; i < imageList.size(); i++) {
		string imageName = imageList.at(i);
		Mat image = imread(folderList.at(0) + imageName);
		LabelImage labelImage = LabelImage(image, imageName);
		labelImage.SetBinaryImage(binaryThresholdList.at(i));
		labelImage.LabelingBy4();
		//labelImage.SetGrayHistogram();
		//labelImage.ShowGrayHistogram();
		//labelImage.SaveGrayHistogram();
		labelImage.ShowBinaryImage();
		//labelImage.SaveBinaryImage();
	}
	cout << "[Main] All image processing complete." << endl;
	waitKey();
	destroyAllWindows();
	return 0;
}