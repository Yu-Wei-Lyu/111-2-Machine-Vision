#include<opencv2/opencv.hpp>
#include <ppl.h>
#include<time.h> 

using namespace std;
using namespace cv;

class LabelImage { // 類別保存不同階段之影像 並不會覆蓋原圖
private:
	const string BINARY_FOLDER = "../Image/Binary/", LABELED_FOLDER = "../Image/Labeled/";
	const int MARK_BLACK = -1, MARK_WHITE = 0, EXPAND_MARK = 1;
	Mat _image, _grayImage, _binaryImage, _labelingImage;
	Mat _grayHistogram;
	int _component;
	int _maxGrayCount;
	string _name;
	vector<int> _grayCount;
	vector<int> _labelVector;
	set<int> _labelSet;
	map<int, vector<uchar>> _colorLabelMap;

public:
	// 類別初始化
	LabelImage(Mat image, string name) { // 原圖, 含副檔名之圖片名稱
		_name = name;
		_image = image;
		_grayCount = vector<int>(256);
		_grayHistogram = Mat(256, 256, CV_8UC1, 255); // 正規化灰階值方圖 (256x256x1)
		_grayImage = Mat(image.rows, image.cols, CV_8UC1);
		SetGrayScaleImage();
		_binaryImage = Mat(image.rows, image.cols, CV_8UC1);
		_labelVector = vector<int>(image.rows * (double)image.cols);
		_labelingImage = Mat(image.rows, image.cols, CV_8UC3);
		_component = 0;
		_maxGrayCount = 0;
	}

	void PrintComponentAmount() {
		cout << _name << " has " << _component << " objects" << endl;
	}

	// 設定灰階化圖像
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
		imshow(_name + " gray", _grayImage);
	}

	// 設定灰階值方圖
	void GetGrayHistogram() {
		uchar* grayPtr;
		int maxCount = 0;
		for (int row = 0; row < _grayImage.rows; row++) {
			grayPtr = _grayImage.ptr<uchar>(row);
			for (int col = 0; col < _grayImage.cols; col++) {
				_grayCount.at(*grayPtr) += 1;
				if (_grayCount.at(*grayPtr) > maxCount) maxCount = _grayCount.at(*grayPtr);
				grayPtr++;
			}
		}

		for (int col = 0; col < 256; col++) {
			int grayValue = (int)(_grayCount.at(col) / (double)maxCount * 256);
			//cout << col << ": " << grayValue << endl; // 查表用
			for (int row = 256 - grayValue; row < 256; row++)
				_grayHistogram.at<uchar>(row, col) = 0;
		}

		imshow(_name + " gray histogram", _grayHistogram);
		imwrite(BINARY_FOLDER + "gray_histogram_" + _name, _grayHistogram);
	}

	// 用門檻值設定二值化影像
	void GetBinaryImage(const int& threshold) {
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

	// 用 Opening 或 Closing 做二值化處理 (填洞與去雜訊)
	void ReconstructBinaryImage(const string restructMode, const int& restructArg1, const int& restructArg2) {
		imshow(_name + " or_binary", _binaryImage);
		imwrite(BINARY_FOLDER + "or_" + _name, _binaryImage);
		if (restructMode == "opening") {
			ErosionBinaryImage(restructArg1);
			imshow(_name + " e_", _binaryImage);
			imwrite(BINARY_FOLDER + "e_" + _name, _binaryImage);
			DilationBinaryImage(restructArg2);
			imshow(_name + " d_", _binaryImage);
			imwrite(BINARY_FOLDER + "d_" + _name, _binaryImage);
		} else if (restructMode == "closing") {
			DilationBinaryImage(restructArg1);
			imshow(_name + " d_", _binaryImage);
			imwrite(BINARY_FOLDER + "d_" + _name, _binaryImage);
			ErosionBinaryImage(restructArg2);
			imshow(_name + " e_", _binaryImage);
			imwrite(BINARY_FOLDER + "e_" + _name, _binaryImage);
		} else if (restructMode == "dilation") {
			DilationBinaryImage(restructArg1);
			imshow(_name + " d_", _binaryImage);
			imwrite(BINARY_FOLDER + "d_" + _name, _binaryImage);
		} else if (restructMode == "erosion") {
			ErosionBinaryImage(restructArg1);
			imshow(_name + " e_", _binaryImage);
			imwrite(BINARY_FOLDER + "e_" + _name, _binaryImage);
		}
		imshow(_name + " binary", _binaryImage);
		imwrite(BINARY_FOLDER + _name, _binaryImage);
	}

	int LabelVectorIndex(int i, int j) {
		return i * _image.cols + j;
	}

	void InitLabelingData() {
		uchar* binaryPtr;
		for (int row = 0; row < _binaryImage.rows; row++) {
			binaryPtr = _binaryImage.ptr<uchar>(row);
			for (int col = 0; col < _binaryImage.cols; col++) {
				_labelVector.at(LabelVectorIndex(row, col)) = (*binaryPtr++ == 0) ? MARK_BLACK : MARK_WHITE;
			}
		}
		_labelSet.clear();
	}

	void LabelPixelBy4Neighbor(const int& row, const int& col, int& labelNumber) {
		int labelTop, labelLeft;
		labelTop = (row > 0) ? _labelVector.at(LabelVectorIndex(row - 1, col)) : 0;
		labelLeft = (col > 0) ? _labelVector.at(LabelVectorIndex(row, col - 1)) : 0;
		if (labelTop == 0 && labelLeft == 0) {
			_labelVector.at(LabelVectorIndex(row, col)) = labelNumber;
			_labelSet.insert(labelNumber);
			++labelNumber;
		}
		else if (labelTop == 0 && labelLeft > 0) _labelVector.at(LabelVectorIndex(row, col)) = labelLeft;
		else if (labelTop > 0 && labelLeft == 0) _labelVector.at(LabelVectorIndex(row, col)) = labelTop;
		else {
			_labelVector.at(LabelVectorIndex(row, col)) = labelLeft;
			if (labelTop != labelLeft) {
				for (int labelIndex = 0; labelIndex < _binaryImage.rows * _binaryImage.cols; labelIndex++) {
					if (_labelVector.at(labelIndex) == labelTop) _labelVector.at(labelIndex) = labelLeft;
				}
				_labelSet.erase(labelTop);
			}
		}
	}

	// 以4連通標記物件
	void LabelingBy4() {
		InitLabelingData();
		int labelNumber = 1;
		cout << "==" << endl;
		for (int row = 0; row < _labelingImage.rows; row++) {
			for (int col = 0; col < _labelingImage.cols; col++) {
				if (_labelVector.at(LabelVectorIndex(row, col)) != 0) {
					LabelPixelBy4Neighbor(row, col, labelNumber);
				}
			}
		}
		_component = _labelSet.size();
		LabelColorImage();
	}

	bool isInVector(const vector<int>& v, int target) {
		for (const int& value : v) {
			if (value == target) {
				return true;
			}
		}
		return false;
	}


	void LabelPixelBy8Neighbor(const int& row, const int& col, int& labelNumber) {
		vector<pair<int, int>> neighbors = { {-1, -1}, {-1, 0}, {-1, 1}, {0, -1} };
		vector<int> neighborLabelSet;
		int conditionCode = -1;
		for (const pair<int, int>& neighbor : neighbors) {
			int nRow = row + neighbor.first, nCol = col + neighbor.second;
			int maskLabelNumber = 0;
			if (nRow >= 0 && nRow < _binaryImage.rows && nCol >= 0 && nCol < _binaryImage.cols - 1) {
				maskLabelNumber = _labelVector.at(LabelVectorIndex(nRow, nCol));
			}
			if (maskLabelNumber != 0 && !isInVector(neighborLabelSet, maskLabelNumber)) {
				neighborLabelSet.push_back(maskLabelNumber);
			}
		}

		if (neighborLabelSet.size() == 0) {
			_labelVector.at(LabelVectorIndex(row, col)) = labelNumber;
			_labelSet.insert(labelNumber);
			++labelNumber;
		} else if (neighborLabelSet.size() == 1) {
			_labelVector.at(LabelVectorIndex(row, col)) = neighborLabelSet.at(0);
		} else {
			int mergeLabel = neighborLabelSet.at(0);
			for (int maskIndex = neighborLabelSet.size() - 1; maskIndex > 0; maskIndex--) {
				for (int labelIndex = 0; labelIndex < _binaryImage.rows * _binaryImage.cols; labelIndex++) {
					if (find(neighborLabelSet.begin(), neighborLabelSet.end(), _labelVector.at(labelIndex)) != neighborLabelSet.end()) {
						_labelVector.at(labelIndex) = mergeLabel;
					}
				}
				neighborLabelSet.pop_back();
			}

		}
	}

	// 以8連通標記物件
	void LabelingBy8() {
		InitLabelingData();
		int labelNumber = 1;
		cout << "==" << endl;
		for (int row = 0; row < _labelingImage.rows; row++) {
			for (int col = 0; col < _labelingImage.cols; col++) {
				if (_labelVector.at(LabelVectorIndex(row, col)) != 0) {
					LabelPixelBy8Neighbor(row, col, labelNumber);
				}
			}
		}
		_component = _labelSet.size();
		LabelColorImage();
	}

	// 對二值化圖像膨脹 iteration 次
	void DilationBinaryImage(int iteration) {
		uchar* binaryPtr;
		for (int repeat = 0; repeat < iteration; ++repeat) {
			InitLabelingData();
			for (int row = 0; row < _binaryImage.rows; ++row) {
				binaryPtr = _binaryImage.ptr<uchar>(row);
				for (int col = 0; col < _binaryImage.cols; ++col) {
					if (_labelVector.at(LabelVectorIndex(row, col)) == MARK_BLACK) {
						ConvolutionReconstruct(row, col, 0);
					}
				}
			}
		}
	}

	// 對二值化圖像侵蝕 iteration 次
	void ErosionBinaryImage(int iteration) {
		uchar* binaryPtr;
		for (int repeat = 0; repeat < iteration; ++repeat) {
			InitLabelingData();
			for (int row = 0; row < _binaryImage.rows; ++row) {
				binaryPtr = _binaryImage.ptr<uchar>(row);
				for (int col = 0; col < _binaryImage.cols; ++col) {
					if (_labelVector.at(LabelVectorIndex(row, col)) == MARK_WHITE) {
						ConvolutionReconstruct(row, col, 255);
					}
				}
			}
		}
	}

	// 用3x3卷積判斷是否修改鄰近的像素點
	void ConvolutionReconstruct(int row, int col, int value) {
		int labelMark = (value == 0) ? MARK_BLACK : MARK_WHITE;
		for (int neighborRow = -1; neighborRow <= 1; ++neighborRow) {
			for (int neighborCol = -1; neighborCol <= 1; ++neighborCol) {
				int nRow = row + neighborRow, nCol = col + neighborCol;
				if (nRow >= 0 && nRow < _binaryImage.rows && nCol >= 0 && nCol < _binaryImage.cols) {
					if (_labelVector.at(LabelVectorIndex(nRow, nCol)) != labelMark) {
						_binaryImage.at<uchar>(nRow, nCol) = value;
						_labelVector.at(LabelVectorIndex(nRow, nCol)) == EXPAND_MARK;
					}
				}
			}
		}
	}

	void LabelColorImage() {
		set<vector<uchar>> colorSet;
		int colorCount = 0;
		int max = 255, min = 0;
		set<int> labelSet = _labelSet;
		srand(time(0));
		for (int labelNumber : _labelSet) {
			bool isInColorSet = false;
			vector<uchar> bufferColor;
			do {
				bufferColor.clear();
				for (int color = 0; color < 3; color++) {
					bufferColor.push_back(rand() % (max - min + 1) + min);
				}
				isInColorSet = colorSet.find(bufferColor) != colorSet.end();
			} while (isInColorSet);
			colorSet.insert(bufferColor);
			_colorLabelMap[labelNumber] = bufferColor;

		}
		for (int row = 0; row < _labelingImage.rows; row++) {
			for (int col = 0; col < _labelingImage.cols; col++) {
				int labelCode = _labelVector.at(LabelVectorIndex(row, col));
				if (labelCode == 0) {
					_labelingImage.at<Vec3b>(row, col) = Vec3b(0, 0, 0);
				}
				else {
					vector<uchar> colorDecode = _colorLabelMap[labelCode];
					_labelingImage.at<Vec3b>(row, col) = Vec3b(colorDecode.at(0), colorDecode.at(1), colorDecode.at(2));
				}
			}
		}
		imshow(_name + " labeled", _labelingImage);
		imwrite(LABELED_FOLDER + _name, _labelingImage);
	}
};

class ImageInfo {
public:
	string Name;
	string RestructMode;
	int BinaryThreshold;
	int RestructArg1;
	int RestructArg2;
	ImageInfo(string name, int binaryThreshold, string restructMode, int restructArg1 = 0, int restructArg2 = 0) {
		this->Name = name;
		this->BinaryThreshold = binaryThreshold;
		this->RestructMode = restructMode;
		this->RestructArg1 = restructArg1;
		this->RestructArg2 = restructArg2;
	}
};

bool isInVector(const vector<int>& v, int target) {
	for (const int& value : v) {
		if (value == target) {
			return true;
		}
	}
	return false;
}

int main() {
	cout << "[Main] Start to processing images, please wait..." << endl;
	vector<int> values = { 0,1,0,1 };
	vector<int> maskLabels;

	for (auto value : values) {
		if (value != 0 && !isInVector(maskLabels, value)) {
			maskLabels.push_back(value);
		}
	}

	for (auto value : maskLabels) {
		cout << value << ", ";
	}
	cout << endl;
	cout << maskLabels.size();
	return 0;
	vector<string> folderList = { "../Image/Source/", "../Image/Binary/" };
	vector<ImageInfo> imageInfoList;
	imageInfoList.push_back(ImageInfo("1.png", 119, "erosion", 1));
	imageInfoList.push_back(ImageInfo("2.png", 221, "dilation", 1));
	imageInfoList.push_back(ImageInfo("3.png", 85, "closing", 4, 5));
	imageInfoList.push_back(ImageInfo("4.png", 227, "orginal"));
	int debug = 3;
	for (int i = debug; i <= debug; ++i) {
	//for (int i = 0; i < imageInfoList.size(); i++) {
		ImageInfo imageInfo = imageInfoList.at(i);
		Mat image = imread("../Image/Source/" + imageInfo.Name);
		LabelImage labelImage = LabelImage(image, imageInfo.Name);
		labelImage.GetBinaryImage(imageInfo.BinaryThreshold);
		labelImage.ReconstructBinaryImage(imageInfo.RestructMode, imageInfo.RestructArg1, imageInfo.RestructArg2);
		labelImage.LabelingBy4();
		labelImage.PrintComponentAmount();
	}
	cout << "[Main] All image processing complete." << endl;
	waitKey();
	destroyAllWindows();
	return 0;
}