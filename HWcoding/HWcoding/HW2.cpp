#include<opencv2/opencv.hpp>
#include <ppl.h>
#include<time.h> 

using namespace std;
using namespace cv;

class LabelImage {
private:
	const string BINARY_FOLDER = "../Image/Binary/", LABELED_FOLDER = "../Image/Labeled/";
	const int MARK_BLACK = -1, MARK_WHITE = 0, EXPAND_MARK = 1;
	Mat _image, _grayImage, _binaryImage, _labelImage;
	int _component;
	string _name;
	vector<int> _labelVector;
	set<int> _labelSet;
	map<int, vector<uchar>> _labelColorMap;
	void (LabelImage::*_labelPixelFunc)(const int&, const int&, int&) = nullptr;

public:
	// 類別初始化
	LabelImage(Mat image, string name) {
		_name = name;
		_image = image;
		_component = 0;
		Initialize();
	}

	// 類別初始化
	void Initialize() {
		int height = _image.rows, width = _image.cols;
		_grayImage = Mat(height, width, CV_8UC1);
		_binaryImage = Mat(height, width, CV_8UC1);
		_labelVector = vector<int>(static_cast<int>(_image.total()));
		_labelImage = Mat(height, width, CV_8UC3);
	}

	// 初始化 labeling 會用到的資料
	void InitLabelingData() {
		uchar* binaryPtr;
		for (int row = 0; row < _binaryImage.rows; ++row) {
			binaryPtr = _binaryImage.ptr<uchar>(row);
			for (int col = 0; col < _binaryImage.cols; ++col) {
				_labelVector.at(LabelVectorIndex(row, col)) = (*binaryPtr++ == 0) ? MARK_BLACK : MARK_WHITE;
			}
		}
		_labelSet.clear();
		_labelImage = Mat(_image.rows, _image.cols, CV_8UC3);
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
		//imshow(_name + "original binary", _binaryImage);
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

	// 二值化填洞與去雜訊處理的四種方法
	void ReconstructBinaryImage(const string& restructMode, const int& restructArg1, const int& restructArg2) {
		if (restructMode == "opening") {
			ErosionBinaryImage(restructArg1);
			DilationBinaryImage(restructArg2);
		}
		else if (restructMode == "closing") {
			DilationBinaryImage(restructArg1);
			ErosionBinaryImage(restructArg2);
		}
		else if (restructMode == "dilation") {
			DilationBinaryImage(restructArg1);
		}
		else if (restructMode == "erosion") {
			ErosionBinaryImage(restructArg1);
		}
		imshow(_name + " binary", _binaryImage);
		imwrite(BINARY_FOLDER + _name, _binaryImage);
	}

	// 依 neighborRule 輸出 label 圖像、數量
	void LabelingByNeighbor(int neighborRule) {
		if (neighborRule == 4) _labelPixelFunc = &LabelImage::LabelPixelBy4Neighbor;
		else if (neighborRule == 8) _labelPixelFunc = &LabelImage::LabelPixelBy8Neighbor;
		else {
			cout << "Only support 4 or 8 neighbor." << endl;
			return;
		}
		InitLabelingData();
		int labelNumber = 1;
		for (int row = 0; row < _labelImage.rows; ++row) {
			for (int col = 0; col < _labelImage.cols; ++col) {
				if (_labelVector.at(LabelVectorIndex(row, col)) != 0) {
					(this->*_labelPixelFunc)(row, col, labelNumber);
				}
			}
		}
		_component = _labelSet.size();
		DrawLabel();
		cout << _name << " with " << neighborRule << "-neighbor has " << _component << " objects" << endl;
		imshow(_name + " " + to_string(neighborRule) + "-neighbor labeled", _labelImage);
		imwrite(LABELED_FOLDER + to_string(neighborRule) + "-neighbor_" + _name, _labelImage);
	}

	// 依 4 連通規則設定label相關資料
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
				for (int labelIndex = 0; labelIndex < _binaryImage.rows * _binaryImage.cols; ++labelIndex) {
					if (_labelVector.at(labelIndex) == labelTop) _labelVector.at(labelIndex) = labelLeft;
				}
				_labelSet.erase(labelTop);
			}
		}
	}

	// 依 8 連通規則設定label相關資料
	void LabelPixelBy8Neighbor(const int& row, const int& col, int& labelNumber) {
		vector<int> neighborLabelSet = GetNeighborLabelBy8(row, col);
		if (neighborLabelSet.size() == 0) {
			_labelVector.at(LabelVectorIndex(row, col)) = labelNumber;
			_labelSet.insert(labelNumber);
			labelNumber++;
		}
		else {
			_labelVector.at(LabelVectorIndex(row, col)) = neighborLabelSet.at(0);
			int mergeLabel = neighborLabelSet.at(0);
			for (int maskIndex = 1; maskIndex < neighborLabelSet.size(); ++maskIndex) {
				int combineLabel = neighborLabelSet.at(maskIndex);
				for (int labelIndex = 0; labelIndex < _binaryImage.rows * _binaryImage.cols; ++labelIndex) {
					if (_labelVector.at(labelIndex) == combineLabel) {
						_labelVector.at(labelIndex) = mergeLabel;
					}
				}
				_labelSet.erase(combineLabel);
			}
		}
	}

	// 依目前像素點回傳8連通判斷需要的資料 (左上、上、右上、左)
	vector<int> GetNeighborLabelBy8(const int& row, const int& col) {
		vector<pair<int, int>> masks = { {-1, -1}, {-1, 0}, {-1, 1}, {0, -1} };
		vector<int> neighborLabelSet;
		int conditionCode = -1;
		for (const pair<int, int>& mask : masks) {
			int nRow = row + mask.first, nCol = col + mask.second;
			int maskLabelNumber = 0;
			if (nRow >= 0 && nRow < _binaryImage.rows && nCol >= 0 && nCol < _binaryImage.cols - 1) {
				maskLabelNumber = _labelVector.at(LabelVectorIndex(nRow, nCol));
			}
			if (maskLabelNumber != 0 && !isInVector(neighborLabelSet, maskLabelNumber)) {
				neighborLabelSet.push_back(maskLabelNumber);
			}
		}
		return neighborLabelSet;
	}

	// 判斷 value 是否在 vector 中
	bool isInVector(const vector<int>& v, int value) {
		for (const int& e : v) {
			if (e == value) return true;
		}
		return false;
	}

	// 依照 _labelSet 做出 label-color map
	void SetColorLabelMap() {
		set<vector<uchar>> colorSet;
		int max = 255, min = 0;
		srand(time(0));
		for (int labelNumber : _labelSet) {
			bool isInColorSet = false;
			vector<uchar> bufferColor;
			do {
				bufferColor.clear();
				for (int color = 0; color < 3; ++color) {
					bufferColor.push_back(rand() % (max - min + 1) + min);
				}
				isInColorSet = colorSet.find(bufferColor) != colorSet.end();
			} while (isInColorSet);
			colorSet.insert(bufferColor);
			_labelColorMap[labelNumber] = bufferColor;
		}
	}

	// 依照 label-color map 將輸出圖著色
	void DrawLabel() {
		SetColorLabelMap();
		uchar* labelImagePtr;
		for (int row = 0; row < _labelImage.rows; ++row) {
			labelImagePtr = _labelImage.ptr<uchar>(row);
			for (int col = 0; col < _labelImage.cols; ++col) {
				int labelCode = _labelVector.at(LabelVectorIndex(row, col));
				vector<uchar> colorDecode = _labelColorMap[labelCode];
				for (int bgr = 0; bgr < 3; ++bgr) {
					*labelImagePtr++ = (labelCode != 0) ? colorDecode.at(bgr) : 0;
				}
			}
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

int main() {
	cout << "[Main] Start to processing images, please wait..." << endl;

	// 設定各圖像處理參數
	vector<ImageInfo> imageInfoList{ 
		ImageInfo("1.png", 119, "opening", 5, 4), 
		ImageInfo("2.png", 221, "dilation", 1), 
		ImageInfo("3.png", 85, "closing", 4, 5),
		ImageInfo("4.png", 228, "orginal") 
	};

	// 執行各圖像處理
	for (int i = 0; i < imageInfoList.size(); ++i) {
		ImageInfo imageInfo = imageInfoList.at(i);
		Mat image = imread("../Image/Source/" + imageInfo.Name);
		LabelImage labelImage = LabelImage(image, imageInfo.Name);
		labelImage.GetBinaryImage(imageInfo.BinaryThreshold);
		labelImage.ReconstructBinaryImage(imageInfo.RestructMode, imageInfo.RestructArg1, imageInfo.RestructArg2);
		labelImage.LabelingByNeighbor(4);
		labelImage.LabelingByNeighbor(8);
	}
	cout << "[Main] All image processing complete." << endl;
	cv::waitKey();
	cv::destroyAllWindows();
	return 0;
}