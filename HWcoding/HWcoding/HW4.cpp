#include<opencv2/opencv.hpp>
#include<time.h> 
#include<cmath>

#define PI 3.14159

using namespace std;
using namespace cv;

class FilterImage {
private:
	const string IMAGE_FOLDER = "../Image/";
	vector<double> _gaussianKernel;
	uchar (FilterImage::* _getValueFunc)(vector<uchar>) = nullptr;

public:
	string FileName, FileExt;
	// 類別初始化
	FilterImage(Mat image, string file) {
		size_t dot_pos = file.rfind('.');
		FileName = file.substr(0, dot_pos);
		FileExt = file.substr(dot_pos);
	}

	// 用kernelSize*kernelSize卷積回傳鄰近的像素點
	vector<uchar> getConvolutionList(const Mat& source, int row, int col, int kernelSize) {
		vector<uchar> pixelList(kernelSize * kernelSize);
		int pixelListIndex = 0;
		int start = -kernelSize / 2, end = start + kernelSize - 1; // if kernelSize = 3*3, start = -1, end = 1 etc...
		for (int windowRow = start; windowRow <= end; ++windowRow) {
			for (int windowCol = start; windowCol <= end; ++windowCol) {
				int nRow = row + windowRow, nCol = col + windowCol;
				if (nRow >= 0 && nRow < source.rows && nCol >= 0 && nCol < source.cols) {
					pixelList.at(pixelListIndex++) = source.at<uchar>(nRow, nCol);
				} else {
					pixelList.at(pixelListIndex++) = 0;
				}
			}
		}
		//int index = 0;
		//for (int i = 0; i < kernelSize; i++) {
		//	for (int j = 0; j < kernelSize; j++) {
		//		cout << (int)pixelList.at(index++) << "\t";
		//	}
		//	cout << endl;
		//}
		return pixelList;
	}

	// 取得卷積範圍內的 Mean 值
	uchar getMeanValue(vector<uchar> pixelList) {
		int value = 0;
		int pixelAmount = pixelList.size();
		for (int pixelIndex = 0; pixelIndex < pixelAmount; ++pixelIndex) {
			value += (int)pixelList.at(pixelIndex);
		}
		return round((double)value / pixelAmount);
	}

	// 快速排序函数
	void quickSort(vector<uchar>& nums, int left, int right) {
		if (left >= right) return;
		int i = left, j = right, pivot = nums[left + (right - left) / 2];
		while (i <= j) {
			while (nums[i] < pivot) i++;
			while (nums[j] > pivot) j--;
			if (i <= j) {
				swap(nums[i], nums[j]);
				i++;
				j--;
			}
		}
		quickSort(nums, left, j);
		quickSort(nums, i, right);
	}

	// 取得卷積範圍內排序後的 Median 值
	uchar getMedianValue(vector<uchar> pixelList) {
		quickSort(pixelList, 0, pixelList.size() - 1);
		int index = 0;
		int medianIndex = (pixelList.size() - 1) / 2;
		return pixelList.at(medianIndex);
	}

	// 依照給定的 kernel 大小和標準化參數給出 kernel權重值
	void setGaussianKernel(int kernelSize, double standardDeviation) {
		_gaussianKernel = vector<double>(kernelSize * kernelSize);
		double weightTotal = 0;
		double kernelIndex = 0;
		int start = -kernelSize / 2, end = start + kernelSize - 1; // if kernelSize = 3*3, start = -1, end = 1 etc...
		for (int windowRow = start; windowRow <= end; ++windowRow) {
			for (int windowCol = start; windowCol <= end; ++windowCol) {
				double rowSquare = pow(windowRow, 2), colSquare = pow(windowCol, 2);
				double exponent = -(rowSquare + colSquare) / (2 * pow(standardDeviation, 2));
				double weight = exp(exponent);
				//cout << weight  << "\t";
				_gaussianKernel.at(kernelIndex++) = weight;
				weightTotal += weight;
			}
			//cout << endl;
		}
		//cout << "total = " << kernelIndex << endl;
		//double newWeightTotal = 0;
		for (double& weight : _gaussianKernel) {
			weight /= weightTotal;
			//newWeightTotal += weight;
		}
		//kernelIndex = 0;
		//for (int windowRow = start; windowRow <= end; ++windowRow) {
		//	for (int windowCol = start; windowCol <= end; ++windowCol) {
		//		cout << _gaussianKernel.at(kernelIndex++) << "\t";
		//	}
		//	cout << endl;
		//}
		//cout << "new total = " << newWeightTotal << endl;
	}

	// 取得卷積範圍內的 Gaussian 值
	uchar getGaussianValue(vector<uchar> pixelList) {
		double value = 0;
		for (int pixelIndex = 0; pixelIndex < pixelList.size(); ++pixelIndex) {
			value += (double)pixelList.at(pixelIndex) * (double)_gaussianKernel.at(pixelIndex);
		}
		return (uchar)round(value);
	}

	// 設定灰階化圖像
	void updateGrayScaleImage(Mat& source) {
		Mat grayImage = Mat(source.rows, source.cols, CV_8UC1);
		const uchar* imagePtr;
		uchar* gray;
		for (int row = 0; row < source.rows; ++row) {
			imagePtr = source.ptr<uchar>(row);
			gray = grayImage.ptr<uchar>(row);
			for (int col = 0; col < source.cols; ++col) {
				uchar blue = *imagePtr++, green = *imagePtr++, red = *imagePtr++;
				*gray++ = (0.3 * red) + (0.59 * green) + (0.11 * blue);
			}
		}
		grayImage.copyTo(source);
	}

	void getFilterImageByMode(Mat& source, Mat& dest, const string& mode, int kernelSize) {
		if (mode == "mean") _getValueFunc = &FilterImage::getMeanValue;
		else if (mode == "median") _getValueFunc = &FilterImage::getMedianValue;
		else if (mode == "gaussian") _getValueFunc = &FilterImage::getGaussianValue;
		dest = Mat(source.rows, source.cols, CV_8UC1);
		uchar* destPtr;
		for (int row = 0; row < source.rows; ++row) {
			destPtr = dest.ptr<uchar>(row);
			for (int col = 0; col < source.cols; ++col) {
				vector<uchar> pixelList = getConvolutionList(source, row, col, kernelSize);
				*destPtr++ = (this->*_getValueFunc)(pixelList);
			}
		}
	}

	// 取得Mean filter image
	void getMeanFilterImage(Mat& source, Mat& dest, int kernelSize) {
		dest = Mat(source.rows, source.cols, CV_8UC1);
		uchar* destPtr;
		for (int row = 0; row < source.rows; ++row) {
			destPtr = dest.ptr<uchar>(row);
			for (int col = 0; col < source.cols; ++col) {
				vector<uchar> pixelList = getConvolutionList(source, row, col, kernelSize);
				*destPtr++ = getMeanValue(pixelList);
			}
		}
	}
};

int main() {
	cout << "[Main] Start to processing images, please wait..." << endl;

	// 設定各圖像處理參數
	vector<string> imageFileList{ "House256_noise.png", "Lena_gray.png", "Mandrill_gray.png", "Peppers_noise.png"};
	vector<string> filterModeList{ "mean", "median", "gaussian" };
	vector<int> kernelSizeList{ 3, 7, 3, 7, 5 };
	//vector<string> imageFileList{ "House256_noise.png" };
	// 執行各圖像處理
	for (const string& sourceImageFile : imageFileList) {
		const Mat sourceImage = imread("../Image/Source/" + sourceImageFile);
		imshow("Source image " + sourceImageFile, sourceImage);
		FilterImage filter = FilterImage(sourceImage, sourceImageFile);
		filter.setGaussianKernel(5, 0.707);
		for (const string& filterMode : filterModeList) {
			for (const int& kernelSize : kernelSizeList) {
				Mat resultImage = Mat(sourceImage.rows, sourceImage.cols, CV_8UC1);
				Mat sampleImage;
				sourceImage.copyTo(sampleImage);
				filter.updateGrayScaleImage(sampleImage);
				for (int repeat = 1; repeat <= 7; ++repeat) {
					filter.getFilterImageByMode(sampleImage, resultImage, filterMode, 3);
					
					string saveFilePath = "../" + filter.FileName + "/";
					string saveFileName = filterMode + to_string(kernelSize) + "x" + to_string(kernelSize) + "_" + sourceImageFile;
					//imshow(saveFileName, resultImage);
					imwrite(saveFilePath + saveFileName, resultImage);
					resultImage.copyTo(sampleImage);
				}
			}
		}
	}
	cout << "[Main] All image processing complete." << endl;
	cv::waitKey();
	cv::destroyAllWindows();
	return 0;
}