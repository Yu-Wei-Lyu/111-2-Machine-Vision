#include<opencv2/opencv.hpp>
#include<time.h> 
#include<cmath>

using namespace std;
using namespace cv;

class EdgeDetector {
private:
	const int KERNEL_SIZE = 3;
	vector<int>* _kernel = nullptr;
	vector<int> _prewittVertical{ -1, 0, 1, -1, 0, 1, -1, 0, 1 };
	vector<int> _prewittHorizontal{ -1, -1, -1, 0, 0, 0, 1, 1, 1 };
	vector<int> _sobelVertical{ -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	vector<int> _sobelHorizontal{ -1, -2, -1, 0, 0, 0, 1, 2, 1 };
	vector<int> _laplacianKernelOne{ 0, 1, 0, 1, -4, 1, 0, 1, 0 };
	vector<int> _laplacianKernelTwo{ 1, 1, 1, 1, -8, 1, 1, 1, 1 };
	
	// 用 3*3 卷積回傳鄰近的像素點
	vector<uchar> getConvolutionList(const Mat& source, int row, int col) {
		vector<uchar> pixelList(KERNEL_SIZE * KERNEL_SIZE);
		int pixelListIndex = 0;
		int start = -KERNEL_SIZE / 2, end = start + KERNEL_SIZE - 1; // kernelSize = 3*3
		for (int windowRow = start; windowRow <= end; ++windowRow) {
			for (int windowCol = start; windowCol <= end; ++windowCol) {
				int nRow = row + windowRow, nCol = col + windowCol;
				if (nRow >= 0 && nRow < source.rows && nCol >= 0 && nCol < source.cols) {
					pixelList.at(pixelListIndex++) = source.at<uchar>(nRow, nCol);
				}
				else {
					pixelList.at(pixelListIndex++) = 0;
				}
			}
		}
		return pixelList;
	}

public:
	string FileName, FileExt;

	EdgeDetector() {	}

	// 類別初始化
	void transformFile(string file) {
		size_t dot_pos = file.rfind('.');
		FileName = file.substr(0, dot_pos);
		FileExt = file.substr(dot_pos);
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
				_gaussianKernel.at(kernelIndex++) = weight;
				weightTotal += weight;
			}
		}
		double newWeightTotal = 0;
		for (double& weight : _gaussianKernel) {
			weight /= weightTotal;
		}
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

	// 取得特定方法過濾圖像
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
};

class FilterKernel {
public:
	string Mode;
	vector<int> KernelSize;

	FilterKernel(string mode, vector<int> size) {
		this->Mode = mode;
		this->KernelSize = size;
	}
};

int main() {
	cout << "[Main] Start to processing images, please wait..." << endl;

	// 設定各圖像處理參數
	vector<string> imageFileList{ 
		"House256_noise.png", 
		"Lena_gray.png", 
		"Mandrill_gray.png", 
		"Peppers_noise.png"
	};
	vector<FilterKernel> filterKernel{
		FilterKernel("mean", { 3, 7 }),
		FilterKernel("median", { 3, 7 }),
		FilterKernel("gaussian", { 5 })
	};
	
	// 執行各圖像處理
	double processing = 0, previousPercent = 0, currentPercent = 0, processTotal = 140;
	FilterImage filter;
	filter.setGaussianKernel(5, 1.414);
	for (const string& sourceImageFile : imageFileList) {
		const Mat sourceImage = imread("../Image/Source/" + sourceImageFile);
		filter.transformFile(sourceImageFile);
		for (const FilterKernel& fk : filterKernel) {
			for (const int& kernelSize : fk.KernelSize) {
				Mat resultImage = Mat(sourceImage.rows, sourceImage.cols, CV_8UC1);
				Mat sampleImage;
				sourceImage.copyTo(sampleImage);
				filter.updateGrayScaleImage(sampleImage);
				for (int repeat = 1; repeat <= 7; ++repeat) {
					filter.getFilterImageByMode(sampleImage, resultImage, fk.Mode, kernelSize);
					string saveFilePath = "../Image/" + filter.FileName + "/";
					string saveFileName = filter.FileName + "(" + fk.Mode + to_string(kernelSize) + "x" + to_string(kernelSize) + "r" + to_string(repeat) + ")" + filter.FileExt;
					imwrite(saveFilePath + saveFileName, resultImage);
					resultImage.copyTo(sampleImage);
					currentPercent = round((++processing / processTotal) * 100);
					if (previousPercent != currentPercent) {
						cout << "\r" << "Processing : " << currentPercent << "%";
					}
					previousPercent = processing;
				}
			}
		}
	}
	cout << "\n[Main] All image processing complete." << endl;
	cv::waitKey();
	cv::destroyAllWindows();
	return 0;
}