#include<opencv2/opencv.hpp>
#include<time.h> 
#include<cmath>

using namespace std;
using namespace cv;

enum class EdgeOperatorName {
	PREWITT_VERTICAL,
	PREWITT_HORIZONTAL,
	PREWITT_BOTH,
	SOBEL_VERTICAL,
	SOBEL_HORIZONTAL,
	SOBEL_BOTH,
	LAPLACIAN_ONE,
	LAPLACIAN_TWO
};

class EdgeDetector {
private:
	const int KERNEL_SIZE = 3;
	vector<int>* _kernelPtr = nullptr;
	vector<int>* _kernelPtr2 = nullptr;
	vector<int> _prewittVertical{ -1, 0, 1, -1, 0, 1, -1, 0, 1 };
	vector<int> _prewittHorizontal{ -1, -1, -1, 0, 0, 0, 1, 1, 1 };
	vector<int> _sobelVertical{ -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	vector<int> _sobelHorizontal{ -1, -2, -1, 0, 0, 0, 1, 2, 1 };
	vector<int> _laplacianKernelOne{ 0, 1, 0, 1, -4, 1, 0, 1, 0 };
	vector<int> _laplacianKernelTwo{ 1, 1, 1, 1, -8, 1, 1, 1, 1 };
	vector<double> _gaussianKernel;
	int rowIndex, colIndex;
	
	// 用 3*3 卷積回傳鄰近的像素點
	vector<uchar> getConvolutionList(const Mat& source) {
		vector<uchar> pixelList(KERNEL_SIZE * KERNEL_SIZE);
		int kernelIndex = 0;
		int start = -KERNEL_SIZE / 2, end = start + KERNEL_SIZE - 1; // kernelSize = 3*3
		for (int windowRow = start; windowRow <= end; ++windowRow) {
			for (int windowCol = start; windowCol <= end; ++windowCol) {
				int nRow = rowIndex + windowRow, nCol = colIndex + windowCol;
				if (nRow >= 0 && nRow < source.rows && nCol >= 0 && nCol < source.cols) {
					pixelList.at(kernelIndex++) = source.at<uchar>(nRow, nCol);
				} else {
					pixelList.at(kernelIndex++) = 0;
				}
			}
		}
		return pixelList;
	}

	// 取得卷積範圍內的 Gaussian 值
	uchar getGaussianValue(vector<uchar> pixelList) {
		double value = 0;
		for (int pixelIndex = 0; pixelIndex < pixelList.size(); ++pixelIndex) {
			value += (double)pixelList.at(pixelIndex) * (double)_gaussianKernel.at(pixelIndex);
		}
		return (uchar)round(value);
	}

	// 取得特定方法過濾圖像
	void getGaussianFilterImage(Mat& source, Mat& dest) {
		dest = Mat(source.rows, source.cols, CV_8UC1);
		uchar* destPtr;
		for (rowIndex = 0; rowIndex < source.rows; ++rowIndex) {
			destPtr = dest.ptr<uchar>(rowIndex);
			for (colIndex = 0; colIndex < source.cols; ++colIndex) {
				vector<uchar> pixelList = getConvolutionList(source);
				*destPtr++ = getGaussianValue(pixelList);
			}
		}
	}

	int getEdgeValue(const Mat& source, const vector<uchar>& pixelList, const vector<int>& kernel) {
		const int start = -KERNEL_SIZE / 2;
		const int end = start + KERNEL_SIZE - 1;
		int edgeValue = 0;
		int kernelIndex = 0;
		for (int windowRow = start; windowRow <= end; ++windowRow) {
			for (int windowCol = start; windowCol <= end; ++windowCol) {
				int nRow = rowIndex + windowRow, nCol = colIndex + windowCol;
				if (nRow >= 0 && nRow < source.rows && nCol >= 0 && nCol < source.cols) {
					edgeValue += kernel.at(kernelIndex) * pixelList.at(kernelIndex);
				}
				++kernelIndex;
			}
		}
		return edgeValue;
	}

	bool isOverThreshold(int threshold, int value1, int value2) {
		int tangent = sqrt(pow(value1, 2) + pow(value2, 2));
		return tangent >= threshold;
	}

	void detectEdge(const Mat& source, Mat& dest, EdgeOperatorName edgeOperator, int threshold) {
		dest.create(source.rows, source.cols, CV_8UC1);
		dest = Mat::zeros(source.rows, source.cols, CV_8UC1);
		uchar* destPtr;
		for (rowIndex = 0; rowIndex < source.rows; ++rowIndex) {
			destPtr = dest.ptr<uchar>(rowIndex);
			for (colIndex = 0; colIndex < source.cols; ++colIndex) {
				vector<uchar> pixelList = getConvolutionList(source);
				int edgeValue = getEdgeValue(source, pixelList, *_kernelPtr);
				int edgeValue2 = 0;
				if (_kernelPtr2 != nullptr) edgeValue2 = getEdgeValue(source, pixelList, *_kernelPtr2);
				if (isOverThreshold(threshold, edgeValue, edgeValue2)) *destPtr = 255;
				++destPtr;
			}
		}
	}

public:

	EdgeDetector() {
		rowIndex = 0;
		colIndex = 0;
	}

	// 類別初始化
	void transformFileName(string file, string& fileName, string& fileExt) {
		size_t dot_pos = file.rfind('.');
		fileName = file.substr(0, dot_pos);
		fileExt = file.substr(dot_pos);
	}

	// 依照給定的 kernel 大小和標準化參數給出 kernel權重值
	void setGaussianKernel(double standardDeviation) {
		_gaussianKernel = vector<double>(KERNEL_SIZE * KERNEL_SIZE);
		double weightTotal = 0;
		double kernelIndex = 0;
		int start = -KERNEL_SIZE / 2;
		int end = start + KERNEL_SIZE - 1;
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
	void updateGrayScaleImage(const Mat& source, Mat& dest) {
		dest.create(source.rows, source.cols, CV_8UC1);
		const uchar* imagePtr;
		uchar* gray;
		for (int row = 0; row < source.rows; ++row) {
			imagePtr = source.ptr<uchar>(row);
			gray = dest.ptr<uchar>(row);
			for (int col = 0; col < source.cols; ++col) {
				uchar blue = *imagePtr++, green = *imagePtr++, red = *imagePtr++;
				*gray++ = (0.3 * red) + (0.59 * green) + (0.11 * blue);
			}
		}
	}

	// 取得特定方法過濾圖像
	void getEdgeImageByMethod(const Mat& source, Mat& dest, EdgeOperatorName edgeOperator, int threshold) {
		_kernelPtr = nullptr;
		_kernelPtr2 = nullptr;
		if (edgeOperator == EdgeOperatorName::PREWITT_VERTICAL) _kernelPtr = &_prewittVertical;
		else if (edgeOperator == EdgeOperatorName::PREWITT_HORIZONTAL) _kernelPtr = &_prewittHorizontal;
		else if (edgeOperator == EdgeOperatorName::SOBEL_VERTICAL) _kernelPtr = &_sobelVertical;
		else if (edgeOperator == EdgeOperatorName::SOBEL_HORIZONTAL) _kernelPtr = &_sobelHorizontal;
		else if (edgeOperator == EdgeOperatorName::LAPLACIAN_ONE) _kernelPtr = &_laplacianKernelOne;
		else if (edgeOperator == EdgeOperatorName::LAPLACIAN_TWO) _kernelPtr = &_laplacianKernelTwo;
		else if (edgeOperator == EdgeOperatorName::PREWITT_BOTH) {
			_kernelPtr = &_prewittVertical;
			_kernelPtr2 = &_prewittHorizontal;
		}
		else if (edgeOperator == EdgeOperatorName::SOBEL_BOTH) {
			_kernelPtr = &_sobelVertical;
			_kernelPtr2 = &_sobelHorizontal;
		}
		detectEdge(source, dest, edgeOperator, threshold);
	}
};

int main() {
	cout << "[Main] Start to processing images, please wait..." << endl;

	// 設定各圖像處理參數
	vector<string> imageFileList{ 
		"House512.png", 
		//"Lena.png", 
		//"Mandrill.png"
	};
	
	vector<int> thresholdList{
		60, 20
	};

	// 執行各圖像處理
	EdgeDetector edgeDetector;
	for (const string& sourceImageFile : imageFileList) {
		const Mat sourceImage = imread("../Image/Source/" + sourceImageFile);
		string fileName, fileExt;
		edgeDetector.transformFileName(sourceImageFile, fileName, fileExt);
		Mat grayImage;
		edgeDetector.updateGrayScaleImage(sourceImage, grayImage);
		imshow(fileName, grayImage);
		Mat resultImage;
		edgeDetector.getEdgeImageByMethod(grayImage, resultImage, EdgeOperatorName::PREWITT_VERTICAL, thresholdList.at(0));
		imshow(fileName + " prewitt vertical", resultImage);
		edgeDetector.getEdgeImageByMethod(grayImage, resultImage, EdgeOperatorName::PREWITT_HORIZONTAL, thresholdList.at(0));
		imshow(fileName + " prewitt horizontal", resultImage);
		edgeDetector.getEdgeImageByMethod(grayImage, resultImage, EdgeOperatorName::PREWITT_BOTH, thresholdList.at(0));
		imshow(fileName + " prewitt both", resultImage);
		edgeDetector.getEdgeImageByMethod(grayImage, resultImage, EdgeOperatorName::SOBEL_VERTICAL, thresholdList.at(0));
		imshow(fileName + " sobel vertical", resultImage);
		edgeDetector.getEdgeImageByMethod(grayImage, resultImage, EdgeOperatorName::SOBEL_HORIZONTAL, thresholdList.at(0));
		imshow(fileName + " sobel horizontal", resultImage);
		edgeDetector.getEdgeImageByMethod(grayImage, resultImage, EdgeOperatorName::SOBEL_BOTH, thresholdList.at(0));
		imshow(fileName + " sobel both", resultImage);
		edgeDetector.getEdgeImageByMethod(grayImage, resultImage, EdgeOperatorName::LAPLACIAN_ONE, thresholdList.at(1));
		imshow(fileName + " laplacian 1", resultImage);
		edgeDetector.getEdgeImageByMethod(grayImage, resultImage, EdgeOperatorName::LAPLACIAN_TWO, thresholdList.at(1));
		imshow(fileName + " laplacian 2", resultImage);
	}
	cout << "\n[Main] All image processing complete." << endl;
	cv::waitKey();
	cv::destroyAllWindows();
	return 0;
}