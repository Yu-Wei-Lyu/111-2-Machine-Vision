#include<opencv2/opencv.hpp>
#include<time.h> 
#include<cmath>

using namespace std;
using namespace cv;

enum class FilterMethod {
	MEAN,
	MEDIAN,
	GAUSSIAN
};

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

class FilterImage {
private:
	vector<double> _gaussianKernel;
	uchar(FilterImage::* _getValueFunc)(vector<uchar>) = nullptr;

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
				}
				else {
					pixelList.at(pixelListIndex++) = 0;
				}
			}
		}
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

	// 取得卷積範圍內的 Gaussian 值
	uchar getGaussianValue(vector<uchar> pixelList) {
		double value = 0;
		for (int pixelIndex = 0; pixelIndex < pixelList.size(); ++pixelIndex) {
			value += (double)pixelList.at(pixelIndex) * (double)_gaussianKernel.at(pixelIndex);
		}
		return (uchar)round(value);
	}

public:
	string FileName, FileExt;

	FilterImage() {	}

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
	void getFilterImageByMode(Mat& source, Mat& dest, FilterMethod mode, int kernelSize, int deep) {
		Mat referenceImage, processImage;
		if (mode == FilterMethod::MEAN) _getValueFunc = &FilterImage::getMeanValue;
		else if (mode == FilterMethod::MEDIAN) _getValueFunc = &FilterImage::getMedianValue;
		else if (mode == FilterMethod::GAUSSIAN) _getValueFunc = &FilterImage::getGaussianValue;
		source.copyTo(referenceImage);
		processImage.create(source.rows, source.cols, CV_8UC1);
		for (int repeat = 0; repeat < deep; ++repeat) {
			uchar* processPtr;
			for (int row = 0; row < source.rows; ++row) {
				processPtr = processImage.ptr<uchar>(row);
				for (int col = 0; col < source.cols; ++col) {
					vector<uchar> pixelList = getConvolutionList(referenceImage, row, col, kernelSize);
					*processPtr++ = (this->*_getValueFunc)(pixelList);
				}
			}
			processImage.copyTo(referenceImage);
		}
		processImage.copyTo(dest);
	}
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

class ImageInfo {
public:
	string FileName, FileExt;
	int PrewittThreshold, SobelThreshold, LaplacianOneThreshold, LaplacianTwoThreshold;

	ImageInfo(string name, int prewittThreshold, int sobelThreshold, int laplacianOneThreshold, int laplacianTwoThreshold) {
		this->PrewittThreshold = prewittThreshold;
		this->SobelThreshold = sobelThreshold;
		this->LaplacianOneThreshold = laplacianOneThreshold;
		this->LaplacianTwoThreshold = laplacianTwoThreshold;
		size_t dot_pos = name.rfind('.');
		this->FileName = name.substr(0, dot_pos);
		this->FileExt = name.substr(dot_pos);

	}

	// 類別初始化
	void write(string identify, Mat& image) {
		imwrite("../Image/" + this->FileName + "/" + identify + "_" + this->FileName + this->FileExt, image);
	}

};

int main() {
	cout << "[Main] Start to processing images, please wait..." << endl;

	// 設定各圖像處理參數
	vector<ImageInfo> imageList{
		//ImageInfo("House512.png", 70, 70, 10, 25 ),
		ImageInfo("Lena.png", 80, 95, 8, 20),
		//ImageInfo("Mandrill.png", 70, 70, 10, 25)
	};

	// 執行各圖像處理
	EdgeDetector edgeDetector;
	FilterImage filter;
	filter.setGaussianKernel(3, 0.707);
	for (ImageInfo& image : imageList) {
		const Mat sourceImage = imread("../Image/Source/" + image.FileName + image.FileExt);
		Mat grayImage;
		edgeDetector.updateGrayScaleImage(sourceImage, grayImage);
		imshow(image.FileName, grayImage);
		Mat resultImage;
		edgeDetector.getEdgeImageByMethod(grayImage, resultImage, EdgeOperatorName::PREWITT_VERTICAL, image.PrewittThreshold);
		//imshow(image.FileName + " prewitt vertical", resultImage);
		image.write("prewitt_vertical", resultImage);
		edgeDetector.getEdgeImageByMethod(grayImage, resultImage, EdgeOperatorName::PREWITT_HORIZONTAL, image.PrewittThreshold);
		//imshow(image.FileName + " prewitt horizontal", resultImage);
		image.write("prewitt_horizontal", resultImage);
		edgeDetector.getEdgeImageByMethod(grayImage, resultImage, EdgeOperatorName::PREWITT_BOTH, image.PrewittThreshold);
		imshow(image.FileName + " prewitt both", resultImage);
		image.write("prewitt_both", resultImage);
		edgeDetector.getEdgeImageByMethod(grayImage, resultImage, EdgeOperatorName::SOBEL_VERTICAL, image.SobelThreshold);
		//imshow(image.FileName + " sobel vertical", resultImage);
		image.write("sobel_vertical", resultImage);
		edgeDetector.getEdgeImageByMethod(grayImage, resultImage, EdgeOperatorName::SOBEL_HORIZONTAL, image.SobelThreshold);
		//imshow(image.FileName + " sobel horizontal", resultImage);
		image.write("sobel_horizontal", resultImage);
		edgeDetector.getEdgeImageByMethod(grayImage, resultImage, EdgeOperatorName::SOBEL_BOTH, image.SobelThreshold);
		imshow(image.FileName + " sobel both", resultImage);
		image.write("sobel_both", resultImage);
		
		Mat gaussianImage, laplacianImage;
		filter.getFilterImageByMode(grayImage, gaussianImage, FilterMethod::GAUSSIAN, 3, 2);
		//imshow(image.FileName + " gaussian", gaussianImage);
		edgeDetector.getEdgeImageByMethod(gaussianImage, resultImage, EdgeOperatorName::LAPLACIAN_ONE, image.LaplacianOneThreshold);
		imshow(image.FileName + " laplacian 1", resultImage);
		image.write("laplacian1", resultImage);
		edgeDetector.getEdgeImageByMethod(gaussianImage, resultImage, EdgeOperatorName::LAPLACIAN_TWO, image.LaplacianTwoThreshold);
		imshow(image.FileName + " laplacian 2", resultImage);
		image.write("laplacian2", resultImage);
	}
	cout << "\n[Main] All image processing complete." << endl;
	cv::waitKey();
	cv::destroyAllWindows();
	return 0;
}