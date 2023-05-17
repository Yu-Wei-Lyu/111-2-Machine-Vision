#include<opencv2/opencv.hpp>
#include<time.h> 
#include<cmath>

using namespace std;
using namespace cv;

typedef enum
{
	PREWITT_VERTICAL,
	PREWITT_HORIZONTAL,
	SOBEL_VERTICAL,
	SOBEL_HORIZONTAL,
	LAPLACIAN_ONE,
	LAPLACIAN_TWO
} edge_operator_t;

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
	int rowIndex, colIndex;
	
	// 用 3*3 卷積回傳鄰近的像素點
	vector<uchar> getConvolutionList(const Mat& source) {
		vector<uchar> pixelList(KERNEL_SIZE * KERNEL_SIZE);
		int pixelListIndex = 0;
		int start = -KERNEL_SIZE / 2, end = start + KERNEL_SIZE - 1; // kernelSize = 3*3
		for (int windowRow = start; windowRow <= end; ++windowRow) {
			for (int windowCol = start; windowCol <= end; ++windowCol) {
				int nRow = rowIndex + windowRow, nCol = colIndex + windowCol;
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

	int getEdgeValue(const Mat& source, vector<uchar> pixelList) {
		int edgeValue = 0;
		int pixelListIndex = 0;
		int start = -KERNEL_SIZE / 2, end = start + KERNEL_SIZE - 1; // kernelSize = 3*3
		for (int windowRow = start; windowRow <= end; ++windowRow) {
			for (int windowCol = start; windowCol <= end; ++windowCol) {
				int nRow = rowIndex + windowRow, nCol = colIndex + windowCol;
				if (nRow >= 0 && nRow < source.rows && nCol >= 0 && nCol < source.cols) {
					edgeValue += (*_kernel).at(pixelListIndex) * pixelList.at(pixelListIndex);
				}
				++pixelListIndex;
			}
		}
		return edgeValue;
	}

public:

	EdgeDetector() {	}

	// 類別初始化
	void transformFileName(string file, string& fileName, string& fileExt) {
		size_t dot_pos = file.rfind('.');
		fileName = file.substr(0, dot_pos);
		fileExt = file.substr(dot_pos);
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
	void getEdgeImageByMethod(Mat& source, Mat& dest, edge_operator_t edgeOperator, int threshold) {
		if (edgeOperator == PREWITT_VERTICAL) _kernel = &_prewittVertical;
		else if (edgeOperator == PREWITT_HORIZONTAL) _kernel = &_prewittHorizontal;
		else if (edgeOperator == SOBEL_VERTICAL) _kernel = &_sobelVertical;
		else if (edgeOperator == SOBEL_HORIZONTAL) _kernel = &_sobelHorizontal;
		else if (edgeOperator == LAPLACIAN_ONE) _kernel = &_laplacianKernelOne;
		else if (edgeOperator == LAPLACIAN_TWO) _kernel = &_laplacianKernelTwo;
		dest.create(source.rows, source.cols, CV_8UC1);
		dest = Mat::zeros(source.rows, source.cols, CV_8UC1);
		uchar* destPtr;
		for (rowIndex = 0; rowIndex < source.rows; ++rowIndex) {
			destPtr = dest.ptr<uchar>(rowIndex);
			for (colIndex = 0; colIndex < source.cols; ++colIndex) {
				vector<uchar> pixelList = getConvolutionList(source);
				int edgeValue = getEdgeValue(source, pixelList);
				if (abs(edgeValue) >= threshold) *destPtr = 255;
				++destPtr;
			}
		}
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
		200
	};

	// 執行各圖像處理
	EdgeDetector edgeDetector;
	for (const string& sourceImageFile : imageFileList) {
		const Mat sourceImage = imread("../Image/Source/" + sourceImageFile);
		string fileName, fileExt;
		edgeDetector.transformFileName(sourceImageFile, fileName, fileExt);
		cout << fileName << " and " << fileExt << endl;
		Mat grayImage;
		edgeDetector.updateGrayScaleImage(sourceImage, grayImage);
		imshow(fileName, grayImage);
		Mat resultImage;
		edgeDetector.getEdgeImageByMethod(grayImage, resultImage, PREWITT_VERTICAL, 200);
		imshow(fileName + "prewitt vertical", resultImage);
		edgeDetector.getEdgeImageByMethod(grayImage, resultImage, PREWITT_HORIZONTAL, 200);
		imshow(fileName + "prewitt horizontal", resultImage);
	}
	cout << "\n[Main] All image processing complete." << endl;
	cv::waitKey();
	cv::destroyAllWindows();
	return 0;
}