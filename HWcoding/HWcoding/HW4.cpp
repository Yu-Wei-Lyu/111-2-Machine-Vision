#include<opencv2/opencv.hpp>
#include<time.h> 
#include<cmath>

#define PI 3.14159

using namespace std;
using namespace cv;

class FilterImage {
private:
	const string IMAGE_FOLDER = "../Image/";
	Mat _image, _filerImage;
	string _name, _ext;
	int _imageRows, _imageCols;
	vector<double> _gaussianKernel;

public:
	// 類別初始化
	FilterImage(Mat image, string file) {
		size_t dot_pos = file.rfind('.');
		_name = file.substr(0, dot_pos);
		_ext = file.substr(dot_pos);
		_image = image;
		initialize();
	}

	// 類別初始化
	void initialize() {
		_imageRows = _image.rows;
		_imageCols = _image.cols;
		_filerImage = Mat(_imageRows, _imageCols, CV_8UC1);
	}

	// 用kernelSize*kernelSize卷積回傳鄰近的像素點
	vector<uchar> getConvolutionList(int row, int col, int kernelSize) {
		vector<uchar> pixelList;
		int pixelListIndex = 0;
		int start = -kernelSize / 2, end = start + kernelSize - 1; // if kernelSize = 3*3, start = -1, end = 1 etc...
		for (int windowRow = start; windowRow <= end; ++windowRow) {
			for (int windowCol = start; windowCol <= end; ++windowCol) {
				int nRow = row + windowRow, nCol = col + windowCol;
				if (nRow >= 0 && nRow < _imageRows && nCol >= 0 && nCol < _imageCols) {
					pixelList.at(pixelListIndex++) = _image.at<uchar>(nRow, nCol);
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
		cout << " --- " << endl;
		int index = 0;
		for (int i = 0; i < pixelList.size(); i++) {
			cout << (int)pixelList.at(index++) << "\t";
		}
		int medianIndex = (pixelList.size() - 1) / 2;
		cout << "Median value = " << (int)pixelList.at(medianIndex);
		return pixelList.at(medianIndex);
	}

	// 依照給定的 kernel 大小和標準化參數給出 kernel權重值
	void setGaussianKernel(int kernelSize, double standardDeviation) {
		_gaussianKernel.clear();
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

	// 取得
};

int main() {
	cout << "[Main] Start to processing images, please wait..." << endl;

	// 設定各圖像處理參數
	//vector<string> imageFileList{ "House256_noise.png", "Lena_gray.png", "Mandrill_gray.png", "Peppers_noise.png"};
	vector<string> imageFileList{ "House256_noise.png" };
	// 執行各圖像處理
	for (const string& imageFile : imageFileList) {
		Mat image = imread("../Image/Source/" + imageFile);
		FilterImage filerImage = FilterImage(image, imageFile);
		filerImage.setGaussianKernel(7, 0.707);
		break;
	}
	cout << "[Main] All image processing complete." << endl;
	cv::waitKey();
	cv::destroyAllWindows();
	return 0;
}