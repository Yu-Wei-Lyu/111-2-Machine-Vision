#include<opencv2/opencv.hpp>
#include<time.h> 
#include<cmath>

using namespace std;
using namespace cv;

class FilterImage {
private:
	const string BINARY_FOLDER = "../Image/Binary/";
	Mat _image, _filerImage;
	string _name, _ext;
	int _imageRows, _imageCols;
	int _layer;

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

	// 用side*side卷積回傳鄰近的像素點
	vector<uchar> getConvolutionList(int row, int col, int side) {
		vector<uchar> pixelList(side * side);
		int pixelListIndex = 0;
		int start = -side / 2, end = start + side - 1; // if side = 3*3, start = -1, end = 1 etc...
		for (int windowRow = start; windowRow <= end; ++windowRow) {
			for (int windowCol = start; windowCol < start + side; ++windowCol) {
				int nRow = row + windowRow, nCol = col + windowCol;
				if (nRow >= 0 && nRow < _imageRows && nCol >= 0 && nCol < _imageCols) {
					pixelList.at(pixelListIndex++) = _image.at<uchar>(nRow, nCol);
				} else {
					pixelList.at(pixelListIndex++) = 0;
				}
			}
		}
		int index = 0;
		for (int i = 0; i < side; i++) {
			for (int j = 0; j < side; j++) {
				cout << (int)pixelList.at(index++) << "\t";
			}
			cout << endl;
		}
		getMedianValue(pixelList);
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

};

int main() {
	cout << "[Main] Start to processing images, please wait..." << endl;

	// 設定各圖像處理參數
	vector<string> imageFileList{ "House256_noise.png", "Lena_gray.png", "Mandrill_gray.png", "Peppers_noise.png"};

	// 執行各圖像處理
	for (const string& imageFile : imageFileList) {
		Mat image = imread("../Image/Source/" + imageFile);
		FilterImage filerImage = FilterImage(image, imageFile);
		filerImage.getConvolutionList(1,1, 7);
		break;
	}
	cout << "[Main] All image processing complete." << endl;
	cv::waitKey();
	cv::destroyAllWindows();
	return 0;
}