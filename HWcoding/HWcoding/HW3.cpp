#include<opencv2/opencv.hpp>
#include<time.h> 
#include<cmath>

#define COLOR_BLACK 0
#define COLOR_GRAY 128
#define COLOR_WHITE 255

using namespace std;
using namespace cv;

class QuadTreeImage {
private:
	const string BINARY_FOLDER = "../Image/Binary/", QUADTREE_FOLDER = "../Image/QuadTree/";
	Mat _image, _grayImage, _binaryImage, _quadTreeImage;
	string _name;
	int _imageRows, _imageCols;
	int _layer;
	
public:
	// 類別初始化
	QuadTreeImage(Mat image, string name) {
		_name = name;
		_image = image;
		initialize();
	}

	// 類別初始化
	void initialize() {
		_imageRows = _image.rows;
		_imageCols = _image.cols;
		_grayImage = Mat(_imageRows, _imageCols, CV_8UC1);
		_binaryImage = Mat(_imageRows, _imageCols, CV_8UC1);
		_quadTreeImage = Mat(_imageRows, _imageCols, CV_8UC1, COLOR_GRAY);
	}

	// 設定灰階化圖像
	void updateGrayScaleImage() {
		const uchar* imagePtr;
		uchar* gray;
		for (int row = 0; row < _imageRows; ++row) {
			imagePtr = _image.ptr<uchar>(row);
			gray = _grayImage.ptr<uchar>(row);
			for (int col = 0; col < _imageCols; ++col) {
				uchar blue = *imagePtr++, green = *imagePtr++, red = *imagePtr++;
				*gray++ = (0.3 * red) + (0.59 * green) + (0.11 * blue);
			}
		}
		//imshow(_name + " gray", _grayImage);
	}

	// 依門檻值設定二值化影像
	void updateBinaryImage(const int& threshold) {
		updateGrayScaleImage();
		const uchar* imagePtr;
		uchar* binary;
		for (int row = 0; row < _imageRows; ++row) {
			imagePtr = _grayImage.ptr<uchar>(row);
			binary = _binaryImage.ptr<uchar>(row);
			for (int col = 0; col < _imageCols; ++col) {
				*binary++ = (*imagePtr++ >= threshold) ? 255 : 0;
			}
		}
		imshow(_name + " original", _binaryImage);
		imwrite(BINARY_FOLDER + _name, _binaryImage);
	}

	// 取得特定範圍的所有二階值
	vector<uchar> getBinaryImageValueList(const Point2i& pointBegin, const Point2i& pointEnd) {
		vector<uchar> pixels;
		const uchar* binaryPtr;
		for (int x = pointBegin.x; x < pointEnd.x; x++) {
			binaryPtr = _binaryImage.ptr<uchar>(x) + pointBegin.y;
			for (int y = pointBegin.y; y < pointEnd.y; y++) {
				pixels.push_back(*binaryPtr++);
			}
		}
		return pixels;
	}

	// 判斷2x2像素是否同色 並回傳白、灰或黑
	uchar getMergeColor(const vector<uchar> pixels) {
		int average = 0;
		for (const uchar& pixel : pixels) {
			average += pixel;
		}
		average /= pixels.size();
		return (average == COLOR_BLACK || average == COLOR_WHITE) ? average : COLOR_GRAY;
	}

	// 更新 Quad tree image 部分區塊
	void updateQuadTreeImage(const Point2i& pointBegin, const Point2i& pointEnd, const uchar& color) {
		uchar* resultPtr;
		for (int x = pointBegin.x; x < pointEnd.x; x++) {
			resultPtr = _quadTreeImage.ptr<uchar>(x);
			resultPtr += pointBegin.y;
			for (int y = pointBegin.y; y < pointEnd.y; y++) {
				*resultPtr++ = color;
			}
		}
	}

	// 用遞迴方式創造指定 Layer 的 QuadTree 並儲存結果圖像
	void updateQuadTreeRecursively(int treeHeight, const Point2i pointBegin, const Point2i pointEnd) {
		vector<uchar> pixels = this->getBinaryImageValueList(pointBegin, pointEnd);
		uchar mergedColor = this->getMergeColor(pixels);
		if (mergedColor == COLOR_BLACK || mergedColor == COLOR_WHITE || treeHeight == 0) {
			this->updateQuadTreeImage(pointBegin, pointEnd, mergedColor);
		}
		if (mergedColor == COLOR_GRAY && treeHeight != 0) {
			int midX = (pointBegin.x + pointEnd.x) / 2, midY = (pointBegin.y + pointEnd.y) / 2;
			--treeHeight;
			updateQuadTreeRecursively(treeHeight, Point2i(pointBegin.x, pointBegin.y), Point2i(midX, midY));
			updateQuadTreeRecursively(treeHeight, Point2i(midX, pointBegin.y), Point2i(pointEnd.x, midY));
			updateQuadTreeRecursively(treeHeight, Point2i(pointBegin.x, midY), Point2i(midX, pointEnd.y));
			updateQuadTreeRecursively(treeHeight, Point2i(midX, midY), Point2i(pointEnd.x, pointEnd.y));
		}
	}

	// 以指定 Layer 數更新 Quad tree image
	void updateQuadTreeImage(int layer) {
		_layer = layer;
		_quadTreeImage = Mat(_imageRows, _imageCols, CV_8UC1, 255);
		updateQuadTreeRecursively(layer, Point2i(0, 0), Point2i(_imageRows, _imageCols));
	}

	// 儲存並顯示 Quad tree image
	void saveQuadTreeImage() {
		size_t dot_pos = _name.rfind('.');
		string fileLayerName = _name.substr(0, dot_pos) + "(Layer_" + to_string(_layer) + ")" + _name.substr(dot_pos);
		imshow(fileLayerName, _quadTreeImage);
		imwrite(QUADTREE_FOLDER + fileLayerName, _quadTreeImage);
	}
};

class ImageInfo {
public:
	string Name;
	int BinaryThreshold;

	ImageInfo(string name, int binaryThreshold) {
		this->Name = name;
		this->BinaryThreshold = binaryThreshold;
	}
};

int main() {
	cout << "[Main] Start to processing images, please wait..." << endl;

	// 設定各圖像處理參數
	vector<ImageInfo> imageInfoList{ 
		ImageInfo("1.png", 135), 
		ImageInfo("2.png", 245), 
		ImageInfo("3.png", 155),
		ImageInfo("4.png", 254),
		
	};

	// 執行各圖像處理
	for (ImageInfo& imageInfo : imageInfoList) {
		Mat image = imread("../Image/Source/" + imageInfo.Name);
		QuadTreeImage quadTreeImage = QuadTreeImage(image, imageInfo.Name);
		quadTreeImage.updateBinaryImage(imageInfo.BinaryThreshold);
		int layerTotal = log2(image.rows);
		for (int i = 1; i <= layerTotal; i++) {
			quadTreeImage.updateQuadTreeImage(i);
			quadTreeImage.saveQuadTreeImage();
		}
	}
	cout << "[Main] All image processing complete." << endl;
	cv::waitKey();
	cv::destroyAllWindows();
	return 0;
}