#include<opencv2/opencv.hpp>
#define SOURCE_PATH  "./Image/Source/"
#define GRAYSCALE_PATH  "./Image/Grayscale/"
#define BINARY_PATH  "./Image/Binary/"
#define INDEX_COLOR_PATH  "./Image/Index color/"
#define RESIZE_PATH  "./Image/Resize/"
#define RESIZE_INTERPOLATION_PATH  "./Image/Resize(interpolation)/"
using namespace std;
using namespace cv;

void grayScaleImage(string fileName, bool showImage = false);
void binaryImage(string fileName, bool showImage = false);

int main()
{
	vector<string> image_name = { "House256.png", "House512.png", "JellyBeans.png", "Lena.png", "Mandrill.png", "Peppers.png" };
	for (int i = 0; i < image_name.size(); i++) {
		binaryImage(image_name.at(i), false);
	}
	waitKey();
	destroyAllWindows();
	return 0;
}

void grayScaleImage(string fileName, bool showImage) {
	Mat image = imread(SOURCE_PATH + fileName);
	Mat grayscale_image = Mat(image.rows, image.cols, CV_8UC1);
	uchar* p;
	int pixel_length = image.rows * image.cols;
	for (int col = 0; col < image.rows; col++) {
		p = image.ptr<uchar>(col);
		uchar* gray = grayscale_image.ptr<uchar>(col);
		for (int row = 0; row < image.cols; row++) {
			uchar blue = *p++, green = *p++, red = *p++;
			*gray++ = (0.3 * red) + (0.59 * green) + (0.11 * blue);
		}
	}
	if (showImage) {
		imshow(SOURCE_PATH + fileName, image);
		imshow(GRAYSCALE_PATH + fileName, grayscale_image);
	}
	imwrite(GRAYSCALE_PATH + fileName, grayscale_image);
}

void binaryImage(string fileName, bool showImage) {
	Mat image = imread(GRAYSCALE_PATH + fileName);
	Mat binary_image = Mat(image.rows, image.cols, CV_8UC1);
	uchar* p;
	for (int col = 0; col < image.rows; col++) {
		p = image.ptr<uchar>(col);
		uchar* binary = binary_image.ptr<uchar>(col);
		for (int row = 0; row < image.cols; row++) {
			uchar gray = *p++;
			*binary++ = gray;
			cout << (int)gray << ":" << (int)binary << endl;
		}
	}
	if (showImage) {
		imshow(GRAYSCALE_PATH + fileName, image);
		imshow(BINARY_PATH + fileName, binary_image);
	}
	imwrite(BINARY_PATH + fileName, binary_image);
}
