#include<opencv2/opencv.hpp>
#define SOURCE_PATH  "./Image/Source/"
#define GRAYSCALE_PATH  "./Image/Grayscale/"
#define INDEX_COLOR_PATH  "./Image/Index color/"
#define RESIZE_PATH  "./Image/Resize/"
#define RESIZE_INTERPOLATION_PATH  "./Image/Resize(interpolation)/"
using namespace std;
using namespace cv;

void grayScaleImage(string filePath, string savePath, bool showImage = false);

int main()
{
	vector<string> imagePath = { "House256.png", "House512.png", "JellyBeans.png", "Lena.png", "Mandrill.png", "Peppers.png" };
	for (int i = 0; i < imagePath.size(); i++) {
		grayScaleImage(SOURCE_PATH + imagePath.at(i), GRAYSCALE_PATH + imagePath.at(i), true);
	}
	waitKey();
	destroyAllWindows();
	return 0;
}

void grayScaleImage(string filePath, string savePath, bool showImage) {
	Mat image = imread(filePath);
	Mat grayscale_image = Mat(image.rows, image.cols, CV_8UC1);
	uchar* p = image.ptr(0);
	uchar* gray = grayscale_image.ptr(0);
	int pixel_length = image.rows * image.cols;
	for (int pixel = 0; pixel < pixel_length; pixel++) {
		uchar blue = *p++, green = *p++, red = *p++;
		*gray++ = (0.3 * red) + (0.59 * green) + (0.11 * blue);
	}
	if (showImage) {
		imshow(filePath, image);
		imshow(filePath + "gray", grayscale_image);
	}
	imwrite(savePath, grayscale_image);
}
