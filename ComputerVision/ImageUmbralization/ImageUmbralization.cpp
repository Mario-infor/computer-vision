#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <ctype.h>
#include <cstring>
#include <cmath>

using namespace std;
using namespace cv;

// Loop through image matrix.
void histo(Mat& I, Mat& H) {

	uchar* ptr;

	H = Mat::zeros(1, 256, CV_32SC1);
	for (int i = 0; i < I.rows; ++i) {

		ptr = I.ptr<uchar>(i);
		for (int j = 0; j < I.cols; ++j, ++ptr) {

			H.at<int>(0, *ptr)++;
		}
	}
}

void umbral(float miuOne, float miuTwo, float desvOne, float desvTwo, float& x1, float& x2) {

	float a = pow(desvTwo, 2) - pow(desvOne, 2);
	float b = -2 * (miuOne * pow(desvTwo, 2) - miuTwo * pow(desvOne, 2));
	float c = (pow(miuOne, 2) * pow(desvTwo, 2)) - (pow(miuTwo, 2) * pow(desvOne, 2)) - (2 * pow(desvOne, 2) * pow(desvTwo, 2) * log10(desvOne / desvTwo));

	float d = pow(b, 2) - (4 * a * c);

	if (d == 0) {
		x1 = -b / (2 * a);
		x2 = -b / (2 * a);
	}
	if (d < 0) {
		x1 = 0;
		x2 = 0;
	}

	x1 = (-b + sqrt(d)) / (2 * a);
	x2 = (-b - sqrt(d)) / (2 * a);
}

int main() {
	Mat I, H;

	I = imread("Img/Jaguar1.png", IMREAD_GRAYSCALE);

	namedWindow("Imagen");

	histo(I, H);

	for (int i = 0; i < 256; ++i) {
		cout << i << " -> " << H.at<int>(0, i) << endl;
	}

	float x1, x2 = -1;
	umbral(30, 20, 4, 5, x1, x2);

	std::cout << "x1 = " << x1 << '\n';
	std::cout << "x2 = " << x2 << '\n';

	return 0;
}
