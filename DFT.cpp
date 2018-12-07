#define _USE_MATH_DEFINES
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <stdint.h>
#include <complex>
#include <math.h> 
#include <vector>

using namespace cv;
using namespace std;

Mat FDFT(Mat complexI)
{
	const unsigned int cols = complexI.cols;
	const unsigned int rows = complexI.rows;

	vector <vector<complex<float>>> fmatrix(rows);
	for (uint i = 0; i < rows; i++)
		fmatrix[i].resize(cols);

	for (uint k = 0; k < cols; k++)
		for (uint b = 0; b < rows; b++)
		{
			std::complex<float> sum(0.0, 0.0);
			for (uint a = 0; a < cols; a++)
			{
				int integers = -2 * k * a;
				std::complex<float> my_exponent(0.0, M_PI / cols * (float)integers);
				sum += complexI.at<complex<float>>(a, b) * std::exp(my_exponent);
			}
			fmatrix[k][b] = sum;
		}

	for (uint k = 0; k < cols; k++)
		for (uint l = 0; l < rows; l++)
		{
			std::complex<float> sum(0.0, 0.0);
			for (uint b = 0; b < rows; b++)
			{
				int integers = -2 * l * b;
				std::complex<float> my_exponent(0.0, M_PI / rows * (float)integers);
				sum += fmatrix[k][b] * std::exp(my_exponent);
			}
			complexI.at<complex<float>>(k, l) = sum;
		}
	return complexI;
}

Mat RDFT(Mat complexI)
{
	const unsigned int cols = complexI.cols;
	const unsigned int rows = complexI.rows;

	vector <vector<complex<float>>> fmatrix(rows);
	for (uint i = 0; i < rows; i++)
		fmatrix[i].resize(cols);

	for (uint k = 0; k < cols; k++)
		for (uint b = 0; b < rows; b++)
		{
			std::complex<float> sum(0.0, 0.0);
			for (uint a = 0; a < cols; a++)
			{
				int integers = 2 * k * a;
				std::complex<float> my_exponent(0.0, M_PI / cols * (float)integers);
				sum += complexI.at<complex<float>>(a, b) * std::exp(my_exponent);
			}
			fmatrix[k][b] = sum / (float)cols;
		}

	for (uint k = 0; k < cols; k++)
		for (uint l = 0; l < rows; l++)
		{
			std::complex<float> sum(0.0, 0.0);
			for (uint b = 0; b < rows; b++)
			{
				int integers = 2 * l * b;
				std::complex<float> my_exponent(0.0, M_PI / rows * (float)integers);
				sum += fmatrix[k][b] * std::exp(my_exponent);
			}
			complexI.at<complex<float>>(k, l) = sum / (float)rows;
		}
	return complexI;
}

Mat LowPassFilter(Mat complexI)
{
	int radius = min(complexI.cols, complexI.rows);
	Point2d center;
	center.x = int(complexI.cols / 2);
	center.y = int(complexI.rows / 2);

	for (uint x = 0; x < complexI.cols; x++)
		for (uint y = 0; y < complexI.rows; y++)
		{
			if ((x - center.x) * (x - center.x) + (y - center.y) * (y - center.y) < radius / 1.7 * radius / 1.7)
				complexI.at<complex<float>>(x, y) = complex<float>(0, 0);
		}
	return complexI;
}

void ShowMagnitude(Mat complexI, string windowname)
{
	Mat padded;
	int m = getOptimalDFTSize(complexI.rows);
	int n = getOptimalDFTSize(complexI.cols);
	copyMakeBorder(complexI, padded, 0, m - complexI.rows, 0, n - complexI.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };

	split(complexI, planes);
	magnitude(planes[0], planes[1], planes[0]);
	Mat magI = planes[0];

	magI += Scalar::all(1);
	log(magI, magI);

	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	Mat q0(magI, Rect(0, 0, cx, cy));
	Mat q1(magI, Rect(cx, 0, cx, cy));
	Mat q2(magI, Rect(0, cy, cx, cy));
	Mat q3(magI, Rect(cx, cy, cx, cy));

	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(magI, magI, 0, 1, CV_MINMAX);

	imshow(windowname, magI);
}

int main()
{
	const char* filename = "D:\\Documents\\Visual Studio 2015\\Projects\\ConsoleApplication1\\x64\\Release\\Lenna.jpg";
	
	Mat original = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	int rsz = min(original.cols, original.rows);
	resize(original, original, Size(rsz, rsz), 0, 0, CV_INTER_LINEAR);
	imshow("Input Image", original);

	Mat padded;                            
	int m = getOptimalDFTSize(original.rows);
	int n = getOptimalDFTSize(original.cols); 
	copyMakeBorder(original, padded, 0, m - original.rows, 0, n - original.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	
	merge(planes, 2, complexI);

	complexI = FDFT(complexI);

	ShowMagnitude(complexI, "Before low pass filter");

	complexI = LowPassFilter(complexI);

	ShowMagnitude(complexI, "After low pass filter");

	complexI = RDFT(complexI);

	split(complexI, planes);                 
	Mat magI = planes[0];

	//magI += Scalar::all(1);                    
	//log(magI, magI);

	//magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

	//int cx = magI.cols / 2;
	//int cy = magI.rows / 2;

	//Mat q0(magI, Rect(0, 0, cx, cy));  
	//Mat q1(magI, Rect(cx, 0, cx, cy));  
	//Mat q2(magI, Rect(0, cy, cx, cy));  
	//Mat q3(magI, Rect(cx, cy, cx, cy)); 

	//Mat tmp;                           
	//q0.copyTo(tmp);
	//q3.copyTo(q0);
	//tmp.copyTo(q3);

	//q1.copyTo(tmp);                   
	//q2.copyTo(q1);
	//tmp.copyTo(q2);
	//
	normalize(magI, magI, 0, 1, CV_MINMAX); 

	imshow("Low pass filtered", magI);
	
	waitKey();

	return 0;
}
