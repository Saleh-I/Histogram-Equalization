#include<opencv2\core.hpp>
#include<opencv2\highgui.hpp>
#include<opencv2\imgcodecs.hpp>
#include<opencv2\imgproc.hpp>

using namespace cv;

int main() {
	Mat img, hist, weighted_hist;
	// read image file
	img = imread("airship.jpg", 0);

	// calculate histogram of the image
	int hsize = 256;
	float range[] = { 0, 256 };  // the upper boundary is exclusive
	const float* histranges = range;
	calcHist(&img, 1, 0, Mat(), hist, 1, &hsize, &histranges);

	// calculate weighted histogram
	weighted_hist = hist / sum(hist);

	// calculate cumulative histogram
	Mat acc_hist = Mat::zeros(weighted_hist.size(), weighted_hist.type());
	acc_hist.at<float>(0) = weighted_hist.at<float>(0);
	for (int i = 1; i < 256; i++)
	{
		acc_hist.at<float>(i) = weighted_hist.at<float>(i) + acc_hist.at<float>(i - 1);
	}
	acc_hist = acc_hist * 255;

	// Mapping
	Mat imgClone = Mat::zeros(img.size(), CV_32FC1);
	img.convertTo(imgClone, CV_32FC1);
	Mat output = Mat::zeros(img.size(), CV_32FC1);
	for (int m = 0; m < img.rows; m++)
	{
		for (int n = 0; n < img.cols; n++)
		{
			output.at<float>(m, n) = acc_hist.at<float>(imgClone.at<float>(m, n));
		}
	}

	// quantize output
	output.convertTo(output, CV_8UC1);
	namedWindow("output", WINDOW_AUTOSIZE);
	imshow("output", output);

	waitKey(0);
	return 0;
}