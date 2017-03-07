#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

void main() {
	std::cout << "OpenCV Version: " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << std::endl;
	//for gray scale images:
	Mat img = imread("lena.bmp", IMREAD_GRAYSCALE);
	//for color images:
	//Mat img = imread("lena.bmp");
	Mat gray;
	// converting from color to grayscale:
	// cvtColor(img,gray,COLOR_BGR2GRAY);
	Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
	gray = img;
	std::vector<KeyPoint> keypoints;
	f2d->detect(gray, keypoints);
	Mat descriptors;
	f2d->compute(gray, keypoints, descriptors);
	Mat img_keypoints;
	drawKeypoints(gray, keypoints, img_keypoints, Scalar::all(-1),
	cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}

