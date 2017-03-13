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

int main() {
	std::cout << "OpenCV Version: " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << std::endl;
	//for gray scale images:
	Mat img = imread("lena.jpg", IMREAD_GRAYSCALE);
	if(! img.data ){
		cout <<  "Could not open or find the image" << std::endl ;
		return -1;
	}
	//for color images:
	//Mat img = imread("lena.bmp");
	Mat gray;
	/*
	namedWindow("Original Image", WINDOW_AUTOSIZE);
	imshow("Original Image", img);
	cv::waitKey(0);
	*/
	// converting from color to grayscale:
	// cvtColor(img,gray,COLOR_BGR2GRAY);
	Ptr<Feature2D> sift_f2d = xfeatures2d::SIFT::create();
	Ptr<Feature2D> surf_f2d = xfeatures2d::SURF::create();
	gray = img;
	std::vector<KeyPoint> sift_keypoints;
	std::vector<KeyPoint> surf_keypoints;
	sift_f2d->detect(gray, sift_keypoints);
	surf_f2d->detect(gray, surf_keypoints);
	Mat sift_descriptors;
	Mat surf_descriptors;
	sift_f2d->compute(gray, sift_keypoints, sift_descriptors);
	surf_f2d->compute(gray, surf_keypoints, surf_descriptors);
	Mat img_sift_keypoints;
	Mat img_surf_keypoints;
	drawKeypoints(gray, sift_keypoints, img_sift_keypoints, Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(gray, surf_keypoints, img_surf_keypoints, Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	namedWindow("SIFT KeyPoints", WINDOW_AUTOSIZE);
	imshow("SIFT KeyPoints", img_sift_keypoints);
	cv::waitKey(0);
	namedWindow("SURF KeyPoints", WINDOW_AUTOSIZE);
	imshow("SURF KeyPoints", img_surf_keypoints);
	cv::waitKey(0);
	return 0;
}

