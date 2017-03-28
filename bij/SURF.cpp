#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

bool dist(DMatch m1, DMatch m2){
	return m1.distance < m2.distance;
}

int main() {
	std::cout << "OpenCV Version: " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << std::endl;
	//for gray scale images:
	Mat img2 = imread("im2.png", IMREAD_GRAYSCALE);
	Mat img6 = imread("im6.png", IMREAD_GRAYSCALE);
	if(! (img2.data && img6.data) ){
		cout <<  "Could not open or find the images" << std::endl ;
		return -1;
	}
	Mat gray2;
	Mat gray6;
	gray2 = img2;
	gray6 = img6;


	// Create Feature Detector Objects
	Ptr<Feature2D> surf_f2d = xfeatures2d::SURF::create();

	// Detect Key Points
	vector<KeyPoint> img2_surf_kps;
	vector<KeyPoint> img6_surf_kps;
	surf_f2d->detect(gray2, img2_surf_kps);
	surf_f2d->detect(gray6, img6_surf_kps);


	// Create Descriptors for the keypoints
	Mat img2_surf_descriptors;
	Mat img6_surf_descriptors;
	surf_f2d->compute(gray2, img2_surf_kps, img2_surf_descriptors);
	surf_f2d->compute(gray2, img6_surf_kps, img6_surf_descriptors);

	// Draw the keypoints to new images and display them
	Mat draw_img2_surf_kps;
	Mat draw_img6_surf_kps;
	drawKeypoints(gray2, img2_surf_kps, draw_img2_surf_kps, Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(gray6, img6_surf_kps, draw_img6_surf_kps, Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	
	// Display the keypoints of all images
	namedWindow("IM2 SURF KeyPoints", WINDOW_AUTOSIZE);
	imshow("IM2 SURF KeyPoints", draw_img2_surf_kps);
	namedWindow("IM6 SURF KeyPoints", WINDOW_AUTOSIZE);
	imshow("IM6 SURF KeyPoints", draw_img6_surf_kps);
	cv::waitKey(0);
	destroyWindow("IM2 SURF KeyPoints");
	destroyWindow("IM6 SURF KeyPoints");

	// Begin Matching Keypoints
	FlannBasedMatcher flann_matcher;
	BFMatcher BruteForceMatcher = BFMatcher(NORM_L1, true);

	vector<DMatch> flann_surf_matches;
	vector<DMatch> bf_surf_matches;
	cout << "FLANN!" << endl;
	flann_matcher.match(img2_surf_descriptors, img6_surf_descriptors, flann_surf_matches);
	cout << "BruteForce!" << endl;
	BruteForceMatcher.match(img2_surf_descriptors, img6_surf_descriptors, bf_surf_matches);

	cout << "# Flann Matches = " << flann_surf_matches.size() << endl;
	cout << "# BF Matches = " << bf_surf_matches.size() << endl;
	
	// Find average distances for Flann and BF matches
	double flann_sum = 0; 
	for(int m=0; m<flann_surf_matches.size(); m++){
		flann_sum += flann_surf_matches[m].distance;
	}
	double bf_sum = 0; 
	for(int m=0; m<bf_surf_matches.size(); m++){
		bf_sum += bf_surf_matches[m].distance;
	}

	cout << "Flann Avg = " << flann_sum/flann_surf_matches.size() << "\tBF Avg = " << bf_sum/bf_surf_matches.size() << endl;

	// Draw and display the good matches
	Mat draw_surf_matches;
	//drawMatches(img2, img2_surf_kps, img6, img6_surf_kps, surf_good_matches, draw_surf_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	//drawMatches(img2, img2_surf_kps, img6, img6_surf_kps, surf_good_matches, draw_surf_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::DEFAULT);
	drawMatches(img2, img2_surf_kps, img6, img6_surf_kps, bf_surf_matches, draw_surf_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::DEFAULT);
		
	namedWindow("SURF Matches", WINDOW_AUTOSIZE);
	imshow("SURF Matches", draw_surf_matches);

	for(int m=0; m<img2_surf_descriptors.rows; m++){
		//cout << bf_surf_matches[m].distance << endl;
	}


	cv::waitKey(0);
	return 0;
}

