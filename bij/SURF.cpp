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

int num_points = 50;
int num_good_matches = 10000;

bool dist(DMatch m1, DMatch m2){
	return m1.distance < m2.distance;
}

int main() {
	std::cout << "OpenCV Version: " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << std::endl;
	//for gray scale images:
	Mat img2 = imread("im2.png", IMREAD_GRAYSCALE);
	Mat img6 = imread("im6.png", IMREAD_GRAYSCALE);
	//Mat img2 = imread("disp2.png", IMREAD_GRAYSCALE);
	//Mat img6 = imread("disp2.png", IMREAD_GRAYSCALE);
	//Mat img6 = imread("disp6.png", IMREAD_GRAYSCALE);
	if(! (img2.data && img6.data) ){
		cout <<  "Could not open or find the images" << std::endl ;
		return -1;
	}
	Mat gray2;
	Mat gray6;
	gray2 = img2;
	gray6 = img6;


	// Create Feature Detector Objects
	Ptr<Feature2D> surf_f2d = xfeatures2d::SURF::create(num_points);

	// Detect Key Points
	vector<KeyPoint> img2_surf_keypoints;
	vector<KeyPoint> img6_surf_keypoints;
	surf_f2d->detect(gray2, img2_surf_keypoints);
	surf_f2d->detect(gray6, img6_surf_keypoints);

	KeyPoint temp1 = img2_surf_keypoints[0];
	vector<KeyPoint> img2_all_surf_kps;
	vector<KeyPoint> img6_all_surf_kps;
	for(int r=0; r<img2.rows; r++){
		for(int c=0; c<img2.cols; c++){
			img2_all_surf_kps.push_back(temp1);
			img2_all_surf_kps.back().pt = Point2f(c, r);
			img2_all_surf_kps.back().angle = -1;
			img6_all_surf_kps.push_back(temp1);
			img6_all_surf_kps.back().pt = Point2f(c, r);
			img6_all_surf_kps.back().angle = -1;
		}
	}

	
			
			

	// Create Descriptors for the keypoints
	Mat img2_surf_descriptors;
	Mat img6_surf_descriptors;
	surf_f2d->compute(gray2, img2_all_surf_kps, img2_surf_descriptors);
	surf_f2d->compute(gray2, img6_all_surf_kps, img6_surf_descriptors);

	// Draw the keypoints to new images and display them
	Mat draw_img2_surf_keypoints;
	Mat draw_img6_surf_keypoints;
	drawKeypoints(gray2, img2_all_surf_kps, draw_img2_surf_keypoints, Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(gray6, img6_all_surf_kps, draw_img6_surf_keypoints, Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	
	// Display the keypoints of all images
	namedWindow("IM2 SURF KeyPoints", WINDOW_AUTOSIZE);
	imshow("IM2 SURF KeyPoints", draw_img2_surf_keypoints);
	namedWindow("IM6 SURF KeyPoints", WINDOW_AUTOSIZE);
	imshow("IM6 SURF KeyPoints", draw_img6_surf_keypoints);
	cv::waitKey(0);
	destroyWindow("IM2 SURF KeyPoints");
	destroyWindow("IM6 SURF KeyPoints");

	// Begin Matching Keypoints
	FlannBasedMatcher flann_matcher;
	BFMatcher BruteForceMatcher = BFMatcher(NORM_L1, true);

	vector<DMatch> surf_matches;
	cout << "FLANN!" << endl;
	flann_matcher.match(img2_surf_descriptors, img6_surf_descriptors, surf_matches);
	//cout << "BruteForce!" << endl;
	//BruteForceMatcher.match(img2_surf_descriptors, img6_surf_descriptors, surf_matches);

	// Sort and filter for Filter out the matches with too great of a distance
	sort(surf_matches.begin(), surf_matches.end(), dist);
	vector<DMatch> surf_good_matches;
	for(int m=0; m<num_good_matches; m++){
			surf_good_matches.push_back(surf_matches[m]);
	}

	// Draw and display the good matches
	Mat draw_surf_matches;
	drawMatches(img2, img2_all_surf_kps, img6, img6_all_surf_kps, surf_good_matches, draw_surf_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	//drawMatches(img2, img2_surf_keypoints, img6, img6_surf_keypoints, surf_good_matches, draw_surf_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::DEFAULT);
		
	namedWindow("SURF Matches", WINDOW_AUTOSIZE);
	imshow("SURF Matches", draw_surf_matches);

	for(int m=0; m<img2_surf_descriptors.rows; m++){
		cout << surf_matches[m].distance << endl;
	}


	cv::waitKey(0);
	return 0;
}

