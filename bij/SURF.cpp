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
	// Load the images
	Mat img2 = imread("im2.png", IMREAD_GRAYSCALE);
	Mat img6 = imread("im6.png", IMREAD_GRAYSCALE);
	Mat disp2 = imread("disp2.png", IMREAD_GRAYSCALE);
	Mat disp6 = imread("disp6.png", IMREAD_GRAYSCALE);
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
	vector<KeyPoint> img2_all_surf_kps;
	vector<KeyPoint> img6_all_surf_kps;
	surf_f2d->detect(gray2, img2_all_surf_kps);
	surf_f2d->detect(gray6, img6_all_surf_kps);

	// Create Descriptors for the keypoints
	Mat img2_surf_descriptors;
	Mat img6_surf_descriptors;
	surf_f2d->compute(gray2, img2_all_surf_kps, img2_surf_descriptors);
	surf_f2d->compute(gray2, img6_all_surf_kps, img6_surf_descriptors);

	// Draw the keypoints to new images and display them
	Mat draw_img2_all_surf_kps;
	Mat draw_img6_all_surf_kps;
	drawKeypoints(gray2, img2_all_surf_kps, draw_img2_all_surf_kps, Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(gray6, img6_all_surf_kps, draw_img6_all_surf_kps, Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	
	// Display the keypoints of all images
	namedWindow("IM2 SURF KeyPoints", WINDOW_AUTOSIZE);
	imshow("IM2 SURF KeyPoints", draw_img2_all_surf_kps);
	namedWindow("IM6 SURF KeyPoints", WINDOW_AUTOSIZE);
	imshow("IM6 SURF KeyPoints", draw_img6_all_surf_kps);
	cv::waitKey(0);
	destroyWindow("IM2 SURF KeyPoints");
	destroyWindow("IM6 SURF KeyPoints");

	// Begin Matching Keypoints
	FlannBasedMatcher flann_matcher;
	BFMatcher BruteForceMatcher = BFMatcher(NORM_L1, true);

	vector<DMatch> surf_matches;
	cout << "FLANN!" << endl;
	flann_matcher.match(img2_surf_descriptors, img6_surf_descriptors, surf_matches);

	// Draw and display the matches
	Mat draw_surf_matches;
	drawMatches(img2, img2_all_surf_kps, img6, img6_all_surf_kps, surf_matches, draw_surf_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	//drawMatches(img2, img2_all_surf_kps, img6, img6_all_surf_kps, surf_good_matches, draw_surf_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::DEFAULT);	


	// Sort the matches based on distance 
	sort(surf_matches.begin(), surf_matches.end(), dist);

	namedWindow("SURF Matches", WINDOW_AUTOSIZE);
	imshow("SURF Matches", draw_surf_matches);
	cv::waitKey(0);

	double dist_sum = 0; 
	for(int m=0; m<surf_matches.size(); m++){
		dist_sum += surf_matches[m].distance; 
		cout << surf_matches[m].distance << endl;
	}
	double avg = dist_sum/surf_matches.size();

	cout << "SURF match distance sum = " << dist_sum << endl;
	cout << "Avg matching distance = " << avg << endl;


	// Calculate the Disparity (img2 to img6)
	double sum = 0; 
	double gt_sum = 0; 
	for(int m=0; m<surf_matches.size(); m++){
		//cout << "------" << endl << surf_matches[m].imgIdx << endl << surf_matches[m].queryIdx << endl << surf_matches[m].trainIdx << endl << "------" << endl;
		int img2_idx = surf_matches[m].queryIdx;
		int img6_idx = surf_matches[m].trainIdx;
		int img2_x = img2_all_surf_kps[img2_idx].pt.x;
		int img2_y = img2_all_surf_kps[img2_idx].pt.y;
		int img6_x = img6_all_surf_kps[img6_idx].pt.x;
		int img6_y = img6_all_surf_kps[img6_idx].pt.y;
		double euc_dist = sqrt(pow(img2_y-img6_y, 2)+pow(img2_x-img6_x, 2));
		//cout << euc_dist << endl;
		sum += euc_dist;

		gt_sum += (disp2.at<uchar>(img2_x, img2_y))/float(4);
	}

	cout << "--- Image 2 Respective Measurements --- " << endl;
	cout << "GT Sum of Distances (Disparity) = " << gt_sum << endl;
	cout << "Sum of Distances (Disparity) = " << sum << endl;
	cout << "GT Average Distance (Disparity) = " << gt_sum/surf_matches.size() << endl;
	cout << "Average Distance (Disparity) = " << sum/surf_matches.size() << endl;

	// Calculate the Disparity (img6 to img2)
	sum = 0; 
	gt_sum = 0; 
	for(int m=0; m<surf_matches.size(); m++){
		//cout << "------" << endl << surf_matches[m].imgIdx << endl << surf_matches[m].queryIdx << endl << surf_matches[m].trainIdx << endl << "------" << endl;
		int img2_idx = surf_matches[m].trainIdx;
		int img6_idx = surf_matches[m].queryIdx;
		int img2_x = img2_all_surf_kps[img2_idx].pt.x;
		int img2_y = img2_all_surf_kps[img2_idx].pt.y;
		int img6_x = img6_all_surf_kps[img6_idx].pt.x;
		int img6_y = img6_all_surf_kps[img6_idx].pt.y;
		double euc_dist = sqrt(pow(img2_y-img6_y, 2)+pow(img2_x-img6_x, 2));
		//cout << euc_dist << endl;
		sum += euc_dist;
		gt_sum += (disp6.at<uchar>(img2_x, img2_y))/float(4);
	}

	cout << "--- Image 2 Respective Measurements --- " << endl;
	cout << "GT Sum of Distances (Disparity) = " << gt_sum << endl;
	cout << "Sum of Distances (Disparity) = " << sum << endl;
	cout << "GT Average Distance (Disparity) = " << gt_sum/surf_matches.size() << endl;
	cout << "Average Distance (Disparity) = " << sum/surf_matches.size() << endl;

	return 0;
}

