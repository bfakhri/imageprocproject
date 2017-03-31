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

const int delta_d = 1;	// Allowable diff in disparity against GT for a "correct" match

bool dist(DMatch m1, DMatch m2){
	return m1.distance < m2.distance;
}

int main() {
	std::cout << "OpenCV Version: " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << std::endl;
	// Load the images
	Mat img2 = imread("im2.png", IMREAD_GRAYSCALE);
	Mat img6 = imread("im6.png", IMREAD_GRAYSCALE);
	Mat disp6 = imread("disp6.png", IMREAD_GRAYSCALE);
	if(! (img2.data && img6.data) ){
		cout <<  "Could not open or find the images" << std::endl ;
		return -1;
	}

	// Create Feature Detector Objects
	Ptr<Feature2D> surf_f2d = xfeatures2d::SURF::create();

	// Detect Key Points
	vector<KeyPoint> img2_surf_kps;
	vector<KeyPoint> img6_surf_kps;
	surf_f2d->detect(img2, img2_surf_kps);
	surf_f2d->detect(img6, img6_surf_kps);

	// Create Descriptors for the keypoints
	Mat img2_surf_descriptors;
	Mat img6_surf_descriptors;
	surf_f2d->compute(img2, img2_surf_kps, img2_surf_descriptors);
	surf_f2d->compute(img6, img6_surf_kps, img6_surf_descriptors);

	// Draw the keypoints to new images and display them
	Mat draw_img2_surf_kps;
	Mat draw_img6_surf_kps;
	drawKeypoints(img2, img2_surf_kps, draw_img2_surf_kps, Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(img6, img6_surf_kps, draw_img6_surf_kps, Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	
	// Display the keypoints of all images
	namedWindow("IM2 SURF KeyPoints", WINDOW_AUTOSIZE);
	imshow("IM2 SURF KeyPoints", draw_img2_surf_kps);
	imwrite("./output/62_img2_surf_kps.jpg", draw_img2_surf_kps); 
	namedWindow("IM6 SURF KeyPoints", WINDOW_AUTOSIZE);
	imshow("IM6 SURF KeyPoints", draw_img6_surf_kps);
	imwrite("./output/62_img6_surf_kps.jpg", draw_img6_surf_kps); 
	cv::waitKey(0);
	//destroyWindow("IM2 SURF KeyPoints");
	//destroyWindow("IM6 SURF KeyPoints");

	// Begin Matching Keypoints
	FlannBasedMatcher flann_matcher;
	BFMatcher BruteForceMatcher = BFMatcher(NORM_L1, true);

	vector<DMatch> surf_matches;
	cout << "FLANN!" << endl;
	flann_matcher.match(img6_surf_descriptors, img2_surf_descriptors, surf_matches);

	// Sort the matches based on distance 
	sort(surf_matches.begin(), surf_matches.end(), dist);

	// Find best matches based on distance
	vector<DMatch> best_matches;
	for(int m=0; m<surf_matches.size(); m++){
		int img6_idx = surf_matches[m].queryIdx;
		int img2_idx = surf_matches[m].trainIdx;
		int img6_x = img6_surf_kps[img6_idx].pt.x;
		int img6_y = img6_surf_kps[img6_idx].pt.y;
		int img2_x = img2_surf_kps[img2_idx].pt.x;
		int img2_y = img2_surf_kps[img2_idx].pt.y;
		double disp_xy = sqrt(pow(img2_y-img6_y, 2)+pow(img2_x-img6_x, 2));
		double gt_xy_disp = (disp6.at<uchar>(img6_x, img6_y))/float(4);
		
		if(abs(disp_xy - gt_xy_disp) < delta_d)
			best_matches.push_back(surf_matches[m]);
	}

	// Draw and display the matches
	Mat draw_surf_matches;
	Mat draw_best_surf_matches;
	drawMatches(img6, img6_surf_kps, img2, img2_surf_kps, surf_matches, draw_surf_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	drawMatches(img6, img6_surf_kps, img2, img2_surf_kps, best_matches, draw_best_surf_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	namedWindow("SURF Matches", WINDOW_AUTOSIZE);
	imshow("SURF Matches", draw_surf_matches);
	imwrite("./output/62_surf_matches.jpg", draw_surf_matches); 
	namedWindow("Best SURF Matches", WINDOW_AUTOSIZE);
	imshow("Best SURF Matches", draw_best_surf_matches);
	imwrite("./output/62_best_surf_matches.jpg", draw_best_surf_matches); 
	cv::waitKey(0);

	double dist_sum = 0; 
	for(int m=0; m<surf_matches.size(); m++){
		dist_sum += surf_matches[m].distance; 
	}
	double avg = dist_sum/surf_matches.size();

	cout << "SURF match distance sum = " << dist_sum << endl;
	cout << "Avg matching distance = " << avg << endl;

	// Calculate the Disparity (img6 to img2) using only best matches
	double total_disp_x = 0;
	double total_disp_y = 0;
	double total_disp_xy = 0;
	double gt_total_disp_xy = 0;
	double rmse_err = 0; 
	for(int m=0; m<best_matches.size(); m++){
		int img6_idx = best_matches[m].queryIdx;
		int img2_idx = best_matches[m].trainIdx;
		int img6_x = img6_surf_kps[img6_idx].pt.x;
		int img6_y = img6_surf_kps[img6_idx].pt.y;
		int img2_x = img2_surf_kps[img2_idx].pt.x;
		int img2_y = img2_surf_kps[img2_idx].pt.y;
		total_disp_x += abs(img2_x-img6_x);
		total_disp_y += abs(img2_y-img6_y);
		double disp_xy = sqrt(pow(img2_y-img6_y, 2)+pow(img2_x-img6_x, 2));
		total_disp_xy += disp_xy; 
		double gt_xy_disp = (disp6.at<uchar>(img6_x, img6_y))/float(4);
		gt_total_disp_xy += gt_xy_disp;
		rmse_err += pow(disp_xy-gt_xy_disp, 2)/best_matches.size();
	}
	rmse_err = sqrt(rmse_err);

	cout << endl << "--- Image 6-2 Respective Measurements --- " << endl;
	cout << "SURF Avg X Disparity = " << total_disp_x/best_matches.size() << endl;
	cout << "SURF Avg Y Disparity = " << total_disp_y/best_matches.size() << endl;
	cout << "SURF Avg XY Disparity = " << total_disp_xy/best_matches.size() << endl;
	cout << "GT Avg XY Disparity = " << gt_total_disp_xy/best_matches.size() << endl;
	cout << "RMSE XY = " << rmse_err << endl;
	cout << "Percent Incorrectly Matched Points = " << 100*(surf_matches.size() - best_matches.size())/surf_matches.size() << "% with threshold = " << delta_d << endl;

	
	
	
	return 0;
}

