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

//const int delta_d = 1;	// Allowable diff in disparity against GT for a "correct" match

bool dist(DMatch m1, DMatch m2){
	return m1.distance < m2.distance;
}

int main() {
	for(int delta_d = 1; delta_d < 100; delta_d++)
	{
		// Load the images
		Mat img2 = imread("im2.png", IMREAD_GRAYSCALE);
		Mat img6 = imread("im6.png", IMREAD_GRAYSCALE);
		Mat disp2 = imread("disp2.png", IMREAD_GRAYSCALE);
		if(! (img2.data && img6.data) ){
			cout <<  "Could not open or find the images" << std::endl ;
			return -1;
		}

		// Create Feature Detector Objects
		Ptr<Feature2D> sift_f2d = xfeatures2d::SIFT::create();

		// Detect Key Points
		vector<KeyPoint> img2_sift_kps;
		vector<KeyPoint> img6_sift_kps;
		sift_f2d->detect(img2, img2_sift_kps);
		sift_f2d->detect(img6, img6_sift_kps);

		// Create Descriptors for the keypoints
		Mat img2_sift_descriptors;
		Mat img6_sift_descriptors;
		sift_f2d->compute(img2, img2_sift_kps, img2_sift_descriptors);
		sift_f2d->compute(img6, img6_sift_kps, img6_sift_descriptors);

		// Draw the keypoints to new images and display them
		Mat draw_img2_sift_kps;
		Mat draw_img6_sift_kps;
		drawKeypoints(img2, img2_sift_kps, draw_img2_sift_kps, Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		drawKeypoints(img6, img6_sift_kps, draw_img6_sift_kps, Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		
		// Begin Matching Keypoints
		FlannBasedMatcher flann_matcher;
		BFMatcher BruteForceMatcher = BFMatcher(NORM_L1, true);

		vector<DMatch> sift_matches;
		flann_matcher.match(img2_sift_descriptors, img6_sift_descriptors, sift_matches);

		// Sort the matches based on distance 
		sort(sift_matches.begin(), sift_matches.end(), dist);

		// Find best matches based on distance
		vector<DMatch> best_matches;
		for(int m=0; m<sift_matches.size(); m++){
			int img2_idx = sift_matches[m].queryIdx;
			int img6_idx = sift_matches[m].trainIdx;
			int img2_x = img2_sift_kps[img2_idx].pt.x;
			int img2_y = img2_sift_kps[img2_idx].pt.y;
			int img6_x = img6_sift_kps[img6_idx].pt.x;
			int img6_y = img6_sift_kps[img6_idx].pt.y;
			double disp_xy = sqrt(pow(img2_y-img6_y, 2)+pow(img2_x-img6_x, 2));
			double gt_xy_disp = (disp2.at<uchar>(img2_x, img2_y))/float(4);
			
			if(abs(disp_xy - gt_xy_disp) < delta_d)
				best_matches.push_back(sift_matches[m]);
		}

		// Draw and display the matches
		Mat draw_sift_matches;
		Mat draw_best_sift_matches;
		drawMatches(img2, img2_sift_kps, img6, img6_sift_kps, sift_matches, draw_sift_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		drawMatches(img2, img2_sift_kps, img6, img6_sift_kps, best_matches, draw_best_sift_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		double dist_sum = 0; 
		for(int m=0; m<sift_matches.size(); m++){
			dist_sum += sift_matches[m].distance; 
		}
		double avg = dist_sum/sift_matches.size();


		// Calculate the Disparity (img2 to img6) using only best matches
		double total_disp_x = 0;
		double total_disp_y = 0;
		double total_disp_xy = 0;
		double gt_total_disp_xy = 0;
		double rmse_err = 0; 
		for(int m=0; m<best_matches.size(); m++){
			int img2_idx = best_matches[m].queryIdx;
			int img6_idx = best_matches[m].trainIdx;
			int img2_x = img2_sift_kps[img2_idx].pt.x;
			int img2_y = img2_sift_kps[img2_idx].pt.y;
			int img6_x = img6_sift_kps[img6_idx].pt.x;
			int img6_y = img6_sift_kps[img6_idx].pt.y;
			total_disp_x += abs(img2_x-img6_x);
			total_disp_y += abs(img2_y-img6_y);
			double disp_xy = sqrt(pow(img2_y-img6_y, 2)+pow(img2_x-img6_x, 2));
			total_disp_xy += disp_xy; 
			double gt_xy_disp = (disp2.at<uchar>(img2_x, img2_y))/float(4);
			gt_total_disp_xy += gt_xy_disp;
			rmse_err += pow(disp_xy-gt_xy_disp, 2)/best_matches.size();
		}
		rmse_err = sqrt(rmse_err);

		cout << "Deta_d, RMSE, IncorrectlyMatched, =  " << delta_d << ", " << rmse_err << ", " <<  100*(sift_matches.size() - best_matches.size())/sift_matches.size() << endl;
	}

	
	
	
	return 0;
}

