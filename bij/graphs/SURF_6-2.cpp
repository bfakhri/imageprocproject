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
		

		// Begin Matching Keypoints
		FlannBasedMatcher flann_matcher;
		BFMatcher BruteForceMatcher = BFMatcher(NORM_L1, true);

		vector<DMatch> surf_matches;
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


		double dist_sum = 0; 
		for(int m=0; m<surf_matches.size(); m++){
			dist_sum += surf_matches[m].distance; 
		}
		double avg = dist_sum/surf_matches.size();


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

		cout << "Deta_d, RMSE, IncorrectlyMatched, =  " << delta_d << ", " << rmse_err << ", " <<  100*(surf_matches.size() - best_matches.size())/surf_matches.size() << endl;
	}

	
	
	
	return 0;
}

