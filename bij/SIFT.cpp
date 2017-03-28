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
	//Mat img2 = imread("im2.png", IMREAD_GRAYSCALE);
	Mat img6 = imread("im6.png", IMREAD_GRAYSCALE);
	Mat img2 = imread("im6.png", IMREAD_GRAYSCALE);
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
	Ptr<Feature2D> sift_f2d = xfeatures2d::SIFT::create(num_points);

	// Detect Key Points
	vector<KeyPoint> img2_all_sift_kps;
	vector<KeyPoint> img6_all_sift_kps;
	for(int r=0; r<img2.rows; r++){
		for(int c=0; c<img2.cols; c++){
			KeyPoint temp_kp = KeyPoint(Point2f(c, r), 16, -1);
			img2_all_sift_kps.push_back(temp_kp);
			img6_all_sift_kps.push_back(temp_kp);
		}
	}

	
			
			

	// Create Descriptors for the keypoints
	Mat img2_sift_descriptors;
	Mat img6_sift_descriptors;
	sift_f2d->compute(gray2, img2_all_sift_kps, img2_sift_descriptors);
	sift_f2d->compute(gray2, img6_all_sift_kps, img6_sift_descriptors);

	// Draw the keypoints to new images and display them
	Mat draw_img2_sift_keypoints;
	Mat draw_img6_sift_keypoints;
	drawKeypoints(gray2, img2_all_sift_kps, draw_img2_sift_keypoints, Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(gray6, img6_all_sift_kps, draw_img6_sift_keypoints, Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	
	// Display the keypoints of all images
	namedWindow("IM2 SIFT KeyPoints", WINDOW_AUTOSIZE);
	imshow("IM2 SIFT KeyPoints", draw_img2_sift_keypoints);
	namedWindow("IM6 SIFT KeyPoints", WINDOW_AUTOSIZE);
	imshow("IM6 SIFT KeyPoints", draw_img6_sift_keypoints);
	cv::waitKey(0);
	destroyWindow("IM2 SIFT KeyPoints");
	destroyWindow("IM6 SIFT KeyPoints");

	// Begin Matching Keypoints
	FlannBasedMatcher flann_matcher;
	BFMatcher BruteForceMatcher = BFMatcher(NORM_L1, true);

	vector<DMatch> sift_matches;
	cout << "FLANN!" << endl;
	flann_matcher.match(img2_sift_descriptors, img6_sift_descriptors, sift_matches);
	//cout << "BruteForce!" << endl;
	//BruteForceMatcher.match(img2_sift_descriptors, img6_sift_descriptors, sift_matches);

	// Sort and filter for Filter out the matches with too great of a distance
	sort(sift_matches.begin(), sift_matches.end(), dist);
	vector<DMatch> sift_good_matches;
	for(int m=0; m<num_good_matches; m++){
			sift_good_matches.push_back(sift_matches[m]);
	}

	// Draw and display the good matches
	Mat draw_sift_matches;
	drawMatches(img2, img2_all_sift_kps, img6, img6_all_sift_kps, sift_good_matches, draw_sift_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	//drawMatches(img2, img2_sift_keypoints, img6, img6_sift_keypoints, sift_good_matches, draw_sift_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::DEFAULT);
		
	namedWindow("SIFT Matches", WINDOW_AUTOSIZE);
	imshow("SIFT Matches", draw_sift_matches);

	double dist_sum = 0; 
	for(int m=0; m<sift_matches.size(); m++){
		dist_sum += sift_matches[m].distance; 
	}
	double avg = dist_sum/sift_matches.size();

	cout << "Sum = " << dist_sum << endl;
	cout << "Avg = " << avg << endl;


	// Calculate the Disparity
	Mat new_int_2_6(img2.rows, img2.cols, DataType<float>::type);
	Mat new_disp_2_6(img2.rows, img2.cols, DataType<float>::type);
	double sum = 0; 
	for(int r=0; r<img2.rows; r++){
		for(int c=0; c<img2.cols; c++){
			int img2_val = img2.at<uchar>(r, c);
			int img6_Idx = sift_matches[r*img2.rows+c].trainIdx;
			int img6_x = img6_all_sift_kps[img6_Idx].pt.x;
			int img6_y = img6_all_sift_kps[img6_Idx].pt.y;
			int img6_val = img6.at<uchar>(img6_y, img6_x);
			double diff = pow(img2_val - img6_val, 2);
			//double diff = abs(img2_val - img6_val);
			new_int_2_6.at<float>(r,c) = diff;
			new_disp_2_6.at<float>(r,c) = sqrt(pow(r-img6_y, 2)+pow(c-img6_x, 2));
			sum += diff;
		}
	}

	cout << sum << endl;
 	Mat dst;
        //normalize(new_disp_2_6, dst, 0, 1, cv::NORM_MINMAX);
	imshow("IMG2-IMG6-DISP", new_int_2_6);
        normalize(new_int_2_6, dst, 0, 1, cv::NORM_MINMAX);
        imshow("test", dst);
	//namedWindow("IMG2-IMG6-DISP", WINDOW_AUTOSIZE);
	cv::waitKey(0);
		

	cout << sift_good_matches[0].imgIdx << endl;
	cout << sift_good_matches[0].queryIdx << endl;
	cout << sift_good_matches[0].trainIdx << endl;
			
	

	cv::waitKey(0);
	return 0;
}

