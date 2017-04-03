EEE 508 | Project 01 | Team: Bijan Fakhri & Anirudh Som
-------------------------------------------------------

Feature Detector/Descriptors
---------------------------
- Scale Invariant Feature Transform (SIFT)
- Speeded Up Robust Features (SURF)

Computer Vision Application
---------------------------
Stereo Correspondence Algorithms

Zipped Folder Contents
----------------------
I.   README.txt : This readme file.
II.  Report.pdf : Report for this Project.
III. Code : Folder containing the executable code for the Project.
IV.  Input : Folder containing the input data.
V. 	 Output : Folder containing the output data after execution of code.
VI.  Project1_GradingSheet.pdf : Grading sheet with team member names.

Code Folder Contents
--------------------
For each of the below cpp files, we calculate the disparity between the obtained
matched keypoints using the FLANN matching algorithm. We also show the error 
evaluations in terms of root mean squre error (RMSE) and the percentage of 
incorrectly matched pixels, as described in [1]. For each of these files we also
provide their respective executable files.

I. 	 SIFT_2-6.cpp : To extract SIFT keypoints and descriptors from the two 
stereo images. Using FLANN matching algorithm to find all and best matches 
from im2.png to im6.png stereo images.

II.	 SIFT_6-2.cpp :To extract SIFT keypoints and descriptors from the two 
stereo images. Using FLANN matching algorithm to find all and best matches 
from im6.png to im2.png stereo images.

III. SURF_2-6.cpp :To extract SURF keypoints and descriptors from the two 
stereo images. Using FLANN matching algorithm to find all and best matches 
from im2.png to im6.png stereo images.

IV.  SURF_6-2.cpp :To extract SURF keypoints and descriptors from the two 
stereo images. Using FLANN matching algorithm to find all and best matches 
from im6.png to im2.png stereo images.


Input Folder Contents
---------------------
- 2003 Stereo datasets with ground truth - cones, teddy
- Description: These datasets were created by Daniel Scharstein, Alexander 
Vandenberg-Rodes, and Rick Szeliski [2]. They consist of high-resolution 
stereo sequences with complex geometry and pixel-accurate ground-truth 
disparity data. The ground-truth disparities are acquired using a novel 
technique that employs structured lighting and does not require the calibration 
of the light projectors.

Output Folder Contents
----------------------
Each of the following output files were extracted for both the teddy and cones 
stereo image datasets.

I.	  img2_sift_kps.jpg : SIFT keypoints and descriptors extracted for im2.png image.

II.   img6_sift_kps.jpg : SIFT keypoints and descriptors extracted for im6.png image.

III.  img2_surf_kps.jpg : SURF keypoints and descriptors extracted for im2.png image.

IV.   img6_surf_kps.jpg : SURF keypoints and descriptors extracted for im6.png image.

V.    img2-img6_all_sift_matches.jpg : All matches with img2.png as reference after 
extracting SIFT features.

VI.   img6-img2_all_sift_matches.jpg : All matches with img6.png as reference after 
extracting SIFT features.

VII.  img2-img6_all_surf_matches.jpg : All matches with img2.png as reference after 
extracting SURF features.

VIIII.img6-img2_all_surf_matches.jpg : All matches with img6.png as reference after 
extracting SURF features.

IX.   img2-img6_best_sift_matches.jpg : Best matches with img2.png as reference after 
extracting SIFT features.

X.    img6-img2_best_sift_matches.jpg : Best matches with img6.png as reference after 
extracting SIFT features.

XI.   img2-img6_best_surf_matches.jpg : Best matches with img2.png as reference after 
extracting SURF features.

XII.  img6-img2_best_surf_matches.jpg : Best matches with img6.png as reference after 
extracting SURF features.


Compiling and Running Instructions
----------------------------------
1)


References
----------

[1] D. Scharstein and R. Szeliski, “A Taxonomy and Evaluation of Dense Two-Frame
Stereo Correspondence Algorithms,” International Journal of Computer Vision, 2002.

[2] Scharstein, Daniel, and Richard Szeliski. "High-accuracy stereo depth maps using 
structured light." Computer Vision and Pattern Recognition, 2003. Proceedings. 2003 
IEEE Computer Society Conference on. Vol. 1. IEEE, 2003.
