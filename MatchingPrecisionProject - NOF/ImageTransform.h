#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

#include <opencv2/opencv.hpp> 

using namespace std;
using namespace cv;

//add rotation to the src image, return the rotated image and the corresponding ground-truth matrix
void AddRotationTransformation(const Mat& src, double rotAngle, Mat& dst, Mat& homo);

//add zoom to the src image, return the zoomed image and the corresponding ground-truth matrix
void AddZoomTransformation(const Mat& src, double zoomScale, Mat& dst, Mat& homo);

//add Viewpoint changes to the src image, return the transformed image and the corresponding ground-truth matrix
void AddViewpointTransformation(const Mat& src, double viewAngle, Mat& dst, Mat& H);

//add Bluring changes to the src image, return the blured image and the corresponding ground-truth matrix
void AddGaussianBluringTransformation(const Mat& src, double blurScale, Mat& dst, Mat& homo);

//add white noises to the src image, return the noised image and the corresponding ground-truth matrix
void AddWhiteNoisesTransformation(const Mat& src, double noiseStrength, Mat& dst, Mat& homo);

//add contrast and lightness to the src image, return the transformed image and the corresponding ground-truth matrix
void AddConstractLightnessTransformation(const Mat& src, double lightChange, Mat& dst, Mat& homo);