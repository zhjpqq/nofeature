#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

using namespace std;
using namespace cv;

const string TransformTypes[6] = { "Rotate", "Zoom", "View", "Blur", "Noise", "Light" };

const string MatcherTypes[4] = {"BruteForce","CrossCheck","DistThreshold","FLANN"};

int RunEvaluator(string detectorName, string descriptorName, string matcherName);

//读取指定的参考帧图像
int ReadReferImage(const string fileName, Mat& referImg);

//Save evaluating results 
void SaveEvaluationResults(vector<vector<double>>& precisionResults, string dir, string featureName, string matcherName, string transformName);

int MatchByDifferentWays(Ptr<DescriptorMatcher> matcher, const Mat& referDescs, const Mat& transferDescs,
	vector<vector<DMatch>>& matchesR2T, int matchType = 0);


//evaluate matching precision on generated datasets under transformations of rotation
int PrecisionEvaluationUnderRotation(Ptr<FeatureDetector> detector, Ptr<DescriptorExtractor> extractor, Ptr<DescriptorMatcher> matcher, int matcherType);


//evaluate matching precision on generated datasets under transformations of zoom
int PrecisionEvaluationUnderZoom(Ptr<FeatureDetector> detector, Ptr<DescriptorExtractor> extractor, Ptr<DescriptorMatcher> matcher, int matcherType);


//evaluate matching precision on generated datasets under transformations of viewpoint
int PrecisionEvaluationUnderViewpoint(Ptr<FeatureDetector> detector, Ptr<DescriptorExtractor> extractor, Ptr<DescriptorMatcher> matcher, int matcherType);


//evaluate matching precision on generated datasets under transformations of gaussian blurring
int PrecisionEvaluationUnderGaussianBluring(Ptr<FeatureDetector> detector, Ptr<DescriptorExtractor> extractor, Ptr<DescriptorMatcher> matcher, int matcherType);


//evaluate matching precision on generated datasets under transformations of white noise
int PrecisionEvaluationUnderWhiteNoises(Ptr<FeatureDetector> detector, Ptr<DescriptorExtractor> extractor, Ptr<DescriptorMatcher> matcher, int matcherType);


//evaluate matching precision on generated datasets under transformations of constract lightness
int PrecisionEvaluationUnderConstractLightness(Ptr<FeatureDetector> detector, Ptr<DescriptorExtractor> extractor, Ptr<DescriptorMatcher> matcher, int matcherType);
