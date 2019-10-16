#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream> 
#include <stdarg.h> 
#include <limits>
#include <algorithm> 
#include <opencv2\opencv.hpp>
#include <opencv2\features2d\features2d.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/internal.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include <vector>

using namespace cv;
using namespace std;

namespace mycv
{

	struct Pattern
	{
		Point2f center;    //采样中心、关键点
		int maxRidus;	   //最大采样圆的半径
		int totalBins;     //共有多少灰度Bins
		vector<int> bins;
		vector<float> sampleRidus;         //存放采样圆半径
		vector<Rect_<float>> sampleRects;  //存放矩形采样框
	};

	class CV_EXPORTS NOF :public Feature2D/*DescriptorExtractor*/
	{
	public:
		// the size of the signature in bytes
		enum { kBytes = 32, HARRIS_SCORE = 0, FAST_SCORE = 1 };

		explicit NOF(int nfeatures = 500, int firstLevel = 0, int nlevels = 8, int patchSize = 31, int scoreType = ORB::HARRIS_SCORE,
			double scaleFactor = 1.2, int edgeThreshold = 31, bool doBinaryEncode = false);
		~NOF();

		/** returns the descriptor length in bytes */
		virtual int descriptorSize() const;

		/** returns the descriptor type */
		virtual int descriptorType() const;

		/** returns the Feature Name */
		string name(){ return "NOF"; };

	public:
		void operator()(InputArray image, InputArray mask, vector<KeyPoint>& keypoints,
			OutputArray descriptors, bool useProvidedKeypoints = false) const;
		void detectImpl(const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask = Mat()) const;
		void computeImpl(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors) const;

		//vector<Mat> imagePyramid;
		//vector<vector<KeyPoint>> allKeypoints;
		//Pattern pattern;
		int nfeatures;
		int firstLevel;
		int nlevels;
		int scoreType;
		int patchSize;
		double scaleFactor;
		int edgeThreshold;
		bool doBinaryEncode;


	};

}