#include "Configure.h"
#include "ImageTransform.h"
#include "PrecisionEvaluator.h"
#include "NOF.h"

int RunEvaluator(string detectorName,string descriptorName, string matcherName)
{
	double clock = (double)cvGetTickCount();

	cv::initModule_features2d();
	cv::initModule_nonfree(); 

	//feature2D detectcor method: ORB, BRISK, FAST, SIFT, SURF, etc
	Ptr<FeatureDetector> detector = new mycv::NOF(800);
	//Ptr<FeatureDetector> detector = FeatureDetector::create("BRISK"); //detector->set("nFeatures", 800);
	
	//Feature2D descriptor method: ORB,BRISK,FREAK,SIFT,SURF	
	Ptr<DescriptorExtractor> extractor = new mycv::NOF();
	//Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("BRISK");
	
	//Feature2D matcher methods: BruteForce,FLANN
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(matcherName);

	int matcherType = 1;

	//PrecisionEvaluationUnderRotation(detector, extractor, matcher, matcherType);

	PrecisionEvaluationUnderZoom(detector, extractor, matcher, matcherType);

	//PrecisionEvaluationUnderViewpoint(detector, extractor, matcher, matcherType);

	//PrecisionEvaluationUnderGaussianBluring(detector, extractor, matcher, matcherType);

	//PrecisionEvaluationUnderWhiteNoises(detector, extractor, matcher, matcherType);

	//PrecisionEvaluationUnderConstractLightness(detector, extractor, matcher, matcherType);

	clock = (double)cvGetTickCount() - clock;
	printf("run time = %gms\n", clock / (cvGetTickFrequency() * 1000));
	printf("run time = %gs\n", clock / (cvGetTickFrequency() * 1000000));

	char ch = getchar();
	waitKey(-1);
	return 0;
}

//读取指定的参考帧图像
int ReadReferImage(const string fileName, Mat& referImg)
{
	referImg = imread(fileName, 1);
	if (referImg.empty())
	{
		cout << "fileName: " << fileName << " cannot open!" << endl;
		return getchar();
	}
	else
		return 0;
}

//Save evaluating results 
void SaveEvaluationResults(vector<vector<double>>& precisionResults,string dir, string featureName, string matcherName, string transformName)
{
	stringstream fileName; fileName <<dir<< matcherName<<"_" <<transformName <<"_"<<featureName<<".csv";
	FILE* fp; fopen_s(&fp, fileName.str().c_str(), "w");
	const int expand = 10000;
	for (int imgIdx = 0; imgIdx < DATASETS_COUNT; imgIdx++)
	{
		for (int t = 0; t < (int)precisionResults[imgIdx].size(); t++)
		{
			fprintf(fp, "%lf ", precisionResults[imgIdx][t] * expand);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}

void DetectKeypointsAndComputeDescriptors(const Mat& referImage,const Mat& transferImage,Ptr<FeatureDetector>& detector, Ptr<DescriptorExtractor>& extractor,
	vector<KeyPoint>& referKeypoints,vector<KeyPoint>& transferKeypoints, Mat& referDescriptors, Mat& transferDescriptors)
{
	//Detect keypoints on refer image, transformed image
	detector->detect(referImage, referKeypoints, Mat());
	detector->detect(transferImage, transferKeypoints, Mat());
	if (referKeypoints.size()>800)
		KeyPointsFilter::retainBest(referKeypoints, 800);
	if (transferKeypoints.size()>800)
		KeyPointsFilter::retainBest(transferKeypoints, 800);

	//extract feature vectors on refer image,, transformed image
	extractor->compute(referImage, referKeypoints, referDescriptors);
	extractor->compute(transferImage, transferKeypoints, transferDescriptors);
}

int MatchByDifferentWays(Ptr<DescriptorMatcher> matcher,const Mat& referDescs,const Mat& transferDescs,
	vector<vector<DMatch>>& matchesR2T,int matchType/*=0*/)
{
	matchesR2T.clear();
	int matchesSize(0); //the number of matched pairs

	if (matchType == 0)//brute force matcher type
	{
		int kNN = 1; //find the best nearest for each query one
		matcher->knnMatch(referDescs, transferDescs, matchesR2T, kNN);
		matchesSize = matchesR2T.size()*kNN;
		
	}
	if (matchType == 1) //cross-check matcher type
	{ 
		int knn = 3;
		vector<vector<DMatch>> matchesRT, matchesTR;
		//match from reference image to transformed image
		matcher->knnMatch(referDescs, transferDescs, matchesRT, knn);
		//match from transformed image to reference image
		matcher->knnMatch(transferDescs,referDescs, matchesTR, knn);
		 
		//cross-check 交叉验证
		for (size_t rt0 = 0; rt0 < matchesRT.size();rt0++)  //遍历全部RT关键点
		{
			vector<DMatch> tempMatches(0);  //保存单个关键点正确匹配对

			for (size_t rt1 = 0; rt1 < matchesRT[rt0].size(); rt1++)  //遍历当前RT关键点的knn=3个描述符
			{
				DMatch& rt_dm = matchesRT[rt0][rt1];
				
				for (size_t tr1 = 0; tr1 < matchesTR[rt_dm.trainIdx].size(); tr1++)
				{
					DMatch& tr_dm = matchesTR[rt_dm.trainIdx][tr1];

					if ((rt_dm.queryIdx == tr_dm.trainIdx) && (rt_dm.trainIdx == tr_dm.queryIdx))
						tempMatches.push_back(rt_dm);
				}
				
			}
			if (!tempMatches.empty())
			{
				matchesR2T.push_back(tempMatches);
				matchesSize += (int)tempMatches.size();
			}				
		}
	}
	if (matchType == 2) //Distance threshold matcher type
	{
		float thresh = 0.7;
		int kNN = 2; //find the two best nearests for each query one
		vector<vector<DMatch>> matchesRT;
		matcher->knnMatch(referDescs, transferDescs, matchesRT, kNN);
		
		vector<DMatch> tempMatches(0);

		for (size_t k = 0; k < matchesRT.size();k++)
		{
			DMatch& dm0 = matchesRT[k][0];
			DMatch& dm1 = matchesRT[k][1];
			if (dm0.distance / dm1.distance < thresh)
				tempMatches.push_back(dm0);
		}
		matchesSize = tempMatches.size();
		matchesR2T.resize(matchesSize);
		for (size_t k = 0; k < matchesR2T.size(); k++)
		{
			matchesR2T[k].push_back(tempMatches[k]);
		}
	}

	return matchesSize;
}

void ObtainCorrectMatchesUsingGroundTruth(Mat& HomographyMat, vector<KeyPoint>& referKeypoints, vector<KeyPoint>& transferKeypoints, 
										vector<vector<DMatch>>& MatchesR2T,vector<DMatch>& CorrectMatches,double epsilon)
{
	//convert KeyPoint class to Point2f type
	vector<Point2f> referPoints, transferPoints;
	KeyPoint::convert(referKeypoints, referPoints);
	KeyPoint::convert(transferKeypoints, transferPoints);

	//transfer the referPoints to the transformed image using ground-truth matrix
	vector<Point2f> GroundTruthPoints;
	cv::perspectiveTransform(referPoints, GroundTruthPoints, HomographyMat);

	//select the correct matches from initial matches acccording to GroundTruthPoints  
	for (size_t k = 0; k < MatchesR2T.size(); k++){
		for (size_t m = 0; m < MatchesR2T[k].size(); m++)
		{
			DMatch& dm = MatchesR2T[k][m];
			if (norm(GroundTruthPoints[dm.queryIdx] - transferPoints[dm.trainIdx]) < epsilon) //判断2个点的欧氏距离是否小于epislon
				CorrectMatches.push_back(dm);
		}
	}	
}


//evaluate matching precision on generated datasets under transformations of rotation
int PrecisionEvaluationUnderRotation(Ptr<FeatureDetector> detector, Ptr<DescriptorExtractor> extractor, Ptr<DescriptorMatcher> matcher, int matcherType)
{ 
	//rotate angle sets : counter clock-wise
	vector<double> RotateAngleSet = { 20, 40, 60, 80, 100, 120, 140, 160,180 };

	vector<double>& TransformScales = RotateAngleSet;
	int transferCount = (int)TransformScales.size();
	
	int transformType = 0;	
	 
	//Record the matching precision on each reference image under increasing transforming scale
	vector<vector<double>> MatchesPrecision(DATASETS_COUNT);

	//loops at reference image datasets
	for (int imgIdx = 0; imgIdx < DATASETS_COUNT; imgIdx++)
	{
		stringstream fileName; fileName << DirName << DATASET_NAMES[imgIdx] << ".ppm";
		
		//read current reference image
		Mat ReferImage;  
		ReadReferImage(fileName.str(), ReferImage);

		//to assure that the rotated image can be seen wholely
		copyMakeBorder(ReferImage, ReferImage, 300, 300, 300, 300, BORDER_CONSTANT, Scalar::all(0));

		//show the reference image
		//namedWindow("Reference Image", 0);
		//imshow("Reference Image", ReferImage);
		//cv::waitKey(100);

		MatchesPrecision[imgIdx].resize(transferCount);

		//loops at different rotation angle
		for (int transIdx = 0; transIdx < transferCount; transIdx++)
		{
			cout << "refer image : " << DATASET_NAMES[imgIdx] <<
				"  , rotate angle: " << TransformScales[transIdx] << endl;

			Mat TransferImage; //the transformed image
			Mat HomographyMat; //the corresponding homography matrix

			//add rotation to reference image
			AddRotationTransformation(ReferImage, TransformScales[transIdx], TransferImage, HomographyMat);

			//show the transformed image
			//namedWindow("Transform Image", 0);
			//imshow("Transform Image", TransferImage);
			//cv::waitKey(20); 

			//detect keypoints and compute descriptors for them on both reference image and transformed image		
			vector<KeyPoint> referKeypoints, transferKeypoints; 
			Mat referDescriptors, transferDescriptors;			
			DetectKeypointsAndComputeDescriptors(ReferImage, TransferImage, detector, extractor, 
				referKeypoints, transferKeypoints, referDescriptors, transferDescriptors);

			//matches from refer to transfer keypoints using a specified matcher type
			vector<vector<DMatch>> MatchesR2T;
			int matchesSize = 
				MatchByDifferentWays(matcher, referDescriptors, transferDescriptors, MatchesR2T, matcherType);
			
			//Draw initial matches result
			/*Mat initResults;
			drawMatches(ReferImage, referKeypoints, TransferImage, transferKeypoints, MatchesR2T,initResults,Scalar(255,0,0));
			namedWindow("Initial Window", 0); imshow("Initial Window", initResults);
			cv::waitKey(20);*/

			//Obtain correct matches using GroundTruth matrix
			vector<DMatch> CorrectMatches(0); double epsilon(2.0);
			ObtainCorrectMatchesUsingGroundTruth(HomographyMat, referKeypoints, transferKeypoints,
				MatchesR2T, CorrectMatches, epsilon);

			//Draw correct matches result
			Mat correctResults;
			drawMatches(ReferImage, referKeypoints, TransferImage, transferKeypoints, CorrectMatches, correctResults, Scalar(255, 0, 0));
			namedWindow("Correct Window", 0); imshow("Correct Window", correctResults);
			cv::waitKey(20);

			//compute matching precision
			double precision = (float)CorrectMatches.size() / ((float)matchesSize);
			cout << "matching precision: " << precision << endl;
			MatchesPrecision[imgIdx][transIdx] = precision;
		}
	}
	
	cout << endl << "保存结果到文件....." << endl;
	string Feature2dName =   "NOF"; // extractor->name();
	cout << Feature2dName;
	SaveEvaluationResults(MatchesPrecision, ".\\evaluate_results\\", Feature2dName, MatcherTypes[matcherType], TransformTypes[0]);
	cout << "保存结果完毕." << endl;
	return cv::waitKey(1000);
}

//evaluate matching precision on generated datasets under transformations of zoom
int PrecisionEvaluationUnderZoom(Ptr<FeatureDetector> detector, Ptr<DescriptorExtractor> extractor, Ptr<DescriptorMatcher> matcher, int matcherType)
{ 
	//zoom scale sets  
	vector<double> ZoomScaleSet = { 2.4, 2.0, 1.6, 1.2, 0.8, 0.4};
	
	vector<double>& TransformScales = ZoomScaleSet;
	int transferCount = (int)TransformScales.size();

	int transformType = 1; 

	//Record the matching precision on each reference image under increasing transforming scale
	vector<vector<double>> MatchesPrecision(DATASETS_COUNT);

	//loops at reference image datasets
	for (int imgIdx = 0; imgIdx < DATASETS_COUNT; imgIdx++)
	{
		stringstream fileName; fileName << DirName << DATASET_NAMES[imgIdx] << ".ppm";
		
		//read current reference image
		Mat ReferImage; 
		ReadReferImage(fileName.str(), ReferImage);

		//show the reference image
		//namedWindow("Reference Image", 0);
		//imshow("Reference Image", ReferImage);
		//cv::waitKey(100);

		MatchesPrecision[imgIdx].resize(transferCount);

		//loops at different zoom scale
		for (int transIdx = 0; transIdx < transferCount; transIdx++)
		{
			cout << "refer image : " << DATASET_NAMES[imgIdx] <<
				"  , zoom scale: " << TransformScales[transIdx] << endl;

			Mat TransferImage; //the transformed image
			Mat HomographyMat; //the corresponding homography matrix

			//add zoom to reference image
			AddZoomTransformation(ReferImage, TransformScales[transIdx], TransferImage, HomographyMat);

			//show the transformed image
			//namedWindow("Transform Image", 0);
			//imshow("Transform Image", TransferImage);
			//cv::waitKey(20);

			//detect keypoints and compute descriptors for them on both reference image and transformed image
			vector<KeyPoint> referKeypoints, transferKeypoints;
			Mat referDescriptors, transferDescriptors;
			DetectKeypointsAndComputeDescriptors(ReferImage, TransferImage, detector, extractor,
				referKeypoints, transferKeypoints, referDescriptors, transferDescriptors);

			vector<vector<DMatch>> MatchesR2T;//matches from refer to transfer keypoints 
			int matchesSize =
				MatchByDifferentWays(matcher, referDescriptors, transferDescriptors, MatchesR2T, matcherType);

			//Draw initial matches result
			/*Mat initResults;
			drawMatches(ReferImage, referKeypoints, TransferImage, transferKeypoints, MatchesR2T,initResults,Scalar(255,0,0));
			namedWindow("Initial Window", 0); imshow("Initial Window", initResults);
			cv::waitKey(20);*/

			//Obtain correct matches using GroundTruth matrix
			vector<DMatch> CorrectMatches(0); double epsilon(2.0);
			ObtainCorrectMatchesUsingGroundTruth(HomographyMat, referKeypoints, transferKeypoints,
				MatchesR2T, CorrectMatches, epsilon);

			//Draw correct matches result
			Mat correctResults;
			drawMatches(ReferImage, referKeypoints, TransferImage, transferKeypoints, CorrectMatches, correctResults, Scalar(255, 0, 0));
			namedWindow("Correct Window", 0); imshow("Correct Window", correctResults);
			cv::waitKey(20);

			//compute matching precision
			double precision = (float)CorrectMatches.size() / ((float)matchesSize);
			cout << "matching precision: " << precision << endl;

			MatchesPrecision[imgIdx][transIdx] = precision;
		}
	}
	cout << endl << "保存结果到文件....." << endl;
	string Feature2dName =   "NOF"; // extractor->name();
	SaveEvaluationResults(MatchesPrecision, ".\\evaluate_results\\", Feature2dName, MatcherTypes[matcherType], TransformTypes[transformType]);
	cout << "保存结果完毕." << endl;
	return waitKey(1000);
}

//evaluate matching precision on generated datasets under transformations of viewpoint
int PrecisionEvaluationUnderViewpoint(Ptr<FeatureDetector> detector, Ptr<DescriptorExtractor> extractor, Ptr<DescriptorMatcher> matcher, int matcherType)
{ 
	//View angle sets  
	vector<double> ViewAngleSet = { 1, 2, 3, 4, 5 };
	
	vector<double>& TransformScales = ViewAngleSet;
	int transferCount = (int)TransformScales.size();

	int transformType = 2; 

	//Record the matching precision on each reference image under transforms
	vector<vector<double>> MatchesPrecision(DATASETS_COUNT);

	//loops at reference image datasets
	for (int imgIdx = 0; imgIdx < DATASETS_COUNT; imgIdx++)
	{
		stringstream fileName; fileName << DirName << DATASET_NAMES[imgIdx] << ".ppm";
		
		//read current reference image
		Mat ReferImage;  
		ReadReferImage(fileName.str(), ReferImage);

		//to assure that the transformed image can be seen wholely
		copyMakeBorder(ReferImage, ReferImage, 50, 400, 50, 50, BORDER_CONSTANT, Scalar::all(0));

		//show the reference image
		//namedWindow("Reference Image", 0);
		//imshow("Reference Image", ReferImage);
		//cv::waitKey(100);

		MatchesPrecision[imgIdx].resize(transferCount);

		//loops at different view angles
		for (int transIdx = 0; transIdx < transferCount; transIdx++)
		{
			cout << "refer image : " << DATASET_NAMES[imgIdx] <<
				"  , view angle: " << TransformScales[transIdx] << endl;

			Mat TransferImage; //the transformed image
			Mat HomographyMat; //the corresponding homography matrix

			//add view changes to reference image
			AddViewpointTransformation(ReferImage, TransformScales[transIdx], TransferImage, HomographyMat);

			//show the transformed image
			//namedWindow("Transform Image", 0);
			//imshow("Transform Image", TransferImage);
			//cv::waitKey(20);

			//detect keypoints and compute descriptors for them on both reference image and transformed image
			vector<KeyPoint> referKeypoints, transferKeypoints;
			Mat referDescriptors, transferDescriptors;
			DetectKeypointsAndComputeDescriptors(ReferImage, TransferImage, detector, extractor,
				referKeypoints, transferKeypoints, referDescriptors, transferDescriptors);

			vector<vector<DMatch>> MatchesR2T;//matches from refer to transfer keypoints 
			int matchesSize =
				MatchByDifferentWays(matcher, referDescriptors, transferDescriptors, MatchesR2T, matcherType);

			//Draw initial matches result
			/*Mat initResults;
			drawMatches(ReferImage, referKeypoints, TransferImage, transferKeypoints, MatchesR2T,initResults,Scalar(255,0,0));
			namedWindow("Initial Window", 0); imshow("Initial Window", initResults);
			cv::waitKey(20);*/

			//Obtain correct matches using GroundTruth matrix
			vector<DMatch> CorrectMatches(0); double epsilon(2.0);
			ObtainCorrectMatchesUsingGroundTruth(HomographyMat, referKeypoints, transferKeypoints,
				MatchesR2T, CorrectMatches, epsilon);

			//Draw correct matches result
			Mat correctResults;
			drawMatches(ReferImage, referKeypoints, TransferImage, transferKeypoints, CorrectMatches, correctResults, Scalar(255, 0, 0));
			namedWindow("Correct Window", 0); imshow("Correct Window", correctResults);
			cv::waitKey(20);

			//compute matching precision
			double precision = (float)CorrectMatches.size() / ((float)matchesSize);
			cout << "matching precision: " << precision << endl;

			MatchesPrecision[imgIdx][transIdx] = precision;
		}
	}
	cout << endl << "保存结果到文件....." << endl;
	string Feature2dName =   "NOF"; // extractor->name();
	SaveEvaluationResults(MatchesPrecision, ".\\evaluate_results\\", Feature2dName, MatcherTypes[matcherType], TransformTypes[transformType]);
	cout << "保存结果完毕." << endl;
	return waitKey(1000);
}

//evaluate matching precision on generated datasets under transformations of gaussian blurring
int PrecisionEvaluationUnderGaussianBluring(Ptr<FeatureDetector> detector, Ptr<DescriptorExtractor> extractor, Ptr<DescriptorMatcher> matcher, int matcherType)
{ 
	//gaussian scale sets  
	vector<double> GaussianScaleSet = { 1.0, 3.0, 5.0, 7.0, 9.0, 11.0 };
	vector<double>& TransformScales = GaussianScaleSet;
	int transferCount = (int)TransformScales.size();

	int transformType = 3; 

	//Record the matching precision on each reference image under transforms
	vector<vector<double>> MatchesPrecision(DATASETS_COUNT);

	//loops at reference image datasets
	for (int imgIdx = 0; imgIdx < DATASETS_COUNT; imgIdx++)
	{
		stringstream fileName; fileName << DirName << DATASET_NAMES[imgIdx] << ".ppm";
		Mat ReferImage;  //current reference image
		if (ReadReferImage(fileName.str(), ReferImage)){
			cerr << "cannot read image..." << endl;
		}

		//show the reference image
		//namedWindow("Reference Image", 0);
		//imshow("Reference Image", ReferImage);
		//cv::waitKey(100);

		MatchesPrecision[imgIdx].resize(transferCount);

		//loops at different gaussian bluring scale
		for (int transIdx = 0; transIdx < transferCount; transIdx++)
		{
			cout << "refer image : " << DATASET_NAMES[imgIdx] <<
				"  , gaussian scale: " << TransformScales[transIdx] << endl;

			Mat TransferImage; //the transformed image
			Mat HomographyMat; //the corresponding homography matrix

			//add gaussian blurings to reference image
			AddGaussianBluringTransformation(ReferImage, TransformScales[transIdx], TransferImage, HomographyMat);

			//show the transformed image
			//namedWindow("Transform Image", 0);
			//imshow("Transform Image", TransferImage);
			//cv::waitKey(20);

			//detect keypoints and compute descriptors for them on both reference image and transformed image
			vector<KeyPoint> referKeypoints, transferKeypoints;
			Mat referDescriptors, transferDescriptors;
			DetectKeypointsAndComputeDescriptors(ReferImage, TransferImage, detector, extractor,
				referKeypoints, transferKeypoints, referDescriptors, transferDescriptors);

			vector<vector<DMatch>> MatchesR2T;//matches from refer to transfer keypoints 
			int matchesSize =
				MatchByDifferentWays(matcher, referDescriptors, transferDescriptors, MatchesR2T, matcherType);

			//Draw initial matches result
			/*Mat initResults;
			drawMatches(ReferImage, referKeypoints, TransferImage, transferKeypoints, MatchesR2T,initResults,Scalar(255,0,0));
			namedWindow("Initial Window", 0); imshow("Initial Window", initResults);
			cv::waitKey(20);*/

			//Obtain correct matches using GroundTruth matrix
			vector<DMatch> CorrectMatches(0); double epsilon(2.0);
			ObtainCorrectMatchesUsingGroundTruth(HomographyMat, referKeypoints, transferKeypoints,
				MatchesR2T, CorrectMatches, epsilon);

			//Draw correct matches result
			Mat correctResults;
			drawMatches(ReferImage, referKeypoints, TransferImage, transferKeypoints, CorrectMatches, correctResults, Scalar(255, 0, 0));
			namedWindow("Correct Window", 0); imshow("Correct Window", correctResults);
			cv::waitKey(20);

			//compute matching precision
			double precision = (float)CorrectMatches.size() / ((float)matchesSize);
			cout << "matching precision: " << precision << endl;

			MatchesPrecision[imgIdx][transIdx] = precision;
		}
	}
	cout << endl << "保存结果到文件....." << endl;
	string Feature2dName =   "NOF"; // extractor->name();
	SaveEvaluationResults(MatchesPrecision, ".\\evaluate_results\\", Feature2dName, MatcherTypes[matcherType], TransformTypes[transformType]);
	cout << "保存结果完毕." << endl;
	return waitKey(1000);
}

//evaluate matching precision on generated datasets under transformations of white noise
int PrecisionEvaluationUnderWhiteNoises(Ptr<FeatureDetector> detector, Ptr<DescriptorExtractor> extractor, Ptr<DescriptorMatcher> matcher, int matcherType)
{ 
	//noise count sets  
	vector<double> NoiseCountSet = { 0.01, 0.03, 0.05, 0.07, 0.09 };
	vector<double>& TransformScales = NoiseCountSet;
	int transferCount = (int)TransformScales.size();

	int transformType = 4; 

	//Record the matching precision on each reference image under transforms
	vector<vector<double>> MatchesPrecision(DATASETS_COUNT);

	//loops at reference image datasets
	for (int imgIdx = 0; imgIdx < DATASETS_COUNT; imgIdx++)
	{
		stringstream fileName; fileName << DirName << DATASET_NAMES[imgIdx] << ".ppm";
		Mat ReferImage;  //current reference image
		if (ReadReferImage(fileName.str(), ReferImage)){
			cerr << "cannot read image..." << endl;
		}

		//show the reference image
		//namedWindow("Reference Image", 0);
		//imshow("Reference Image", ReferImage);
		//cv::waitKey(100);

		MatchesPrecision[imgIdx].resize(transferCount);

		//loops at different noise strength
		for (int transIdx = 0; transIdx < transferCount; transIdx++)
		{
			cout << "refer image : " << DATASET_NAMES[imgIdx] <<
				"  , noise count: " << TransformScales[transIdx] << endl;

			Mat TransferImage; //the transformed image
			Mat HomographyMat; //the corresponding homography matrix

			//add white noises to reference image
			AddWhiteNoisesTransformation(ReferImage, TransformScales[transIdx], TransferImage, HomographyMat);

			//show the transformed image
			//namedWindow("Transform Image", 0);
			//imshow("Transform Image", TransferImage);
			//cv::waitKey(20);

			//detect keypoints and compute descriptors for them on both reference image and transformed image
			vector<KeyPoint> referKeypoints, transferKeypoints;
			Mat referDescriptors, transferDescriptors;
			DetectKeypointsAndComputeDescriptors(ReferImage, TransferImage, detector, extractor,
				referKeypoints, transferKeypoints, referDescriptors, transferDescriptors);

			vector<vector<DMatch>> MatchesR2T;//matches from refer to transfer keypoints 
			int matchesSize =
				MatchByDifferentWays(matcher, referDescriptors, transferDescriptors, MatchesR2T, matcherType);

			//Draw initial matches result
			/*Mat initResults;
			drawMatches(ReferImage, referKeypoints, TransferImage, transferKeypoints, MatchesR2T,initResults,Scalar(255,0,0));
			namedWindow("Initial Window", 0); imshow("Initial Window", initResults);
			cv::waitKey(20);*/

			//Obtain correct matches using GroundTruth matrix
			vector<DMatch> CorrectMatches(0); double epsilon(2.0);
			ObtainCorrectMatchesUsingGroundTruth(HomographyMat, referKeypoints, transferKeypoints,
				MatchesR2T, CorrectMatches, epsilon);

			//Draw correct matches result
			Mat correctResults;
			drawMatches(ReferImage, referKeypoints, TransferImage, transferKeypoints, CorrectMatches, correctResults, Scalar(255, 0, 0));
			namedWindow("Correct Window", 0); imshow("Correct Window", correctResults);
			cv::waitKey(20);

			//compute matching precision
			double precision = (float)CorrectMatches.size() / ((float)matchesSize);
			cout << "matching precision: " << precision << endl;

			MatchesPrecision[imgIdx][transIdx] = precision;
		}
	}
	cout << endl << "保存结果到文件....." << endl;
	string Feature2dName =   "NOF"; // extractor->name();
	SaveEvaluationResults(MatchesPrecision, ".\\evaluate_results\\", Feature2dName, MatcherTypes[matcherType], TransformTypes[transformType]);
	cout << "保存结果完毕." << endl;
	return waitKey(1000);
}

//evaluate matching precision on generated datasets under transformations of constract lightness
int PrecisionEvaluationUnderConstractLightness(Ptr<FeatureDetector> detector, Ptr<DescriptorExtractor> extractor, Ptr<DescriptorMatcher> matcher, int matcherType)
{ 
	//lightness change sets  
	vector<double> LightnessChangeSet = { -25, -10, 5, 20, 35 };
	vector<double>& TransformScales = LightnessChangeSet;
	int transferCount = (int)TransformScales.size();

	int transformType = 5; 

	//Record the matching precision on each reference image under transforms
	vector<vector<double>> MatchesPrecision(DATASETS_COUNT);

	//loops at reference image datasets
	for (int imgIdx = 0; imgIdx < DATASETS_COUNT; imgIdx++)
	{
		stringstream fileName; fileName << DirName << DATASET_NAMES[imgIdx] << ".ppm";
		Mat ReferImage;  //current reference image
		if (ReadReferImage(fileName.str(), ReferImage)){
			cerr << "cannot read image..." << endl;
		}

		////show the reference image
		//namedWindow("Reference Image", 0);
		//imshow("Reference Image", ReferImage);
		//cv::waitKey(100);

		MatchesPrecision[imgIdx].resize(transferCount);

		//loops at different noise strength
		for (int transIdx = 0; transIdx < transferCount; transIdx++)
		{
			cout << "refer image : " << DATASET_NAMES[imgIdx] <<
				"  , lightness changes: " << TransformScales[transIdx] << endl;

			Mat TransferImage; //the transformed image
			Mat HomographyMat; //the corresponding homography matrix

			//add lightness changes to reference image
			AddConstractLightnessTransformation(ReferImage, TransformScales[transIdx], TransferImage, HomographyMat);

			////show the transformed image
			//namedWindow("Transform Image", 0);
			//imshow("Transform Image", TransferImage);
			//cv::waitKey(20);

			//detect keypoints and compute descriptors for them on both reference image and transformed image
			vector<KeyPoint> referKeypoints, transferKeypoints;
			Mat referDescriptors, transferDescriptors;
			DetectKeypointsAndComputeDescriptors(ReferImage, TransferImage, detector, extractor,
				referKeypoints, transferKeypoints, referDescriptors, transferDescriptors);

			vector<vector<DMatch>> MatchesR2T;//matches from refer to transfer keypoints 
			int matchesSize =
				MatchByDifferentWays(matcher, referDescriptors, transferDescriptors, MatchesR2T, matcherType);

			//Draw initial matches result
			/*Mat initResults;
			drawMatches(ReferImage, referKeypoints, TransferImage, transferKeypoints, MatchesR2T,initResults,Scalar(255,0,0));
			namedWindow("Initial Window", 0); imshow("Initial Window", initResults);
			cv::waitKey(20);*/

			//Obtain correct matches using GroundTruth matrix
			vector<DMatch> CorrectMatches(0); double epsilon(2.0);
			ObtainCorrectMatchesUsingGroundTruth(HomographyMat, referKeypoints, transferKeypoints,
				MatchesR2T, CorrectMatches, epsilon);

			//Draw correct matches result
			Mat correctResults;
			drawMatches(ReferImage, referKeypoints, TransferImage, transferKeypoints, CorrectMatches, correctResults, Scalar(255, 0, 0));
			namedWindow("Correct Window", 0); imshow("Correct Window", correctResults);
			cv::waitKey(20);

			//compute matching precision
			double precision = (float)CorrectMatches.size() / ((float)matchesSize);
			cout << "matching precision: " << precision << endl;

			MatchesPrecision[imgIdx][transIdx] = precision;
		}
	}
	cout << endl << "保存结果到文件....." << endl;
	string Feature2dName =   "NOF"; // extractor->name();
	SaveEvaluationResults(MatchesPrecision, ".\\evaluate_results\\", Feature2dName, MatcherTypes[matcherType], TransformTypes[transformType]);
	cout << "保存结果完毕." << endl;
	return waitKey(1000);
}
