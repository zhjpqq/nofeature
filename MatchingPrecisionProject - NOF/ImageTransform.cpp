#include "ImageTransform.h"

//add rotation to the src image, return the rotated image and the corresponding ground-truth matrix
void AddRotationTransformation(const Mat& src, double rotAngle, Mat& dst, Mat& homo)
{
	Point2f rotCenter(src.cols / 2.f, src.rows / 2.f);//take the src image center as rotation center
	Mat rotMat(2, 3, CV_32FC1);
	rotMat = cv::getRotationMatrix2D(rotCenter, rotAngle, 1.0); //obtain the rotation matrix
	Mat_<float> rotmat = rotMat;

	//compute the homography matrix
	vector<Point2f> srcVetexes = { Point2f(0, 0), Point2f(src.cols, 0),Point2f(0, src.rows), Point2f(src.cols, src.rows) };
	vector<Point2f> dstVetexes = { Point2f(0, 0), Point2f(0, 0), Point2f(0, 0), Point2f(0, 0) };
	for (size_t kpIdx = 0; kpIdx < 4; kpIdx++){
		Point2f& pt = srcVetexes[kpIdx];
		Point2f& rpt = dstVetexes[kpIdx];
		rpt.x = pt.x*rotmat(0, 0) + pt.y*rotmat(0, 1) + rotmat(0, 2);
		rpt.y = pt.x*rotmat(1, 0) + pt.y*rotmat(1, 1) + rotmat(1, 2);
	}
	//obtain the correspoinding homography matrix using the relation of four image vetexes
	homo = cv::getPerspectiveTransform(srcVetexes, dstVetexes);

	//warp the src image to dst image using rotation matrix
	cv::warpAffine(src, dst, rotMat, src.size());
	// or use homography matrix to generate dst image
	// cv::warpPerspective(src, dst, homo, src.size());
	return;
}

//add zoom to the src image, return the zoomed image and the corresponding ground-truth matrix
void AddZoomTransformation(const Mat& src, double zoomScale, Mat& dst, Mat& homo)
{

	//compute the homography matrix
	vector<Point2f> srcVetexes = { Point2f(0, 0), Point2f(src.cols, 0), Point2f(0, src.rows), Point2f(src.cols, src.rows) };
	vector<Point2f> dstVetexes = { Point2f(0, 0), Point2f(0, 0), Point2f(0, 0), Point2f(0, 0) };
	for (size_t kpIdx = 0; kpIdx < 4; kpIdx++)
		dstVetexes[kpIdx] = zoomScale*srcVetexes[kpIdx];

	//obtain the correspoinding homography matrix using the relation of four image vetexes
	homo = cv::getPerspectiveTransform(srcVetexes, dstVetexes);

	//the size of dst image
	Size dsize(/*cvRound*/(src.size().width*zoomScale), /*cvRound*/(src.size().height*zoomScale));

	//resize the src image to dst image using bilinear interploration
	resize(src, dst, dsize, 0., 0., INTER_LINEAR);
	// or use homography matrix to generate dst image
	//cv::warpPerspective(src, dst, homo, dsize);

	return;
}

//add Viewpoint changes to the src image, return the transformed image and the corresponding ground-truth matrix
void AddViewpointTransformation(const Mat& src, double viewAngle, Mat& dst, Mat& H)
{
	H.create(3, 3, CV_32FC1);
	setIdentity(H); //设定对角线全为1

	stringstream ss;
	ss << ".\\viewpoints\\" << "H1to" << viewAngle + 1 << "p.xml";
	FileStorage FS(ss.str(), FileStorage::READ);
	if (FS.isOpened())
		FS["H"] >> H; //读入视角变换矩阵

	// use homography matrix to generate dst image
	cv::warpPerspective(src, dst, H, src.size(), cv::INTER_CUBIC);

}

//add Bluring changes to the src image, return the blured image and the corresponding ground-truth matrix
void AddGaussianBluringTransformation(const Mat& src, double blurScale, Mat& dst, Mat& homo)
{
	//obtain the correspoinding homography matrix using the relation of four image vetexes
	/*vector<Point2f> srcVetexes = { Point2f(0, 0), Point2f(src.cols, 0),
	Point2f(0, src.rows), Point2f(src.cols, src.rows) };
	vector<Point2f> dstVetexes = srcVetexes;
	homo = cv::getPerspectiveTransform(srcVetexes, dstVetexes);*/

	//in fact, homo is a identity matrix under image bluring
	homo = Mat::zeros(3, 3, CV_32FC1);
	setIdentity(homo);  //单位矩阵

	//blurring the src image to dst image using a const aperture 7*7 and the blurScale as sigma
	cv::GaussianBlur(src, dst, Size(7, 7), blurScale, blurScale, 4);

	return;
}

//add white noises to the src image, return the noised image and the corresponding ground-truth matrix
void AddWhiteNoisesTransformation(const Mat& src, double noiseStrength, Mat& dst, Mat& homo)
{
	//obtain the correspoinding homography matrix using the relation of four image vetexes
	/*vector<Point2f> srcVetexes = { Point2f(0, 0), Point2f(src.cols, 0),
	Point2f(0, src.rows), Point2f(src.cols, src.rows) };
	vector<Point2f> dstVetexes = srcVetexes;
	homo = cv::getPerspectiveTransform(srcVetexes, dstVetexes);*/

	//in fact, homo is a identity matrix under image noise
	homo = Mat::zeros(3, 3, CV_32FC1);
	setIdentity(homo);

	src.copyTo(dst);
	//噪声数量
	int noiseCount = int(noiseStrength*dst.rows*dst.cols);
	Mat_<Vec3b> dstImg = dst;    //一次索引3个通道，一次访问一个像素点，不用分通道访问或索引
	//开始在目标图像随机添加噪声
	for (int k = 0; k < noiseCount; k++){
		//rand()是随机数生成函数
		int col = rand() % src.cols;   //求余，范围在0—cols
		int row = rand() % src.rows;   //随机生成位置
		uchar randVal = static_cast<uchar>(rand() % 255);  //随机生成亮度值
		dstImg(row, col) = Vec3b(randVal, randVal, randVal);//Vec3b：处理3通道图像的单个像素
	}
	return;
}

//add contrast and lightness to the src image, return the transformed image and the corresponding ground-truth matrix
void AddConstractLightnessTransformation(const Mat& src, double lightChange, Mat& dst, Mat& homo)
{
	//obtain the correspoinding homography matrix using the relation of four image vetexes
	/*vector<Point2f> srcVetexes = { Point2f(0, 0), Point2f(src.cols, 0),
	Point2f(0, src.rows), Point2f(src.cols, src.rows) };
	vector<Point2f> dstVetexes = srcVetexes;
	homo = cv::getPerspectiveTransform(srcVetexes, dstVetexes);*/

	//in fact, homo is a identity matrix under image lightness changes
	homo = Mat::zeros(3, 3, CV_32FC1);
	setIdentity(homo);

	//! scales array elements, computes absolute values and converts the results to 8-bit unsigned integers: 
	//cv::convertScaleAbs(src, dst, 1.0, lightChange);// dst(i)=saturate_cast<uchar>abs(src(i)*alpha+beta)
	
	src.copyTo(dst);
	for (int r = 0; r < dst.rows; r++)
		for (int c = 0; c < dst.cols; c++)
		{
			Vec3b pixel = dst.at<Vec3b>(r, c);
			for (int ch = 0; ch < 3; ch++)
			{
				if ((pixel[ch] + lightChange >=0) && (pixel[ch] + lightChange <= 255))
					pixel[ch] += lightChange;

				if (pixel[ch] + lightChange < 0)
					pixel[ch] = 0;

				if (pixel[ch] + lightChange > 255)
					pixel[ch] = 255;
			}
			dst.at<Vec3b>(r, c) = pixel;
		}
}
