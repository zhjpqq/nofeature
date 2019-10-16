#include "NOF.h"
#include <iterator>

namespace mycv
{
	const float HARRIS_K = 0.04f;   //响应公式中的比例系数
	const int maxBins = 64;
	const int totalRects = 4;

	NOF::NOF(int _nfeatures, int _firstLevel, int _nlevels, int _patchSize, int _scoreType,
		double _scaleFactor, int _edgeThreshold, bool _doBinaryEncode)
		:nfeatures(_nfeatures), firstLevel(_firstLevel), nlevels(_nlevels), patchSize(_patchSize), scoreType(_scoreType),
		scaleFactor(_scaleFactor), edgeThreshold(_edgeThreshold), doBinaryEncode(_doBinaryEncode)
	{
	}
	NOF::~NOF()
	{
	}
	// 金字塔尺度因子
	static inline float getScale(int level, int firstLevel, double scaleFactor)
	{
		return (float)std::pow(scaleFactor, (double)(level - firstLevel));
	}

	// 计算Harris响应
	static void
		HarrisResponses(const Mat& img, vector<KeyPoint>& pts, int blockSize, float harris_k)
	{
		CV_Assert(img.type() == CV_8UC1 && blockSize*blockSize <= 2048);	//检测图像类型和大小是否合适

		size_t ptidx, ptsize = pts.size();	          //关键点的索引和总数量

		const uchar* ptr00 = img.ptr<uchar>();        //指向img的第一行第一个列，y=0，return data + step.p[0]*y;
		int step = (int)(img.step / img.elemSize1());   //行步长=行字节长/通道字节长，以字节为单位计量，step+1即为前进一个像素。elemSize1单个通道字节数，CV8UC1下为1字节，CV16UC1下为2字节
		int r = blockSize / 2;

		float scale = (1 << 2) * blockSize * 255.0f;  //1向右移2位变成4，
		scale = 1.0f / scale;
		float scale_sq_sq = scale * scale * scale * scale;

		AutoBuffer<int> ofsbuf(blockSize*blockSize);  //声明一个blockSize*blockSize大小的Buffer，进行采样框内的差分计算
		int* ofs = ofsbuf;
		for (int i = 0; i < blockSize; i++)
			for (int j = 0; j < blockSize; j++)
				ofs[i*blockSize + j] = (int)(i*step + j);

		for (ptidx = 0; ptidx < ptsize; ptidx++)     //在关键点容器pts内，逐个计算关键点的响应 
		{
			int x0 = cvRound(pts[ptidx].pt.x - r);    //在当前关键点pt下，找到其PathSize大小的采样矩形框中的第一个像素的坐标(x0,y0)
			int y0 = cvRound(pts[ptidx].pt.y - r);

			const uchar* ptr0 = ptr00 + y0*step + x0; //ptr0指向采样框的头像素(x0,y0)，ptr00指向图像数据头(0,0)，y0行标，x0列标，step按行前进的步长
			int a = 0, b = 0, c = 0;

			for (int k = 0; k < blockSize*blockSize; k++) //在采样框内循环计算差分矩阵
			{
				const uchar* ptr = ptr0 + ofs[k]; //对采样框内的像素进行逐点循环
				int Ix = (ptr[1] - ptr[-1]) * 2 + (ptr[-step + 1] - ptr[-step - 1]) + (ptr[step + 1] - ptr[step - 1]); //x方向差分
				int Iy = (ptr[step] - ptr[-step]) * 2 + (ptr[step - 1] - ptr[-step - 1]) + (ptr[step + 1] - ptr[-step + 1]); //y方向差分
				a += Ix*Ix;  //计算M矩阵的A\B\C\D量
				b += Iy*Iy;  //+=为M计算式中的"∑"，未进行高斯加权模糊
				c += Ix*Iy;
			}
			//HR = detM - k*(traceM)*(traceM) 
			//依据关键点采样框内的二阶矩矩阵M来计算Harris响应，用行列式和迹进行近似计算代替特征值计算 
			pts[ptidx].response = ((float)a * b - (float)c * c -
				harris_k * ((float)a + b) * ((float)a + b))*scale_sq_sq;
		}
	}

	// 构建金字塔图像
	void buildPyramidImages(const Mat& image, int firstLevel, int levelsNum, double scaleFactor, int border,
		vector<Mat>& imagePyramid, Mat mask = Mat())
	{
		// Pre-compute the scale pyramids
		imagePyramid.clear();
		imagePyramid.resize(levelsNum);
		vector<Mat> maskPyramid(levelsNum);

		for (int level = 0; level < levelsNum; ++level)
		{
			float scale = 1 / getScale(level, firstLevel, scaleFactor);        //计算当前层级的尺度因子
			Size sz(cvRound(image.cols*scale), cvRound(image.rows*scale));     //按尺度因子获得当前层级下的图像尺寸
			Size wholeSize(sz.width + border * 2, sz.height + border * 2);	   //按边界尺寸扩展当前层级下的图像尺寸，防止采样时边界缺失
			Mat temp(wholeSize, image.type()), masktemp;
			imagePyramid[level] = temp(Rect(border, border, sz.width, sz.height));//抓取ROI，(border, border)ROI起始坐标；非拷贝，只将其图像头放入imagePyramid[level]

			if (!mask.empty())	//如果掩膜非空
			{
				masktemp = Mat(wholeSize, mask.type());
				maskPyramid[level] = masktemp(Rect(border, border, sz.width, sz.height));
			}

			// Compute the resized image
			if (level != firstLevel)
			{
				if (level < firstLevel)	 //基本不执行
				{
					resize(image, imagePyramid[level], sz, 0, 0, INTER_LINEAR);
					if (!mask.empty())
						resize(mask, maskPyramid[level], sz, 0, 0, INTER_LINEAR);
				}
				else   //主要执行 imagePyramid[level-1] → imagePyramid[level]
				{
					resize(imagePyramid[level - 1], imagePyramid[level], sz, 0, 0, INTER_LINEAR);//注意resize的用法
					if (!mask.empty())
					{
						resize(maskPyramid[level - 1], maskPyramid[level], sz, 0, 0, INTER_LINEAR);
						threshold(maskPyramid[level], maskPyramid[level], 254, 0, THRESH_TOZERO);
					}
				}
				//将imagePyramid[level]填充到temp, 并在imagePyramid[level]周围添加border个像素宽的边界
				copyMakeBorder(imagePyramid[level], temp, border, border, border, border,
					BORDER_REFLECT_101 + BORDER_ISOLATED);
				if (!mask.empty())
					copyMakeBorder(maskPyramid[level], masktemp, border, border, border, border,
					BORDER_CONSTANT + BORDER_ISOLATED);
			}
			else
			{
				copyMakeBorder(image, temp, border, border, border, border,
					BORDER_REFLECT_101);
				if (!mask.empty())
					copyMakeBorder(mask, masktemp, border, border, border, border,
					BORDER_CONSTANT + BORDER_ISOLATED);
			}
		}
	}

	// 初始化采样模式图
	void initialPattern(Pattern &pat, KeyPoint& kpt, int maxRectBins, int totalRects)
	{
		pat.center = kpt.pt;
		pat.maxRidus = kpt.size;
		pat.totalBins = 0;
		//共有几个采样框、每个框内有多少Bins
		pat.bins.resize(totalRects);
		pat.sampleRidus.resize(totalRects);
		pat.sampleRects.resize(totalRects);

		for (int i = 0; i < totalRects; i++)
		{
			pat.sampleRidus[i] = pat.maxRidus*(1 - float(i) / totalRects);
			pat.sampleRects[i] = Rect_<float>(pat.center.x - pat.sampleRidus[i], pat.center.y - pat.sampleRidus[i], 2 * pat.sampleRidus[i], 2 * pat.sampleRidus[i]);
			pat.bins[i] = cvRound(maxRectBins*(1 - float(i) / totalRects));
			pat.totalBins += pat.bins[i];
			//cout << pat.sampleRects[i] <<"::"<<pat.sampleRidus[i]<< endl;
		}
		//cout << "Total Bins of Histogram：" << pat.totalBins << endl;
	}

	// 二值化float描述符
	void binaryDescriptor(Mat& floatDesc, Mat& binaryDesc)
	{
		//floatHist: histSize×1
		binaryDesc.create(floatDesc.rows, floatDesc.cols, CV_8U);
		int temp, cols = floatDesc.cols;
		for (int i = 0; i < cols - 1; i++)
		{
			if (floatDesc.at<float>(0, i) > floatDesc.at<float>(0, i + 1))
				binaryDesc.at<uchar>(0, i) = 1;
			else
				binaryDesc.at<uchar>(0, i) = 0;
		}

		if (floatDesc.at<float>(0, cols - 1) > floatDesc.at<float>(0, 0))
			binaryDesc.at<uchar>(0, cols - 1) = 1;
		else
			binaryDesc.at<uchar>(0, cols - 1) = 0;
	}

	// 计算金字塔中各层的关键点
	static void computeKeyPoints(const vector<Mat>& imagePyramid, const vector<Mat>& maskPyramid,
		vector<vector<KeyPoint>>& allKeypoints, int nfeatures, int firstLevel, double scaleFactor,
		int edgeThreshold, int patchSize, int scoreType)
	{
		//**********A。 确定金字塔中每层图像上的关键点数目 ***************

		int nlevels = (int)imagePyramid.size();	 //vector<Mat>中共有多少层图像
		vector<int> nfeaturesPerLevel(nlevels);  //每一层图像上的关键点数目

		// fill the extractors and descriptors for the corresponding scales
		// 首先计算第0层图像上的关键点数目
		float factor = (float)(1.0 / scaleFactor);
		float ndesiredFeaturesPerScale = nfeatures*(1 - factor) / (1 - (float)pow((double)factor, (double)nlevels));

		//然后计算每个尺度层级所需要的关键点数目 ，关键点数目按层级递减
		//公式为 nfeaturesPerLevel[level+1] = cvRound(ndesiredFeaturesPerScale[level]*factor); 
		//第nlevel层关键点数目由总数nfetures-前n-1层的总数
		int sumFeatures = 0;
		for (int level = 0; level < nlevels - 1; level++)
		{
			nfeaturesPerLevel[level] = cvRound(ndesiredFeaturesPerScale);
			sumFeatures += nfeaturesPerLevel[level];
			ndesiredFeaturesPerScale *= factor;
		}
		nfeaturesPerLevel[nlevels - 1] = std::max(nfeatures - sumFeatures, 0);

		// Make sure we forget about what is too close to the boundary
		//edge_threshold_ = std::max(edge_threshold_, patch_size_/2 + kKernelWidth / 2 + 2);

		//********************B。 确定采样区域的行列关系umax[v],用于IC_Angle中计算方向角***************

		//********************C。 检测金字塔中每层图像上的关键点***************************************

		//0.确定共有多少层关键点 →1.确定每层关键点数量 →2.构造当前层关键点容器 →3.FAST检测关键点 →4.去除边缘关键点 →5.Harris优选关键点 →6.给出关键点的octave\patchSize属性 →7.计算关键点的方向角angle属性

		allKeypoints.resize(nlevels);	//设定vector<vector<KeyPoint>> allKeypoints中共容纳多少层关键点

		for (int level = 0; level < nlevels; ++level)
		{
			int featuresNum = nfeaturesPerLevel[level];		//当前层的关键点数量
			allKeypoints[level].reserve(featuresNum * 2);   //依据当前层的关键点数量，设定好其容器的大小,为关键点数量×2，因为要从2倍的FAST关键点中挑选出一半的Harris关键点 

			vector<KeyPoint> & keypoints = allKeypoints[level]; //用keypoints指向当前层关键点的容器

			// Detect FAST features, 20 is a good threshold
			// 使用FAST检测当前层关键点，并将其放入对应容器 
			FastFeatureDetector fd(20, true);
			fd.detect(imagePyramid[level], keypoints, maskPyramid[level]);

			// Remove keypoints very close to the border
			// 去除靠近边界的关键点，根据图像尺寸、边界尺寸、关键点坐标之间的关系来判断关键点是否在边界之内
			// static void runByImageBorder( vector<KeyPoint>& keypoints, Size imageSize, int borderSize );
			KeyPointsFilter::runByImageBorder(keypoints, imagePyramid[level].size(), edgeThreshold);

			//用Harris响应选择出最佳关键点
			if (scoreType == ORB::HARRIS_SCORE)
			{
				// Keep more points than necessary as FAST does not give amazing corners
				// 按FAST响应值，保留2倍的所需关键点数量
				KeyPointsFilter::retainBest(keypoints, 2 * featuresNum);

				// Compute the Harris cornerness (better scoring than FAST)
				// 重新计算上述关键点的Harris响应值，替换FAST响应值
				HarrisResponses(imagePyramid[level], keypoints, 7, HARRIS_K);
			}

			//cull to the final desired level, using the new Harris scores or the original FAST scores.
			//保留最佳的featuresNum个关键点
			KeyPointsFilter::retainBest(keypoints, featuresNum);

			//获得当前level层的尺度因子，与当前层级level、第一层firstLevel、尺度因子scaleFactor三者有关
			float sf = getScale(level, firstLevel, scaleFactor);

			// Set the level of the coordinates
			// 设定当前关键点的层级属性octave，以及与之匹配的采样框大小size
			for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
				keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
			{
				keypoint->octave = level;
				keypoint->size = patchSize*sf;
				keypoint->angle = 0;			//由于不计算方向角，使用默认方向角为0
			}
			////计算当前层级上所有关键点的方向角
			//computeOrientation(imagePyramid[level], keypoints, halfPatchSize, umax);
		}
	}

	// 计算单个关键点的描述符
	void computeSingleDescriptor(const Mat& image, KeyPoint& kpt, Mat& desc, bool doBinary = false)
	{
		// → 基于单个关键点 → 计算灰度直方图  →→ 将直方图编码为float描述符 →→ 将folat描述符编码为binary描述符

		//为关键点初始化采样模式图
		Pattern pat;
		initialPattern(pat, kpt, maxBins, totalRects);

		//计算各个采样框内的灰度直方图
		vector<Mat> hist(pat.sampleRects.size());       //存放单个关键点的全部直方图
		for (int i = 0; i < pat.sampleRects.size(); i++)//遍历全部尺寸的采样框
		{
			//、提取ROI区域
			//cout << pat.sampleRects[i] << endl;
			//cout << image.size()<<endl;
			const Mat img = image(pat.sampleRects[i]);
			/// 当前框中的Bins数量
			int histSize = pat.bins[i];
			/// 直方图的Ranges范围
			float range[] = { 0, 256 };
			const float* histRange = { range };
			bool uniform = true; bool accumulate = false;
			calcHist(&img, 1, 0, Mat(), hist[i], 1, &histSize, &histRange, uniform, accumulate);
			normalize(hist[i], hist[i], 0, 1, NORM_MINMAX, -1, Mat());
			/*cout << endl << hist[i].cols << endl;*/  //确定hist[i]是按行排列？按列排列？→ 列向量，存储binValue → range[256]/bins[i]为统计步长
		}

		//将灰度直方图编码为float描述符
		Mat _desc;
		_desc.create(1, pat.totalBins, CV_32FC1);     //注意要为desc开辟空间，否则只指向空图像头
		int step = 0;								 //注意步长是动态的
		for (int i = 0; i < hist.size(); i++)        //遍历各个采样框对应的直方图 
		{
			for (int j = 0; j < hist[i].rows; j++)   //遍历直方图内的Bins
			{
				*_desc.ptr<float>(0, step + j) = hist[i].at<float>(j, 0);
			}
			step += hist[i].rows;
		}
		//cout << "描述符:" << _desc << endl;
		//cout << "描述符尺寸：" << _desc.size() << endl;
		//cout << "描述符类型：" << _desc.type() << endl;
		//waitKey(33333);

		//将float描述符编码为binary
		if (doBinary == true)
			binaryDescriptor(_desc, desc);
		else
			desc = _desc;
	}

	// 完整NOF操作函数()
	void NOF::operator()(InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
						 OutputArray _descriptors, bool useProvidedKeypoints) const
	{
		cout << "不可用！！！" << endl;
		////*****************预备，构造初始化数据和容器************************

		//CV_Assert(patchSize >= 2);

		//bool do_keypoints = !useProvidedKeypoints;    //是否使用提供的关键点
		//bool do_descriptors = true/*_descriptors.needed()*/;  //是否需要计算描述符

		//if ((!do_keypoints && !do_descriptors) || _image.empty()) //如果关键点已知、且不计算描述子，或者图像为空，则直接返回
		//	return;

		////ROI handling
		//const int HARRIS_BLOCK_SIZE = 9;    //Harris响应区域
		//int halfPatchSize = patchSize / 2;  //描述符提取区域
		//int border = std::max(edgeThreshold, std::max(halfPatchSize, HARRIS_BLOCK_SIZE / 2)) + 1;//边界值取三个量中的最大者

		//Mat image = _image.getMat(), mask = _mask.getMat();
		//if (image.type() != CV_8UC1)		//转换为灰度图像
		//	cvtColor(_image, image, CV_BGR2GRAY);

		////*****************首先，确定图像金字塔的层数************************

		//int levelsNum = this->nlevels;      //金字塔层数由输入参数决定 

		//if (!do_keypoints)//在关键点已提供的情况下，金字塔层数可通过传入的关键点层数得到
		//{
		//	// ****************注意，这是ORB/BRISK/SIFT与SURF/FREAK的区别*****************************
		//	// 是在金字塔上建立描述符，还是在原图上建立描述符 的区别
		//	// 事实上，描述子应该与octave参数无关，而仅仅与关键点的采样区大小size有关
		//	// 但本文中，是与octave相关的，换言之，描述子在金字塔上按octave逐级采样，而非在原始图中按size采样

		//	levelsNum = 0;
		//	for (size_t i = 0; i < _keypoints.size(); i++)//找到传入关键点中的最大octave，作为金字塔层数
		//		levelsNum = std::max(levelsNum, std::max(_keypoints[i].octave, 0));
		//	levelsNum++;
		//}

		////*****************其次，完成图像金字塔抽取************************

		//// Pre-compute the scale pyramids
		//vector<Mat> imagePyramid(levelsNum), maskPyramid(levelsNum);
		//for (int level = 0; level < levelsNum; ++level)
		//{
		//	float scale = 1 / getScale(level, firstLevel, scaleFactor);      //计算当前层级的尺度因子
		//	Size sz(cvRound(image.cols*scale), cvRound(image.rows*scale)); //按尺度因子获得当前层级下的图像尺寸
		//	Size wholeSize(sz.width + border * 2, sz.height + border * 2);	   //按边界尺寸扩展当前层级下的图像尺寸，防止采样时边界缺失
		//	Mat temp(wholeSize, image.type()), masktemp;
		//	imagePyramid[level] = temp(Rect(border, border, sz.width, sz.height));//抓取ROI，(border, border)ROI起始坐标；非拷贝，只将其图像头放入imagePyramid[level]

		//	if (!mask.empty())//如果掩膜非空
		//	{
		//		masktemp = Mat(wholeSize, mask.type());
		//		maskPyramid[level] = masktemp(Rect(border, border, sz.width, sz.height));
		//	}

		//	// Compute the resized image
		//	if (level != firstLevel)
		//	{
		//		if (level < firstLevel) //基本不执行
		//		{
		//			resize(image, imagePyramid[level], sz, 0, 0, INTER_LINEAR);
		//			if (!mask.empty())
		//				resize(mask, maskPyramid[level], sz, 0, 0, INTER_LINEAR);
		//		}
		//		else //imagePyramid[level-1] → imagePyramid[level]
		//		{
		//			resize(imagePyramid[level - 1], imagePyramid[level], sz, 0, 0, INTER_LINEAR);//注意resize的用法
		//			if (!mask.empty())
		//			{
		//				resize(maskPyramid[level - 1], maskPyramid[level], sz, 0, 0, INTER_LINEAR);
		//				threshold(maskPyramid[level], maskPyramid[level], 254, 0, THRESH_TOZERO);
		//			}
		//		}
		//		//将imagePyramid[level]填充到temp, 并在imagePyramid[level]周围添加border个像素宽的边界
		//		copyMakeBorder(imagePyramid[level], temp, border, border, border, border,
		//			BORDER_REFLECT_101 + BORDER_ISOLATED);
		//		if (!mask.empty())
		//			copyMakeBorder(maskPyramid[level], masktemp, border, border, border, border,
		//			BORDER_CONSTANT + BORDER_ISOLATED);
		//	}
		//	else
		//	{
		//		copyMakeBorder(image, temp, border, border, border, border,
		//			BORDER_REFLECT_101);
		//		if (!mask.empty())
		//			copyMakeBorder(mask, masktemp, border, border, border, border,
		//			BORDER_CONSTANT + BORDER_ISOLATED);
		//	}
		//}

		////*****************再次，计算并优选关键点************************

		//// Pre-compute the keypoints (we keep the best over all scales, so this has to be done beforehand
		//vector <vector<KeyPoint>> allKeypoints;
		//if (do_keypoints)
		//{
		//	// Get keypoints, those will be far enough from the border that no check will be required for the descriptor
		//	// 传入图像金字塔，计算金字塔中的全部关键点
		//	computeKeyPoints(imagePyramid, maskPyramid, allKeypoints,
		//		nfeatures, firstLevel, scaleFactor,
		//		edgeThreshold, patchSize, scoreType);
		//}

		////****************再次，校正金字塔图像中各层关键点坐标*********************** 
		//_keypoints.clear();
		//for (int level = 0; level < levelsNum; ++level)
		//{
		//	// Get the features and compute their orientation
		//	vector<KeyPoint>& keypoints = allKeypoints[level];   //获得第level层关键点及其数量
		//	int nkeypoints = (int)keypoints.size();

		//	// Copy to the output data   
		//	//校正关键点坐标，消除尺度变换
		//	if (level != firstLevel)
		//	{
		//		float scale = getScale(level, firstLevel, scaleFactor);
		//		for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
		//			keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
		//			keypoint->pt *= scale;
		//	}
		//	// And add the keypoints to the output  //insert(const_iterator _Where, _Iter _First, _Iter _Last)
		//	// _keypoints.clear()后，_keypoints.begain()与_keypoints.end()同位置，区间[)
		//	// 使用inset()函数将每层level的关键点逐层添加到_keypoints的末尾
		//	_keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
		//}


		////*****************最后，开始计算描述符************************
		//Mat descriptors;
		//if (do_descriptors)     //判断是否计算描述符
		//{
		//	//-------------构造描述符容器：descriptors
		//	int nkeypoints = 0;
		//	for (int level = 0; level < levelsNum; ++level)
		//		nkeypoints += (int)allKeypoints[level].size();  //统计全部关键点数量
		//	if (nkeypoints == 0)								//检查存在关键点
		//		_descriptors.release();
		//	else
		//	{
		//		_descriptors.create(nkeypoints, descriptorSize(), CV_32F); //按关键点数量和描述符类型，构造描述符容器
		//		descriptors = _descriptors.getMat();   //********** 将OutputArray转换为Mat型接口，OutputArray可作为Mat型、vector型等使用
		//	}

		//	//****************** 3. 计算金字塔各层关键点的描述符***************************
		//	_keypoints.clear();
		//	for (int level = 0; level < levelsNum; ++level) //遍历金字塔内各层
		//	{
		//		//取出当前层关键点、当前层图像
		//		vector<KeyPoint>& kpt_per_level = allKeypoints[level];
		//		Mat& img_per_level = imagePyramid[level];

		//		for (int k = 0; k < kpt_per_level.size(); k++) //遍历层内各关键点
		//		{
		//			Mat desc;
		//			desc = descriptors.rowRange(k, k + 1);
		//			computeSingleDescriptor(img_per_level, kpt_per_level[k], desc, doBinaryEncode);
		//			//descriptors.push_back(desc);
		//		}
		//	}
		//}
	}

	// 关键点检测
	void NOF::detectImpl(const Mat& _image, vector<KeyPoint>& _keypoints, const Mat& _mask) const
	{
		//*****************预备，构造初始化数据和容器************************

		CV_Assert(patchSize >= 2);

		//ROI handling
		const int HARRIS_BLOCK_SIZE = 9;    //Harris响应区域
		int halfPatchSize = patchSize / 2;  //描述符提取区域
		int border = std::max(edgeThreshold, std::max(halfPatchSize, HARRIS_BLOCK_SIZE / 2)) + 1;//边界值取三个量中的最大者

		Mat image = _image, mask = _mask;
		if (image.type() != CV_8UC1)		//转换为灰度图像
			cvtColor(_image, image, CV_BGR2GRAY);

		//*****************首先，确定图像金字塔的层数************************

		int levelsNum = this->nlevels;      //金字塔层数由输入参数决定 

		//*****************其次，完成图像金字塔抽取************************

		// Pre-compute the scale pyramids
		vector<Mat> imagePyramid(levelsNum), maskPyramid(levelsNum);
		for (int level = 0; level < levelsNum; ++level)
		{
			float scale = 1 / getScale(level, firstLevel, scaleFactor);      //计算当前层级的尺度因子
			Size sz(cvRound(image.cols*scale), cvRound(image.rows*scale));   //按尺度因子获得当前层级下的图像尺寸
			Size wholeSize(sz.width + border * 2, sz.height + border * 2);	 //按边界尺寸扩展当前层级下的图像尺寸，防止采样时边界缺失
			Mat temp(wholeSize, image.type()), masktemp;
			imagePyramid[level] = temp(Rect(border, border, sz.width, sz.height));//抓取ROI，(border, border)ROI起始坐标；非拷贝，只将其图像头放入imagePyramid[level]

			if (!mask.empty())//如果掩膜非空
			{
				masktemp = Mat(wholeSize, mask.type());
				maskPyramid[level] = masktemp(Rect(border, border, sz.width, sz.height));
			}

			// Compute the resized image
			if (level != firstLevel)
			{
				if (level < firstLevel) //基本不执行
				{
					resize(image, imagePyramid[level], sz, 0, 0, INTER_LINEAR);
					if (!mask.empty())
						resize(mask, maskPyramid[level], sz, 0, 0, INTER_LINEAR);
				}
				else //imagePyramid[level-1] → imagePyramid[level]
				{
					resize(imagePyramid[level - 1], imagePyramid[level], sz, 0, 0, INTER_LINEAR);//注意resize的用法
					if (!mask.empty())
					{
						resize(maskPyramid[level - 1], maskPyramid[level], sz, 0, 0, INTER_LINEAR);
						threshold(maskPyramid[level], maskPyramid[level], 254, 0, THRESH_TOZERO);
					}
				}
				//将imagePyramid[level]填充到temp, 并在imagePyramid[level]周围添加border个像素宽的边界
				copyMakeBorder(imagePyramid[level], temp, border, border, border, border,
					BORDER_REFLECT_101 + BORDER_ISOLATED);
				if (!mask.empty())
					copyMakeBorder(maskPyramid[level], masktemp, border, border, border, border,
					BORDER_CONSTANT + BORDER_ISOLATED);
			}
			else
			{
				copyMakeBorder(image, temp, border, border, border, border,
					BORDER_REFLECT_101);
				if (!mask.empty())
					copyMakeBorder(mask, masktemp, border, border, border, border,
					BORDER_CONSTANT + BORDER_ISOLATED);
			}
		}
		//*****************再次，计算并优选金字塔各层关键点************************

		// Pre-compute the keypoints (we keep the best over all scales, so this has to be done beforehand
		vector <vector<KeyPoint>> allKeypoints;
		if (true)
		{
			// Get keypoints, those will be far enough from the border that no check will be required for the descriptor
			// 传入图像金字塔，计算金字塔中各层图像上的全部关键点
			computeKeyPoints(imagePyramid, maskPyramid, allKeypoints,
				nfeatures, firstLevel, scaleFactor,
				edgeThreshold, patchSize, scoreType);
		}
		//****************最后，校正金字塔图像中各层关键点坐标*********************** 
		_keypoints.clear();
		for (int level = 0; level < levelsNum; ++level)
		{
			// Get the features and compute their orientation
			vector<KeyPoint>& keypoints = allKeypoints[level];   //获得第level层关键点及其数量
			int nkeypoints = (int)keypoints.size();

			// Copy to the output data   
			//校正关键点坐标，消除尺度变换
			if (level != firstLevel)
			{
				float scale = getScale(level, firstLevel, scaleFactor);
				for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
					keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
					keypoint->pt *= scale;
			}
			// And add the keypoints to the output  //insert(const_iterator _Where, _Iter _First, _Iter _Last)
			// _keypoints.clear()后，_keypoints.begain()与_keypoints.end()同位置，区间[)
			// 使用inset()函数将每层level的关键点逐层添加到_keypoints的末尾
			_keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
		}
	}

	// 描述符计算
	void NOF::computeImpl(const Mat& _image, vector<KeyPoint>& keypoints, Mat& descriptors) const
	{
		//1.构建金字塔图像 →2.构建金字塔关键点 →3.逐层计算各关键点描述符（1.逐点加载采样模式 →2.逐点计算描述符）

		Mat image;
		if (_image.channels() == 3)
			cvtColor(_image, image, CV_BGR2GRAY);
		else
			image = _image;

		//*************** 1.构建金字塔图像 *********************

		//int firstLevel = 0;													     //金字塔首层索引
		int levelsNum = 0;						                                     //金字塔总层数
		//((MGHF*)this)->edgeThreshold = keypoints[0].size;                          //金字塔边缘阈值
		for (size_t i = 0; i < keypoints.size(); i++)
		{
			levelsNum = std::max(levelsNum, std::max(keypoints[i].octave, 0)); //找到关键点中的最大octave，作为金字塔层数		
			//((MGHF*)this)->edgeThreshold = std::min(keypoints[i].size, (float)((MGHF*)this)->edgeThreshold); //找到关键点中的最大采样半径size,作为边界阈值，防止采样模式图超出图像边界
		}
		levelsNum++;														   //总数比索引号+1

		int border = edgeThreshold + 1;   //在原图内圈加上border，采样锚点即关键点不能超出border框，且采样范围不能超出图像边界，因此一般，采样R≤border
		vector<Mat> imagePyramid;
		buildPyramidImages(image, this->firstLevel, levelsNum, this->scaleFactor, border, imagePyramid);

		//*************** 2.计算金字塔各层关键点 *********************

		// 1.过滤边缘关键点 →→ 2.按金字塔层级归类关键点 →→ 3.按尺度比例重新缩放关键点坐标、采样范围

		// Cluster the input keypoints depending on the level they were computed at
		vector<vector<KeyPoint>> allKeypoints;
		allKeypoints.clear();
		allKeypoints.resize(levelsNum);
		for (int i = 0; i<keypoints.size(); i++)
			allKeypoints[keypoints[i].octave].push_back(keypoints[i]);

		// Make sure we rescale the coordinates and size of keypoints  
		for (int level = 0; level < levelsNum; ++level)
		{
			if (level == firstLevel)//第一层不调整
				continue;

			vector<KeyPoint> & kpts = allKeypoints[level];		        //取出第level层的关键点
			float scale = 1 / getScale(level, firstLevel, scaleFactor); //获得level层的尺度因子
			for (vector<KeyPoint>::iterator keypoint = kpts.begin(),
				keypointEnd = kpts.end(); keypoint != keypointEnd; ++keypoint)
			{
				//cout <<"层数索引："<<level<< "调整前关键点尺寸:" << keypoint->size << "比例因子:"<<scale<<endl;
				keypoint->pt *= scale;			//调整坐标尺度、采样领域尺度，*scale ，与采样生成金字塔时相反
				keypoint->size *= scale;        //未调整前，关键点尺寸，随层数逐级变大，从31、44、……，调整后全部为31
				//keypoint->size *= scale;        //再次调整后，关键点尺寸，随层级逐级变小，从31、25……8
				//cout << "层数索引："<< level << "调整后关键点尺寸:" << keypoint->size << "比例因子:" << scale << endl;
			}
		}
		for (int level = 0; level < levelsNum; ++level)
		{
			vector<KeyPoint> & kpts = allKeypoints[level];		        //取出第level层的关键点
			// Remove keypoints very close to the border  
			KeyPointsFilter::runByImageBorder(kpts, imagePyramid[level].size(), edgeThreshold);
		}

		//****************** 3. 计算金字塔各层关键点的描述符***************************

		for (int level = 0; level < levelsNum; ++level) //遍历金字塔内各层
		{
			//取出当前层关键点、当前层图像
			vector<KeyPoint>& kpt_per_level = allKeypoints[level];
			Mat& img_per_level = imagePyramid[level];

			for (int k = 0; k < kpt_per_level.size(); k++) //遍历层内各关键点
			{
				Mat desc;
				computeSingleDescriptor(img_per_level, kpt_per_level[k], desc, doBinaryEncode);
				descriptors.push_back(desc);
			}
		}
	}

	/** returns the descriptor size */
	int NOF::descriptorSize() const
	{
		return 576;
	}
	/** returns the descriptor type */
	int NOF::descriptorType() const
	{
		return CV_32F;
	}

}