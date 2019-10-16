#include "NOF.h"
#include <iterator>

namespace mycv
{
	const float HARRIS_K = 0.04f;   //��Ӧ��ʽ�еı���ϵ��
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
	// �������߶�����
	static inline float getScale(int level, int firstLevel, double scaleFactor)
	{
		return (float)std::pow(scaleFactor, (double)(level - firstLevel));
	}

	// ����Harris��Ӧ
	static void
		HarrisResponses(const Mat& img, vector<KeyPoint>& pts, int blockSize, float harris_k)
	{
		CV_Assert(img.type() == CV_8UC1 && blockSize*blockSize <= 2048);	//���ͼ�����ͺʹ�С�Ƿ����

		size_t ptidx, ptsize = pts.size();	          //�ؼ����������������

		const uchar* ptr00 = img.ptr<uchar>();        //ָ��img�ĵ�һ�е�һ���У�y=0��return data + step.p[0]*y;
		int step = (int)(img.step / img.elemSize1());   //�в���=���ֽڳ�/ͨ���ֽڳ������ֽ�Ϊ��λ������step+1��Ϊǰ��һ�����ء�elemSize1����ͨ���ֽ�����CV8UC1��Ϊ1�ֽڣ�CV16UC1��Ϊ2�ֽ�
		int r = blockSize / 2;

		float scale = (1 << 2) * blockSize * 255.0f;  //1������2λ���4��
		scale = 1.0f / scale;
		float scale_sq_sq = scale * scale * scale * scale;

		AutoBuffer<int> ofsbuf(blockSize*blockSize);  //����һ��blockSize*blockSize��С��Buffer�����в������ڵĲ�ּ���
		int* ofs = ofsbuf;
		for (int i = 0; i < blockSize; i++)
			for (int j = 0; j < blockSize; j++)
				ofs[i*blockSize + j] = (int)(i*step + j);

		for (ptidx = 0; ptidx < ptsize; ptidx++)     //�ڹؼ�������pts�ڣ��������ؼ������Ӧ 
		{
			int x0 = cvRound(pts[ptidx].pt.x - r);    //�ڵ�ǰ�ؼ���pt�£��ҵ���PathSize��С�Ĳ������ο��еĵ�һ�����ص�����(x0,y0)
			int y0 = cvRound(pts[ptidx].pt.y - r);

			const uchar* ptr0 = ptr00 + y0*step + x0; //ptr0ָ��������ͷ����(x0,y0)��ptr00ָ��ͼ������ͷ(0,0)��y0�б꣬x0�б꣬step����ǰ���Ĳ���
			int a = 0, b = 0, c = 0;

			for (int k = 0; k < blockSize*blockSize; k++) //�ڲ�������ѭ�������־���
			{
				const uchar* ptr = ptr0 + ofs[k]; //�Բ������ڵ����ؽ������ѭ��
				int Ix = (ptr[1] - ptr[-1]) * 2 + (ptr[-step + 1] - ptr[-step - 1]) + (ptr[step + 1] - ptr[step - 1]); //x������
				int Iy = (ptr[step] - ptr[-step]) * 2 + (ptr[step - 1] - ptr[-step - 1]) + (ptr[step + 1] - ptr[-step + 1]); //y������
				a += Ix*Ix;  //����M�����A\B\C\D��
				b += Iy*Iy;  //+=ΪM����ʽ�е�"��"��δ���и�˹��Ȩģ��
				c += Ix*Iy;
			}
			//HR = detM - k*(traceM)*(traceM) 
			//���ݹؼ���������ڵĶ��׾ؾ���M������Harris��Ӧ��������ʽ�ͼ����н��Ƽ����������ֵ���� 
			pts[ptidx].response = ((float)a * b - (float)c * c -
				harris_k * ((float)a + b) * ((float)a + b))*scale_sq_sq;
		}
	}

	// ����������ͼ��
	void buildPyramidImages(const Mat& image, int firstLevel, int levelsNum, double scaleFactor, int border,
		vector<Mat>& imagePyramid, Mat mask = Mat())
	{
		// Pre-compute the scale pyramids
		imagePyramid.clear();
		imagePyramid.resize(levelsNum);
		vector<Mat> maskPyramid(levelsNum);

		for (int level = 0; level < levelsNum; ++level)
		{
			float scale = 1 / getScale(level, firstLevel, scaleFactor);        //���㵱ǰ�㼶�ĳ߶�����
			Size sz(cvRound(image.cols*scale), cvRound(image.rows*scale));     //���߶����ӻ�õ�ǰ�㼶�µ�ͼ��ߴ�
			Size wholeSize(sz.width + border * 2, sz.height + border * 2);	   //���߽�ߴ���չ��ǰ�㼶�µ�ͼ��ߴ磬��ֹ����ʱ�߽�ȱʧ
			Mat temp(wholeSize, image.type()), masktemp;
			imagePyramid[level] = temp(Rect(border, border, sz.width, sz.height));//ץȡROI��(border, border)ROI��ʼ���ꣻ�ǿ�����ֻ����ͼ��ͷ����imagePyramid[level]

			if (!mask.empty())	//�����Ĥ�ǿ�
			{
				masktemp = Mat(wholeSize, mask.type());
				maskPyramid[level] = masktemp(Rect(border, border, sz.width, sz.height));
			}

			// Compute the resized image
			if (level != firstLevel)
			{
				if (level < firstLevel)	 //������ִ��
				{
					resize(image, imagePyramid[level], sz, 0, 0, INTER_LINEAR);
					if (!mask.empty())
						resize(mask, maskPyramid[level], sz, 0, 0, INTER_LINEAR);
				}
				else   //��Ҫִ�� imagePyramid[level-1] �� imagePyramid[level]
				{
					resize(imagePyramid[level - 1], imagePyramid[level], sz, 0, 0, INTER_LINEAR);//ע��resize���÷�
					if (!mask.empty())
					{
						resize(maskPyramid[level - 1], maskPyramid[level], sz, 0, 0, INTER_LINEAR);
						threshold(maskPyramid[level], maskPyramid[level], 254, 0, THRESH_TOZERO);
					}
				}
				//��imagePyramid[level]��䵽temp, ����imagePyramid[level]��Χ���border�����ؿ�ı߽�
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

	// ��ʼ������ģʽͼ
	void initialPattern(Pattern &pat, KeyPoint& kpt, int maxRectBins, int totalRects)
	{
		pat.center = kpt.pt;
		pat.maxRidus = kpt.size;
		pat.totalBins = 0;
		//���м���������ÿ�������ж���Bins
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
		//cout << "Total Bins of Histogram��" << pat.totalBins << endl;
	}

	// ��ֵ��float������
	void binaryDescriptor(Mat& floatDesc, Mat& binaryDesc)
	{
		//floatHist: histSize��1
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

	// ����������и���Ĺؼ���
	static void computeKeyPoints(const vector<Mat>& imagePyramid, const vector<Mat>& maskPyramid,
		vector<vector<KeyPoint>>& allKeypoints, int nfeatures, int firstLevel, double scaleFactor,
		int edgeThreshold, int patchSize, int scoreType)
	{
		//**********A�� ȷ����������ÿ��ͼ���ϵĹؼ�����Ŀ ***************

		int nlevels = (int)imagePyramid.size();	 //vector<Mat>�й��ж��ٲ�ͼ��
		vector<int> nfeaturesPerLevel(nlevels);  //ÿһ��ͼ���ϵĹؼ�����Ŀ

		// fill the extractors and descriptors for the corresponding scales
		// ���ȼ����0��ͼ���ϵĹؼ�����Ŀ
		float factor = (float)(1.0 / scaleFactor);
		float ndesiredFeaturesPerScale = nfeatures*(1 - factor) / (1 - (float)pow((double)factor, (double)nlevels));

		//Ȼ�����ÿ���߶Ȳ㼶����Ҫ�Ĺؼ�����Ŀ ���ؼ�����Ŀ���㼶�ݼ�
		//��ʽΪ nfeaturesPerLevel[level+1] = cvRound(ndesiredFeaturesPerScale[level]*factor); 
		//��nlevel��ؼ�����Ŀ������nfetures-ǰn-1�������
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

		//********************B�� ȷ��������������й�ϵumax[v],����IC_Angle�м��㷽���***************

		//********************C�� ����������ÿ��ͼ���ϵĹؼ���***************************************

		//0.ȷ�����ж��ٲ�ؼ��� ��1.ȷ��ÿ��ؼ������� ��2.���쵱ǰ��ؼ������� ��3.FAST���ؼ��� ��4.ȥ����Ե�ؼ��� ��5.Harris��ѡ�ؼ��� ��6.�����ؼ����octave\patchSize���� ��7.����ؼ���ķ����angle����

		allKeypoints.resize(nlevels);	//�趨vector<vector<KeyPoint>> allKeypoints�й����ɶ��ٲ�ؼ���

		for (int level = 0; level < nlevels; ++level)
		{
			int featuresNum = nfeaturesPerLevel[level];		//��ǰ��Ĺؼ�������
			allKeypoints[level].reserve(featuresNum * 2);   //���ݵ�ǰ��Ĺؼ����������趨���������Ĵ�С,Ϊ�ؼ���������2����ΪҪ��2����FAST�ؼ�������ѡ��һ���Harris�ؼ��� 

			vector<KeyPoint> & keypoints = allKeypoints[level]; //��keypointsָ��ǰ��ؼ��������

			// Detect FAST features, 20 is a good threshold
			// ʹ��FAST��⵱ǰ��ؼ��㣬����������Ӧ���� 
			FastFeatureDetector fd(20, true);
			fd.detect(imagePyramid[level], keypoints, maskPyramid[level]);

			// Remove keypoints very close to the border
			// ȥ�������߽�Ĺؼ��㣬����ͼ��ߴ硢�߽�ߴ硢�ؼ�������֮��Ĺ�ϵ���жϹؼ����Ƿ��ڱ߽�֮��
			// static void runByImageBorder( vector<KeyPoint>& keypoints, Size imageSize, int borderSize );
			KeyPointsFilter::runByImageBorder(keypoints, imagePyramid[level].size(), edgeThreshold);

			//��Harris��Ӧѡ�����ѹؼ���
			if (scoreType == ORB::HARRIS_SCORE)
			{
				// Keep more points than necessary as FAST does not give amazing corners
				// ��FAST��Ӧֵ������2��������ؼ�������
				KeyPointsFilter::retainBest(keypoints, 2 * featuresNum);

				// Compute the Harris cornerness (better scoring than FAST)
				// ���¼��������ؼ����Harris��Ӧֵ���滻FAST��Ӧֵ
				HarrisResponses(imagePyramid[level], keypoints, 7, HARRIS_K);
			}

			//cull to the final desired level, using the new Harris scores or the original FAST scores.
			//������ѵ�featuresNum���ؼ���
			KeyPointsFilter::retainBest(keypoints, featuresNum);

			//��õ�ǰlevel��ĳ߶����ӣ��뵱ǰ�㼶level����һ��firstLevel���߶�����scaleFactor�����й�
			float sf = getScale(level, firstLevel, scaleFactor);

			// Set the level of the coordinates
			// �趨��ǰ�ؼ���Ĳ㼶����octave���Լ���֮ƥ��Ĳ������Сsize
			for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
				keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
			{
				keypoint->octave = level;
				keypoint->size = patchSize*sf;
				keypoint->angle = 0;			//���ڲ����㷽��ǣ�ʹ��Ĭ�Ϸ����Ϊ0
			}
			////���㵱ǰ�㼶�����йؼ���ķ����
			//computeOrientation(imagePyramid[level], keypoints, halfPatchSize, umax);
		}
	}

	// ���㵥���ؼ����������
	void computeSingleDescriptor(const Mat& image, KeyPoint& kpt, Mat& desc, bool doBinary = false)
	{
		// �� ���ڵ����ؼ��� �� ����Ҷ�ֱ��ͼ  ���� ��ֱ��ͼ����Ϊfloat������ ���� ��folat����������Ϊbinary������

		//Ϊ�ؼ����ʼ������ģʽͼ
		Pattern pat;
		initialPattern(pat, kpt, maxBins, totalRects);

		//��������������ڵĻҶ�ֱ��ͼ
		vector<Mat> hist(pat.sampleRects.size());       //��ŵ����ؼ����ȫ��ֱ��ͼ
		for (int i = 0; i < pat.sampleRects.size(); i++)//����ȫ���ߴ�Ĳ�����
		{
			//����ȡROI����
			//cout << pat.sampleRects[i] << endl;
			//cout << image.size()<<endl;
			const Mat img = image(pat.sampleRects[i]);
			/// ��ǰ���е�Bins����
			int histSize = pat.bins[i];
			/// ֱ��ͼ��Ranges��Χ
			float range[] = { 0, 256 };
			const float* histRange = { range };
			bool uniform = true; bool accumulate = false;
			calcHist(&img, 1, 0, Mat(), hist[i], 1, &histSize, &histRange, uniform, accumulate);
			normalize(hist[i], hist[i], 0, 1, NORM_MINMAX, -1, Mat());
			/*cout << endl << hist[i].cols << endl;*/  //ȷ��hist[i]�ǰ������У��������У��� ���������洢binValue �� range[256]/bins[i]Ϊͳ�Ʋ���
		}

		//���Ҷ�ֱ��ͼ����Ϊfloat������
		Mat _desc;
		_desc.create(1, pat.totalBins, CV_32FC1);     //ע��ҪΪdesc���ٿռ䣬����ָֻ���ͼ��ͷ
		int step = 0;								 //ע�ⲽ���Ƕ�̬��
		for (int i = 0; i < hist.size(); i++)        //���������������Ӧ��ֱ��ͼ 
		{
			for (int j = 0; j < hist[i].rows; j++)   //����ֱ��ͼ�ڵ�Bins
			{
				*_desc.ptr<float>(0, step + j) = hist[i].at<float>(j, 0);
			}
			step += hist[i].rows;
		}
		//cout << "������:" << _desc << endl;
		//cout << "�������ߴ磺" << _desc.size() << endl;
		//cout << "���������ͣ�" << _desc.type() << endl;
		//waitKey(33333);

		//��float����������Ϊbinary
		if (doBinary == true)
			binaryDescriptor(_desc, desc);
		else
			desc = _desc;
	}

	// ����NOF��������()
	void NOF::operator()(InputArray _image, InputArray _mask, vector<KeyPoint>& _keypoints,
						 OutputArray _descriptors, bool useProvidedKeypoints) const
	{
		cout << "�����ã�����" << endl;
		////*****************Ԥ���������ʼ�����ݺ�����************************

		//CV_Assert(patchSize >= 2);

		//bool do_keypoints = !useProvidedKeypoints;    //�Ƿ�ʹ���ṩ�Ĺؼ���
		//bool do_descriptors = true/*_descriptors.needed()*/;  //�Ƿ���Ҫ����������

		//if ((!do_keypoints && !do_descriptors) || _image.empty()) //����ؼ�����֪���Ҳ����������ӣ�����ͼ��Ϊ�գ���ֱ�ӷ���
		//	return;

		////ROI handling
		//const int HARRIS_BLOCK_SIZE = 9;    //Harris��Ӧ����
		//int halfPatchSize = patchSize / 2;  //��������ȡ����
		//int border = std::max(edgeThreshold, std::max(halfPatchSize, HARRIS_BLOCK_SIZE / 2)) + 1;//�߽�ֵȡ�������е������

		//Mat image = _image.getMat(), mask = _mask.getMat();
		//if (image.type() != CV_8UC1)		//ת��Ϊ�Ҷ�ͼ��
		//	cvtColor(_image, image, CV_BGR2GRAY);

		////*****************���ȣ�ȷ��ͼ��������Ĳ���************************

		//int levelsNum = this->nlevels;      //����������������������� 

		//if (!do_keypoints)//�ڹؼ������ṩ������£�������������ͨ������Ĺؼ�������õ�
		//{
		//	// ****************ע�⣬����ORB/BRISK/SIFT��SURF/FREAK������*****************************
		//	// ���ڽ������Ͻ�����������������ԭͼ�Ͻ��������� ������
		//	// ��ʵ�ϣ�������Ӧ����octave�����޹أ���������ؼ���Ĳ�������Сsize�й�
		//	// �������У�����octave��صģ�����֮���������ڽ������ϰ�octave�𼶲�����������ԭʼͼ�а�size����

		//	levelsNum = 0;
		//	for (size_t i = 0; i < _keypoints.size(); i++)//�ҵ�����ؼ����е����octave����Ϊ����������
		//		levelsNum = std::max(levelsNum, std::max(_keypoints[i].octave, 0));
		//	levelsNum++;
		//}

		////*****************��Σ����ͼ���������ȡ************************

		//// Pre-compute the scale pyramids
		//vector<Mat> imagePyramid(levelsNum), maskPyramid(levelsNum);
		//for (int level = 0; level < levelsNum; ++level)
		//{
		//	float scale = 1 / getScale(level, firstLevel, scaleFactor);      //���㵱ǰ�㼶�ĳ߶�����
		//	Size sz(cvRound(image.cols*scale), cvRound(image.rows*scale)); //���߶����ӻ�õ�ǰ�㼶�µ�ͼ��ߴ�
		//	Size wholeSize(sz.width + border * 2, sz.height + border * 2);	   //���߽�ߴ���չ��ǰ�㼶�µ�ͼ��ߴ磬��ֹ����ʱ�߽�ȱʧ
		//	Mat temp(wholeSize, image.type()), masktemp;
		//	imagePyramid[level] = temp(Rect(border, border, sz.width, sz.height));//ץȡROI��(border, border)ROI��ʼ���ꣻ�ǿ�����ֻ����ͼ��ͷ����imagePyramid[level]

		//	if (!mask.empty())//�����Ĥ�ǿ�
		//	{
		//		masktemp = Mat(wholeSize, mask.type());
		//		maskPyramid[level] = masktemp(Rect(border, border, sz.width, sz.height));
		//	}

		//	// Compute the resized image
		//	if (level != firstLevel)
		//	{
		//		if (level < firstLevel) //������ִ��
		//		{
		//			resize(image, imagePyramid[level], sz, 0, 0, INTER_LINEAR);
		//			if (!mask.empty())
		//				resize(mask, maskPyramid[level], sz, 0, 0, INTER_LINEAR);
		//		}
		//		else //imagePyramid[level-1] �� imagePyramid[level]
		//		{
		//			resize(imagePyramid[level - 1], imagePyramid[level], sz, 0, 0, INTER_LINEAR);//ע��resize���÷�
		//			if (!mask.empty())
		//			{
		//				resize(maskPyramid[level - 1], maskPyramid[level], sz, 0, 0, INTER_LINEAR);
		//				threshold(maskPyramid[level], maskPyramid[level], 254, 0, THRESH_TOZERO);
		//			}
		//		}
		//		//��imagePyramid[level]��䵽temp, ����imagePyramid[level]��Χ���border�����ؿ�ı߽�
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

		////*****************�ٴΣ����㲢��ѡ�ؼ���************************

		//// Pre-compute the keypoints (we keep the best over all scales, so this has to be done beforehand
		//vector <vector<KeyPoint>> allKeypoints;
		//if (do_keypoints)
		//{
		//	// Get keypoints, those will be far enough from the border that no check will be required for the descriptor
		//	// ����ͼ�������������������е�ȫ���ؼ���
		//	computeKeyPoints(imagePyramid, maskPyramid, allKeypoints,
		//		nfeatures, firstLevel, scaleFactor,
		//		edgeThreshold, patchSize, scoreType);
		//}

		////****************�ٴΣ�У��������ͼ���и���ؼ�������*********************** 
		//_keypoints.clear();
		//for (int level = 0; level < levelsNum; ++level)
		//{
		//	// Get the features and compute their orientation
		//	vector<KeyPoint>& keypoints = allKeypoints[level];   //��õ�level��ؼ��㼰������
		//	int nkeypoints = (int)keypoints.size();

		//	// Copy to the output data   
		//	//У���ؼ������꣬�����߶ȱ任
		//	if (level != firstLevel)
		//	{
		//		float scale = getScale(level, firstLevel, scaleFactor);
		//		for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
		//			keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
		//			keypoint->pt *= scale;
		//	}
		//	// And add the keypoints to the output  //insert(const_iterator _Where, _Iter _First, _Iter _Last)
		//	// _keypoints.clear()��_keypoints.begain()��_keypoints.end()ͬλ�ã�����[)
		//	// ʹ��inset()������ÿ��level�Ĺؼ��������ӵ�_keypoints��ĩβ
		//	_keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
		//}


		////*****************��󣬿�ʼ����������************************
		//Mat descriptors;
		//if (do_descriptors)     //�ж��Ƿ����������
		//{
		//	//-------------����������������descriptors
		//	int nkeypoints = 0;
		//	for (int level = 0; level < levelsNum; ++level)
		//		nkeypoints += (int)allKeypoints[level].size();  //ͳ��ȫ���ؼ�������
		//	if (nkeypoints == 0)								//�����ڹؼ���
		//		_descriptors.release();
		//	else
		//	{
		//		_descriptors.create(nkeypoints, descriptorSize(), CV_32F); //���ؼ������������������ͣ���������������
		//		descriptors = _descriptors.getMat();   //********** ��OutputArrayת��ΪMat�ͽӿڣ�OutputArray����ΪMat�͡�vector�͵�ʹ��
		//	}

		//	//****************** 3. �������������ؼ����������***************************
		//	_keypoints.clear();
		//	for (int level = 0; level < levelsNum; ++level) //�����������ڸ���
		//	{
		//		//ȡ����ǰ��ؼ��㡢��ǰ��ͼ��
		//		vector<KeyPoint>& kpt_per_level = allKeypoints[level];
		//		Mat& img_per_level = imagePyramid[level];

		//		for (int k = 0; k < kpt_per_level.size(); k++) //�������ڸ��ؼ���
		//		{
		//			Mat desc;
		//			desc = descriptors.rowRange(k, k + 1);
		//			computeSingleDescriptor(img_per_level, kpt_per_level[k], desc, doBinaryEncode);
		//			//descriptors.push_back(desc);
		//		}
		//	}
		//}
	}

	// �ؼ�����
	void NOF::detectImpl(const Mat& _image, vector<KeyPoint>& _keypoints, const Mat& _mask) const
	{
		//*****************Ԥ���������ʼ�����ݺ�����************************

		CV_Assert(patchSize >= 2);

		//ROI handling
		const int HARRIS_BLOCK_SIZE = 9;    //Harris��Ӧ����
		int halfPatchSize = patchSize / 2;  //��������ȡ����
		int border = std::max(edgeThreshold, std::max(halfPatchSize, HARRIS_BLOCK_SIZE / 2)) + 1;//�߽�ֵȡ�������е������

		Mat image = _image, mask = _mask;
		if (image.type() != CV_8UC1)		//ת��Ϊ�Ҷ�ͼ��
			cvtColor(_image, image, CV_BGR2GRAY);

		//*****************���ȣ�ȷ��ͼ��������Ĳ���************************

		int levelsNum = this->nlevels;      //����������������������� 

		//*****************��Σ����ͼ���������ȡ************************

		// Pre-compute the scale pyramids
		vector<Mat> imagePyramid(levelsNum), maskPyramid(levelsNum);
		for (int level = 0; level < levelsNum; ++level)
		{
			float scale = 1 / getScale(level, firstLevel, scaleFactor);      //���㵱ǰ�㼶�ĳ߶�����
			Size sz(cvRound(image.cols*scale), cvRound(image.rows*scale));   //���߶����ӻ�õ�ǰ�㼶�µ�ͼ��ߴ�
			Size wholeSize(sz.width + border * 2, sz.height + border * 2);	 //���߽�ߴ���չ��ǰ�㼶�µ�ͼ��ߴ磬��ֹ����ʱ�߽�ȱʧ
			Mat temp(wholeSize, image.type()), masktemp;
			imagePyramid[level] = temp(Rect(border, border, sz.width, sz.height));//ץȡROI��(border, border)ROI��ʼ���ꣻ�ǿ�����ֻ����ͼ��ͷ����imagePyramid[level]

			if (!mask.empty())//�����Ĥ�ǿ�
			{
				masktemp = Mat(wholeSize, mask.type());
				maskPyramid[level] = masktemp(Rect(border, border, sz.width, sz.height));
			}

			// Compute the resized image
			if (level != firstLevel)
			{
				if (level < firstLevel) //������ִ��
				{
					resize(image, imagePyramid[level], sz, 0, 0, INTER_LINEAR);
					if (!mask.empty())
						resize(mask, maskPyramid[level], sz, 0, 0, INTER_LINEAR);
				}
				else //imagePyramid[level-1] �� imagePyramid[level]
				{
					resize(imagePyramid[level - 1], imagePyramid[level], sz, 0, 0, INTER_LINEAR);//ע��resize���÷�
					if (!mask.empty())
					{
						resize(maskPyramid[level - 1], maskPyramid[level], sz, 0, 0, INTER_LINEAR);
						threshold(maskPyramid[level], maskPyramid[level], 254, 0, THRESH_TOZERO);
					}
				}
				//��imagePyramid[level]��䵽temp, ����imagePyramid[level]��Χ���border�����ؿ�ı߽�
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
		//*****************�ٴΣ����㲢��ѡ����������ؼ���************************

		// Pre-compute the keypoints (we keep the best over all scales, so this has to be done beforehand
		vector <vector<KeyPoint>> allKeypoints;
		if (true)
		{
			// Get keypoints, those will be far enough from the border that no check will be required for the descriptor
			// ����ͼ�������������������и���ͼ���ϵ�ȫ���ؼ���
			computeKeyPoints(imagePyramid, maskPyramid, allKeypoints,
				nfeatures, firstLevel, scaleFactor,
				edgeThreshold, patchSize, scoreType);
		}
		//****************���У��������ͼ���и���ؼ�������*********************** 
		_keypoints.clear();
		for (int level = 0; level < levelsNum; ++level)
		{
			// Get the features and compute their orientation
			vector<KeyPoint>& keypoints = allKeypoints[level];   //��õ�level��ؼ��㼰������
			int nkeypoints = (int)keypoints.size();

			// Copy to the output data   
			//У���ؼ������꣬�����߶ȱ任
			if (level != firstLevel)
			{
				float scale = getScale(level, firstLevel, scaleFactor);
				for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
					keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
					keypoint->pt *= scale;
			}
			// And add the keypoints to the output  //insert(const_iterator _Where, _Iter _First, _Iter _Last)
			// _keypoints.clear()��_keypoints.begain()��_keypoints.end()ͬλ�ã�����[)
			// ʹ��inset()������ÿ��level�Ĺؼ��������ӵ�_keypoints��ĩβ
			_keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
		}
	}

	// ����������
	void NOF::computeImpl(const Mat& _image, vector<KeyPoint>& keypoints, Mat& descriptors) const
	{
		//1.����������ͼ�� ��2.�����������ؼ��� ��3.��������ؼ�����������1.�����ز���ģʽ ��2.��������������

		Mat image;
		if (_image.channels() == 3)
			cvtColor(_image, image, CV_BGR2GRAY);
		else
			image = _image;

		//*************** 1.����������ͼ�� *********************

		//int firstLevel = 0;													     //�������ײ�����
		int levelsNum = 0;						                                     //�������ܲ���
		//((MGHF*)this)->edgeThreshold = keypoints[0].size;                          //��������Ե��ֵ
		for (size_t i = 0; i < keypoints.size(); i++)
		{
			levelsNum = std::max(levelsNum, std::max(keypoints[i].octave, 0)); //�ҵ��ؼ����е����octave����Ϊ����������		
			//((MGHF*)this)->edgeThreshold = std::min(keypoints[i].size, (float)((MGHF*)this)->edgeThreshold); //�ҵ��ؼ����е��������뾶size,��Ϊ�߽���ֵ����ֹ����ģʽͼ����ͼ��߽�
		}
		levelsNum++;														   //������������+1

		int border = edgeThreshold + 1;   //��ԭͼ��Ȧ����border������ê�㼴�ؼ��㲻�ܳ���border���Ҳ�����Χ���ܳ���ͼ��߽磬���һ�㣬����R��border
		vector<Mat> imagePyramid;
		buildPyramidImages(image, this->firstLevel, levelsNum, this->scaleFactor, border, imagePyramid);

		//*************** 2.�������������ؼ��� *********************

		// 1.���˱�Ե�ؼ��� ���� 2.���������㼶����ؼ��� ���� 3.���߶ȱ����������Źؼ������ꡢ������Χ

		// Cluster the input keypoints depending on the level they were computed at
		vector<vector<KeyPoint>> allKeypoints;
		allKeypoints.clear();
		allKeypoints.resize(levelsNum);
		for (int i = 0; i<keypoints.size(); i++)
			allKeypoints[keypoints[i].octave].push_back(keypoints[i]);

		// Make sure we rescale the coordinates and size of keypoints  
		for (int level = 0; level < levelsNum; ++level)
		{
			if (level == firstLevel)//��һ�㲻����
				continue;

			vector<KeyPoint> & kpts = allKeypoints[level];		        //ȡ����level��Ĺؼ���
			float scale = 1 / getScale(level, firstLevel, scaleFactor); //���level��ĳ߶�����
			for (vector<KeyPoint>::iterator keypoint = kpts.begin(),
				keypointEnd = kpts.end(); keypoint != keypointEnd; ++keypoint)
			{
				//cout <<"����������"<<level<< "����ǰ�ؼ���ߴ�:" << keypoint->size << "��������:"<<scale<<endl;
				keypoint->pt *= scale;			//��������߶ȡ���������߶ȣ�*scale ����������ɽ�����ʱ�෴
				keypoint->size *= scale;        //δ����ǰ���ؼ���ߴ磬������𼶱�󣬴�31��44��������������ȫ��Ϊ31
				//keypoint->size *= scale;        //�ٴε����󣬹ؼ���ߴ磬��㼶�𼶱�С����31��25����8
				//cout << "����������"<< level << "������ؼ���ߴ�:" << keypoint->size << "��������:" << scale << endl;
			}
		}
		for (int level = 0; level < levelsNum; ++level)
		{
			vector<KeyPoint> & kpts = allKeypoints[level];		        //ȡ����level��Ĺؼ���
			// Remove keypoints very close to the border  
			KeyPointsFilter::runByImageBorder(kpts, imagePyramid[level].size(), edgeThreshold);
		}

		//****************** 3. �������������ؼ����������***************************

		for (int level = 0; level < levelsNum; ++level) //�����������ڸ���
		{
			//ȡ����ǰ��ؼ��㡢��ǰ��ͼ��
			vector<KeyPoint>& kpt_per_level = allKeypoints[level];
			Mat& img_per_level = imagePyramid[level];

			for (int k = 0; k < kpt_per_level.size(); k++) //�������ڸ��ؼ���
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