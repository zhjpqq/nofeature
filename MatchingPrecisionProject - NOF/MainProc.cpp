#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include "PrecisionEvaluator.h"

using namespace std;
using namespace cv;
 
int main(int argc, char* argv[])
{ 
	string detectorName = "ORB";
	string descriptorName = "ORB"; 
	string matcherName = "BruteForce";
	
	RunEvaluator(detectorName,descriptorName,matcherName);

	return 0;
}
 