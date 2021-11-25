/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - thr.cpp
// TOPIC: RGB explicit thresholding
//
// Getting-Started-File for OpenCV
// University of Bristol
//
/////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;
cv::Mat convolute( Mat image, Mat kernel){
	Mat newImage = imread("mandrill.jpg",1);		
	for(int x=0; x< image.rows; x++){
		for (int y=0; y< image.cols; y++){

			Vec3f ret = Vec3b(0,0,0);
			for(int r=0; r<kernel.rows; r++){
				for (int c=0; c< kernel.cols; c++){
					int m = r-1;
					int n = c-1;
					float h = kernel.at<float>(r,c);

					//ignore where e.g. x-m is outside image
					if((x-n)<0 || (y-m)<0 || (x-n) >image.cols || (y-m) > image.rows){
						continue;
					}

					ret += image.at<Vec3b>(x-m, y-n)*h;
				}
			}
			newImage.at<Vec3b>(x, y)=ret;
		}
	}
	return newImage;
}


int main() { 
	// Read image from file
  	Mat image = imread("mandrill.jpg", 1);
  	Mat newImage = imread("mandrill.jpg",1);

	//MY CONVOLUTION FILTER
	float mykernel[] = {1,1,1,1,1,1,1,1,1};
	Mat kernel(3, 3, CV_32F, mykernel);
	kernel = kernel/9;
	newImage = convolute(image, kernel);

	//INBUILD CONVOLUTION FILTER
	// filter2D(image, newImage, -1, kernel, Point(-1,-1),9, 4);


  	//Save thresholded image
  	imwrite("convNewMandrill.jpg", newImage);
  	return 0;	
}

// Vec3b convolute( Mat image, int x, int y, Mat kernel){
// 	Vec3f ret = Vec3b(0,0,0);
// 	for(int r=0; r<kernel.rows; r++){
// 		for (int c=0; c< kernel.cols; c++){
// 			int m = r-1;
// 			int n = c-1;
// 			float h = kernel.at<float>(r,c);

// 			//ignore where e.g. x-m is outside image
// 			if((x-n)<0 || (y-m)<0 || (x-n) >image.cols || (y-m) > image.rows){
// 				continue;
// 			}

// 			ret += image.at<Vec3b>(x-m, y-n)*h;
// 		}
// 	}
// 	return ret;
// }

// int main() { 
// 	// Read image from file
//   	Mat image = imread("mandrill.jpg", 1);
//   	Mat newImage = imread("mandrill.jpg",1);

// 	//MY CONVOLUTION FILTER
// 	float mykernel[] = {1,1,1,1,1,1,1,1,1};
// 	Mat kernel(3, 3, CV_32F, mykernel);
// 	kernel = kernel/9;
//    	// set the output value as the sum of the convolution
// 	for(int r=0; r< image.rows; r++){
// 		for (int c=0; c< image.cols; c++){
//    			newImage.at<Vec3b>(r, c) = convolute(image, r, c, kernel);
// 		}
// 	}

// 	//INBUILD CONVOLUTION FILTER
// 	// filter2D(image, newImage, -1, kernel, Point(-1,-1),9, 4);


//   	//Save thresholded image
//   	imwrite("convMandrill.jpg", newImage);

//   	return 0;	
// }