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

int main() { 

  // Read image from file
  Mat image = imread("mandrill3.jpg", 1);
  Mat newImage = imread("mandrill3.jpg",1);
  
  cvtColor( image, newImage, CV_HSV2BGR );

  //Save thresholded image
  imwrite("fixMandrill3.jpg", newImage);

  return 0;
}