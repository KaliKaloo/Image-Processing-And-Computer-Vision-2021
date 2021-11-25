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
  Mat image = imread("mandrill0.jpg", 1);

  //opencv works with BGR not RGB
  // Threshold by looping through all pixels
  for(int y=0; y<image.rows; y++) {
   for(int x=0; x<image.cols; x++) {
     uchar pixelBlue = image.at<Vec3b>(y,x)[0];
     uchar pixelGreen = image.at<Vec3b>(y,x)[1];
     uchar pixelRed = image.at<Vec3b>(y,x)[2]; 
     //noticed that what should be red is green so take the green pixels from channel [1]
     //and put it in the red channel [2] 
     
     image.at<Vec3b>(y,x)[0]=pixelRed; //set new red values in the blue channel
     image.at<Vec3b>(y,x)[1]=pixelBlue;
     image.at<Vec3b>(y,x)[2]=pixelGreen;
    } 
    }

  //Save thresholded image
  imwrite("fixMandrill0.jpg", image);

  return 0;
}