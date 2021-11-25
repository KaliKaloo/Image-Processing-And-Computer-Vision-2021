/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - load.cpp
// TOPIC: load and display an image
//
// Getting-Started-File for OpenCV
// University of Bristol
//
/////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;           //make available OpenCV namespace

int main() {

  //declare a matrix container to hold an image
  Mat image;

  //load image from a file into the container
  image = imread("mandrill.jpg", CV_LOAD_IMAGE_UNCHANGED);
  Mat dst; 
  threshold(image,dst,0, 128, THRESH_BINARY);
  imwrite("opencv-threshold-example.jpg", dst); 

  // for (int i = 0; i < image.rows; i++) {
  //   for (int j = 0; j < image.cols; j++) {
  //     int pixelValue = (int)image.at<uchar>(i,j);
  //     if (pixelValue >= 128){
  //       image.at<uchar>(i,j) = 255;
  //     }else{
  //       image.at<uchar>(i,j) = 0;
  //     }
  //   } 
  // }
  //construct a window for image display
  namedWindow("Display window", CV_WINDOW_AUTOSIZE);
  
  //visualise the loaded image in the window
  imshow("Display window", dst);

  //wait for a key press until returning from the program
  waitKey(0);

  //free memory occupied by image 
  image.release();

  return 0;
}
