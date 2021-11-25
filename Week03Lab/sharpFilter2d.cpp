// header inclusion
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

int main( int argc, char** argv )
{

	// LOADING THE IMAGE
	char* imageName = argv[1];

	Mat image;
	image = imread( imageName, 1 );
	Mat newImage;
	// newImage = imread( imageName, 1 );

	if( argc != 2 || !image.data )
	{
	printf( " No image data \n " );
	return -1;
	}

	// CONVERT COLOUR, BLUR AND SAVE
	Mat gray_image;
	cvtColor( image, gray_image, CV_BGR2GRAY );

	Mat carBlurred;
	// GaussianBlur(gray_image,23,carBlurred);
	cv::GaussianBlur(gray_image, newImage, cv::Size(0,0), 25);
	cv::addWeighted(gray_image, 1.7, newImage, -0.8, 0, newImage);

	imwrite( "car2DSharper.jpg", newImage );
	// filter2D(image, newImage, -1, kernel, Point(-1,-1),9, 4);

	return 0;
}
