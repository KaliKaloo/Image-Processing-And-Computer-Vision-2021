/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <sstream>
#include <fstream>
#include <iostream>
#include <stdio.h>


using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat image);
vector<std::string> splitString(string &line, char delimiter);
void draw_rect (Mat image, Rect r, Scalar c);
void draw_face_truths (string imageNum, Mat image);

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;	

/** @function main */
int main( int argc, const char** argv )
{
	string imageNum = argv[1];

    //    // 1. Read Input Image
	Mat image = imread("No_entry/NoEntry" +imageNum+ ".bmp", CV_LOAD_IMAGE_COLOR);

	// // 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// // 3. Detect Faces and Display Result
	draw_face_truths(imageNum,image);
	detectAndDisplay(image);

	// // 4. Save Result Image
	imwrite( "face_groundTruth/gt_detected"+imageNum+".jpg", image );

	return 0;
}

/** @function splitString */
vector<std::string> splitString(string &line, char delimiter) {
	std::vector<std::string> tokens;
	int position;
	while ((position = line.find(delimiter)) != std::string::npos) {
		tokens.push_back(line.substr(0, position));
		line.erase(0, position + 1);
	}
	tokens.push_back(line); 
	return tokens;
}

/** @function draw_rect */
void draw_rect (Mat image, Rect r, Scalar c){
		rectangle(image, Point(r.x, r.y), Point(r.x + r.width, r.y + r.height), c, 2);
}

/** @function draw_face_truths */
void draw_face_truths (string imageNum, Mat image){
	vector<Rect> face_truths;
	string file_name = "face_truths/faces-ground-truths.csv";
	ifstream file("face_groundTruth/faces-ground-truths.csv");
	string line;

	while(getline(file, line)) {
		vector<string> tokens = splitString(line, ',');
		if((tokens[0]=="NoEntry"+imageNum+".jpg")&&(tokens[0]==tokens[0])){
			//RECT(height, width, x, y) try swapping h and w if not working
			Rect r = Rect(atoi(tokens[1].c_str()),atoi(tokens[2].c_str()),atoi(tokens[3].c_str()),atoi(tokens[4].c_str()));
			face_truths.push_back(r);
		}
	}
	file.close();

	// Draw ground truth faces
	for( int i = 0; i < face_truths.size(); i++ ){
		draw_rect(image, face_truths[i], Scalar(0,0,255));
	}
	// Print nukber of true faces
	cout<<"[Number of true faces] " << face_truths.size() <<endl;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat image )
{
	std::vector<Rect> faces;
	Mat image_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( image, image_gray, CV_BGR2GRAY );
	equalizeHist( image_gray, image_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( image_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(10, 10), Size(300,300) );

    // 3. Draw box around faces found from viola jones
	for( int i = 0; i < faces.size(); i++ )
	{
		draw_rect(image, faces[i], Scalar(0,255,0));
	}

	// 4. Print number of faces detected
	cout<<"[Number of detected faces] " << faces.size() <<endl;
}
 

