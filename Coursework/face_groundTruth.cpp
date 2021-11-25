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
#include <algorithm>
#include <string>


using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame, vector<Rect> face_t_pos);

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;

vector<string> ssplit(const string str, string del =",")
{
	vector<string> result;
	char* a = const_cast<char*>(str.c_str());
	char* current;

	current = strtok(a, del.c_str());

	while(current!= NULL){
		result.push_back(current);
		current = strtok(NULL, del.c_str());
	}

	return result;
}	

vector<std::string> splitString(const std::string &line, char delimiter) {
	std::vector<std::string> tokens;
	string theLine = line;

	int position;
	while ((position = theLine.find(delimiter)) != std::string::npos) {
		tokens.push_back(theLine.substr(0, position));
		theLine.erase(0, position + 1);
	}
	tokens.push_back(theLine);
	return tokens;
}

vector<Rect> readCSV (string imageNum){
	vector<Rect> truths;
	string file_name = "face_truths/faces-ground-truths.csv";
	ifstream file("face_groundTruth/faces-ground-truths.csv");
	string line;

	while(getline(file, line)) {
		vector<string> tokens = splitString(line, ',');
		if((tokens[0]=="NoEntry"+imageNum+".jpg")&&(tokens[0]==tokens[0])){
			//RECT(height, width, x, y) try swapping h and w if not working
			Rect r = Rect(atoi(tokens[1].c_str()),atoi(tokens[2].c_str()),atoi(tokens[3].c_str()),atoi(tokens[4].c_str()));
			truths.push_back(r);
		}
	}
	file.close();
	return truths;
}

/** @function main */
int main( int argc, const char** argv )
{
	string imageNum = argv[1];

    //    // 1. Read Input Image
	Mat frame = imread("No_entry/NoEntry" +imageNum+ ".bmp", CV_LOAD_IMAGE_COLOR);

	// // 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// // 3. Detect Faces and Display Result
	vector<Rect> face_t_pos = readCSV(imageNum);
	detectAndDisplay( frame, face_t_pos);

	// // 4. Save Result Image
	imwrite( "face_groundTruth/gt_detected"+imageNum+".jpg", frame );

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame , vector<Rect> face_t_pos)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(10, 10), Size(300,300) );

    // 3. Draw box around faces found from viola jones
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0), 2);
	}

	// 4. Draw ground truth faces
	for( int i = 0; i < face_t_pos.size(); i++ )
	{
		rectangle(frame, Point(face_t_pos[i].x, face_t_pos[i].y), Point(face_t_pos[i].x + face_t_pos[i].width, face_t_pos[i].y + face_t_pos[i].height), Scalar( 0, 0, 255 ), 2);
		
	}
 
	// 5. Print number of faces
	cout<<"[Number of faces (ground truth)] " << face_t_pos.size() <<endl;
	cout<<"[Number of faces detected] " << faces.size() <<endl;

}
