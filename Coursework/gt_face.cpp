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
void detectAndDisplay( Mat image, vector<Rect> &detected_faces);
vector<std::string> splitString(string &line, char delimiter);
void draw_rect (Mat image, Rect r, Scalar c);
void draw_truth_faces (string imageNum, Mat image, vector<Rect> &truth_faces);
float num_correctly_detected_faces(vector<Rect> truth_faces, vector<Rect> detected_faces);
float get_true_positive_rate(int correct_faces, vector<Rect> truth_faces);
float get_f1_score(float true_positive, float false_positive, float false_negative);


/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;	

/** @function main */
int main( int argc, const char** argv )
{
	string imageNum = argv[1];

	// MAKE SURE TO CHANGE THE WAY YOU READ IN THE FILE NAME!!!!!!
    // 1. Read Input Image
	Mat image = imread("No_entry/NoEntry" +imageNum+ ".bmp", CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
	
	cout <<" "<<endl;
	
	// 3. Detect Faces and Display Result
	vector<Rect> truth_faces;
	draw_truth_faces(imageNum,image,truth_faces);
	
	vector<Rect> detected_faces;
	detectAndDisplay(image, detected_faces);

	// 4. Save Result Image
	imwrite( "groundTruth_faces/gt_detected"+imageNum+".jpg", image );

	float correct_faces = num_correctly_detected_faces(truth_faces, detected_faces);
	float tpr = get_true_positive_rate(correct_faces, truth_faces);
	float false_positive = detected_faces.size() - correct_faces;
	float false_negative = truth_faces.size() - correct_faces;
	float f1_score = get_f1_score(correct_faces, false_positive, false_negative);


	cout<< "[Correctly identified faces] " <<correct_faces <<endl;
	cout<< "[True positive rate] " <<tpr <<endl;
	cout<< "[F1-Score] " <<f1_score <<endl;
	cout <<" "<<endl;

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
void draw_truth_faces (string imageNum, Mat image, vector<Rect> &truth_faces){
	ifstream file("groundTruth_faces/faces-ground-truths.csv");
	string line;

	while(getline(file, line)) {
		vector<string> tokens = splitString(line, ',');
		
		// MAKE SURE TO CHANGE THE WAY YOU READ IN THE FILE NAME!!!!!!

		if((tokens[0]=="NoEntry"+imageNum+".jpg")&&(tokens[0]==tokens[0])){
			//RECT(height, width, x, y) try swapping h and w if not working
			Rect r = Rect(atoi(tokens[1].c_str()),atoi(tokens[2].c_str()),atoi(tokens[3].c_str()),atoi(tokens[4].c_str()));
			truth_faces.push_back(r);
		}
	}
	file.close();

	// Draw ground truth faces
	for( int i = 0; i < truth_faces.size(); i++ ){
		draw_rect(image, truth_faces[i], Scalar(0,0,255));
	}
	// Print number of true faces
	cout<<"[Number of true faces] " << truth_faces.size() <<endl;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat image , vector<Rect> &detected_faces)
{
	Mat image_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( image, image_gray, CV_BGR2GRAY );
	equalizeHist( image_gray, image_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( image_gray, detected_faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(10, 10), Size(300,300) );

    // 3. Draw box around faces found from viola jones
	for( int i = 0; i < detected_faces.size(); i++ )
	{
		draw_rect(image, detected_faces[i], Scalar(0,255,0));
	}

	// 4. Print number of faces detected
	cout<<"[Number of detected faces] " << detected_faces.size() <<endl;
} 

/** @function intersection_over_union */
float num_correctly_detected_faces(vector<Rect> truth_faces, vector<Rect> detected_faces){
	float theshold = 0.6;
	float max_iou = 0;
	int correct_faces = 0;

	for(int i=0; i<truth_faces.size(); i++){
		Rect tf = truth_faces[i];
		float tf_x1 = tf.x + tf.width;
		float tf_y1 = tf.y + tf.height;

		for(int j=0; j<detected_faces.size(); j++){
			Rect df = detected_faces[j];

			// calculate intersection over union
			float df_x1 = df.x + df.width;
			float df_y1 = df.y + df.height;


			float xDiff = min(df_x1, tf_x1) - max(df.x,tf.x);
			float yDiff = min(df_y1, tf_y1) - max(df.y,tf.y);

			if (xDiff <=0 or yDiff <=0) {
				continue;
			}
			else{
				float intersect_area = xDiff *yDiff;
				float union_area = (df.width*df.height) + (tf.width*tf.height) - intersect_area;
				
				float iou = intersect_area/union_area;
				if( iou > max_iou) {
					max_iou = iou;
				}
			}
		}
		if(max_iou > theshold) correct_faces++;
	}
return correct_faces;
}

float get_true_positive_rate(int correct_faces, vector<Rect> truth_faces){
	if(truth_faces.size() > 0) return correct_faces/truth_faces.size();
	else{
		cout << "No true faces" << endl;
		return 0;
	}
}

float get_f1_score(float true_positive, float false_positive, float false_negative){
	if(true_positive == 0 && false_positive == 0 && false_negative == 0){
		return 0;
	}else{
		return (true_positive/(true_positive+0.5*(false_positive+false_negative)));
	}
}