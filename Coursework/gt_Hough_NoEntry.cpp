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

void sobel(Mat &input, Mat &output_x, Mat &output_y, Mat &output_mag, Mat &output_dir);
void normalise(Mat &input, string num);
void threshold(Mat &input, int t, Mat &output);
void gaussian(Mat &input, int size, Mat &output);
void filter_non_max(Mat &input_mag, Mat &input_dir);
vector<vector<int> > hough_circles(Mat &input, int r_min, int r_max, double threshold, Mat &direction);
void draw_circles(Mat &input, vector<vector<int> > circles);

int ***malloc3dArray(int dim1, int dim2, int dim3) {
    int i, j, k;
    int ***array = (int ***) malloc(dim1 * sizeof(int **));
 
    for (i = 0; i < dim1; i++) {
        array[i] = (int **) malloc(dim2 * sizeof(int *));
	    for (j = 0; j < dim2; j++) {
  	        array[i][j] = (int *) malloc(dim3 * sizeof(int));
	    }
 
    }
    return array;
}

/** Global variables */
String cascade_name = "NoEntrycascade/cascade.xml";
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
	imwrite( "goundTruth_NoEntry_Hough/detected_NE_"+imageNum+"/gt_NE_detected"+imageNum+".jpg", image );

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
	ifstream file("goundTruth_NoEntry_Hough/noentry-ground-truths.csv");
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
	// Print nukber of true faces
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
	cout<<"[Number of detected no entry signs] " << detected_faces.size() <<endl;
} 

/** @function intersection_over_union */
float num_correctly_detected_faces(vector<Rect> truth_faces, vector<Rect> detected_faces){
	float theshold = 0.4;
	int correct_faces = 0;


	for(int i=0; i<truth_faces.size(); i++){
		Rect tf = truth_faces[i];
		float tf_x1 = tf.x + tf.width;
		float tf_y1 = tf.y + tf.height;
		float max_iou = 0;

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
	if(truth_faces.size() > 0) return correct_faces/float(truth_faces.size());
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



void sobel(Mat &input, Mat &output_x, Mat &output_y, Mat &output_mag, Mat &output_dir) {
    output_x.create(input.size(), CV_32FC1);
    output_y.create(input.size(), CV_32FC1);
    output_mag.create(input.size(), CV_32FC1);
    output_dir.create(input.size(), CV_32FC1);

	//make kerenls
	cv::Mat kX = (cv::Mat_<float>(3,3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
	cv::Mat kY = (cv::Mat_<float>(3,3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

	int kernelRadiusX = ( kX.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kX.size[1] - 1 ) / 2;

	Mat paddedInput;
	copyMakeBorder( input, paddedInput, kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY, BORDER_REPLICATE );

	for ( int i = 0; i < input.rows; i++ ) {	
		for( int j = 0; j < input.cols; j++ ) {
			float sum_x = 0.0;
			float sum_y = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ ) {
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ ) {
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					float imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					float kernel_x = kX.at<float>( kernelx, kernely );
					float kernel_y = kY.at<float>( kernelx, kernely );

					sum_x += imageval * kernel_x;
					sum_y += imageval * kernel_y;
				}
			}
			output_x.at<float>(i, j) = (float) sum_x;
			output_y.at<float>(i, j) = (float) sum_y;
			output_mag.at<float>(i, j) = (float) sqrt((sum_y*sum_y) + (sum_x*sum_x));
			output_dir.at<float>(i, j) = (float) atan2(sum_y,sum_x);
		}
	}
}

void gaussian(Mat &input, int size, Mat &output)
{
	output.create(input.size(), input.type());

	cv::Mat kX = cv::getGaussianKernel(size, -1);
	cv::Mat kY = cv::getGaussianKernel(size, -1);

	// make it 2D multiply one by the transpose of the other
	cv::Mat kernel = kX * kY.t();

	int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
	int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput, 
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE );

	for ( int i = 0; i < input.rows; i++ ) {	
		for( int j = 0; j < input.cols; j++ ) {
			double sum = 0.0;
			for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ ) {
				for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ ) {
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;
					// get the values from the padded image and the kernel
					int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
					double kernalval = kernel.at<double>( kernelx, kernely );
					// do the multiplication
					sum += imageval * kernalval;
				}
			}
			// set the output value as the sum of the convolution
			output.at<uchar>(i, j) = (uchar) sum;
		}
	}
}


void threshold(Mat &input, int threshold, Mat &output) {
	if(threshold >= 0 && threshold <= 255){
		output.create(input.size(), input.type());

		for(int i = 0; i < input.rows; i++) {
			for(int j = 0; j < input.cols; j++) {
				int pixel = (int) input.at<uchar>(i, j);
				if(pixel > threshold) {
					output.at<uchar>(i,j) = (uchar) 255;
				} else {
					output.at<uchar>(i,j) = (uchar) 0;
				}
			}
		}
		imwrite("coin_threshold.jpg", output);
	}
}

vector<vector<int> > hough_circles(Mat &input, int r_min, int r_max, double threshold, Mat &direction) {

	int ***hough_space = malloc3dArray(input.rows, input.cols, r_max);

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            for (int r = 0; r < r_max; r++) {
                hough_space[i][j][r] = 0;
            }
        }
    }
	cout << "Created array" << endl;

	//-- Make the hough space -- //
    for (int x = 0; x < input.rows; x++) {
        for (int y = 0; y < input.cols; y++) {
			if(input.at<uchar>(x,y) == 255) {
				for (int r = 0; r < r_max; r++) {
					int xc = int(r * sin(direction.at<float>(x,y)));
					int yc = int(r * cos(direction.at<float>(x,y)));

					int a = x - xc;
					int b = y - yc;
					if(a >= 0 && a < input.rows && b >= 0 && b < input.cols) {
						hough_space[a][b][r] ++;
					}

					int c = x + xc;
					int d = y + yc;
					if(c >= 0 && c < input.rows && d >= 0 && d < input.cols) {
						hough_space[c][d][r] ++;
					}
				}
			}
        }
    }

	Mat hough_output(input.rows, input.cols, CV_32FC1);
    for (int x = 0; x < input.rows; x++) {
        for (int y = 0; y < input.cols; y++) {
            for (int r = r_min; r < r_max; r++) {
                hough_output.at<float>(x,y) += hough_space[x][y][r];
            }
 
        }
    }

    imwrite( "hough_space.jpg", hough_output );

	//--- get the circles ---//
	vector<vector<int> > circles;
	for (int x = 0; x < input.rows; x++) {
        for (int y = 0; y < input.cols; y++) {
			bool test_pass = true;
		
			int max_r = 0;
			int currentMax = 0;
			for (int r = r_min; r < r_max; r++) {
				if ( hough_space[x][y][r] > currentMax) {
					currentMax = hough_space[x][y][r];
					max_r = r;
				}	
			}
				
			for(int i = 0; i < circles.size(); i++) {
				vector<int> circle = circles[i];
				int xs = circle[0];
				int ys = circle[1];
				int rs = circle[2];

				//equation of a circle (x'-x)^2+(y'-y)^2 = r^2 were x & y are center
				if(!(pow((xs-x),2) + pow((ys-y),2) > pow(rs,2))) {
					test_pass = false;
				}
			}
			if(hough_space[x][y][max_r] > threshold && test_pass) {
				vector<int> circle;
				circle.push_back(x);
				circle.push_back(y);
				circle.push_back(max_r);
				circles.push_back(circle);
			}
        }
    }

	cout << "Number of circles: " << circles.size() << endl;

	return circles;
}

void draw_circles(Mat &input, vector<vector<int> > circles) {

	for(int i = 0; i < circles.size(); i++) {
		vector<int> c = circles[i];
		Point center = Point(c[1], c[0]);
		circle(input, center, 1, Scalar(0, 255, 0), 3, 8, 0);
		int radius = c[2];
		circle(input, center, radius, Scalar(0, 0, 255), 2, 8, 0);
	}

	stringstream ss;
	ss << (int) circles.size();
	imwrite("detected_circles_"+ss.str()+".jpg", input);

}