#include "opencv2/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <bits/stdc++.h>
using namespace cv;

// Masques
Mat Sx = (Mat_<float>(3,3) << 1, 0, -1, 2, 0, -2, 1, 0, -1) / 4;
Mat Sy = (Mat_<float>(3,3) << 1, 2, 1, 0, 0, 0, -1, -2, -1) / 4;
Mat L = (Mat_<float>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
Mat d0 = (Mat_<float>(3, 3) << 0, 0, 0, 0, 1, 0, 0, 0, 0 );
Mat m = (Mat_<float>(3,3) << 1, 2, 1, 2, 4, 2, 1, 2, 1) / 16;


Mat filtreM( Mat input, Mat M, int offset = 0 ) {
    Mat output = input.clone();

    //CHAQUE PIXEL
    for ( int row = 0; row < input.rows; row++ ) {
        for ( int col = 0; col < input.cols; col++ ) {
            float sum = 0;

            //CHAQUE PIXEL DE LA MATRICE
            for ( int m_row = 0; m_row < M.rows; m_row++ ) {
                for ( int m_col = 0; m_col < M.cols; m_col++ ) {
                    int x = row + m_row - M.rows / 2;
                    int y = col + m_col - M.cols / 2;

                    //SI LE PIXEL EST DANS L'IMAGE
                    if ( x >= 0 && x < input.rows && y >= 0 && y < input.cols ) {
                        sum += input.at<uchar>(x, y) * M.at<float>(m_row, m_col);
                    }
                }
            }
            output.at<uchar>( row, col ) = sum + offset;
        }
    }
    return output;
}

Mat euclidianDistance(Mat m1, Mat m2){
    Mat output = m1.clone();
    for ( int row = 0; row < m1.rows; row++ ) {
        for ( int col = 0; col < m1.cols; col++ ) {
            float sum = 0;
            sum = sqrt(pow(m1.at<uchar>(row, col) - 128, 2) + pow(m2.at<uchar>(row, col) - 128, 2));
            output.at<uchar>( row, col ) = sum;
        }
    }
    return output;
}

Mat gradient(Mat m){
    Mat mx = filtreM( m, Sx, 128 );
    Mat my = filtreM( m, Sy, 128 );
    return euclidianDistance(mx, my);
}

Mat MarrHildreth(Mat G,Mat R, int T){

    Mat output(G.rows, G.cols, CV_8UC1, Scalar(255));
    //pour chaque pixel
    for(int row = 0; row < G.rows; row++){
        for(int col = 0; col < G.cols; col++){
            if(G.at<uchar>(row, col) > T){
                int signPixel = R.at<uchar>(row,col) > 128 ? 1 : -1;

                //pour chaque pixel autour
                for(int m_row = 0; m_row < 3; m_row++){
                    for(int m_col = 0; m_col < 3; m_col++){
                        int x = row + m_row - 1;
                        int y = col + m_col - 1;
                        if ( x >= 0 && x < G.rows && y >= 0 && y < G.cols ){
                            if(R.at<uchar>(x, y) * signPixel < 128 * signPixel)
                                output.at<uchar>(row, col) = 0;
                        }
                    }
                }
            }
        }
    }
    return output;
}

float rand01(){
    return rand()/(double)RAND_MAX;
}

Mat esquisse(Mat m){
    Mat R  = filtreM(m,L, 128);
    Mat G = gradient(m);
    int T = getTrackbarPos("T", "TP2");
    Mat C = MarrHildreth(G, R, T);

    int t = getTrackbarPos("t", "TP2");
    int l = getTrackbarPos("l", "TP2");

    Mat Ix = filtreM( m, Sx, 128 );
    Mat Iy = filtreM( m, Sy, 128 );

    Mat output(m.rows, m.cols, CV_8UC1, Scalar(255));

    for(int row = 0; row < C.rows; row++){
        for(int col = 0; col < C.cols; col++){
            if(C.at<uchar>(row, col) == 0){
                if(rand01() < t /100.0){
                    float teta = atan2(-(Ix.at<uchar>(row, col) - 128), Iy.at<uchar>(row, col) -128)  + 0.02 * ( rand01() - 0.5 );
                    float lp = m.at<uchar>(row, col)/255.0 * l/ 100.0;
                    float x_start = col + lp * cos(teta);
                    float y_start = row + lp * sin(teta);
                    float x_end = col - lp * cos(teta);
                    float y_end = row - lp * sin(teta);
                    cv::line(output, Point(x_start, y_start), Point(x_end, y_end), Scalar(0), 1);
                }
            }
        }
    }
    return output;
}

int main( int argc, char* argv[])
{
    if ( argc != 2 ) {
        std::cout << "usage: " << argv[0] << " <input/\"video\">\n";
        return -1;
    }
    std::string input = argv[1];

    namedWindow("TP2");

    Mat inputM;
    VideoCapture cap;
    Mat inputSave;
    
    if (input.compare("video") == 0) {
        cap = VideoCapture(0);
        if (!cap.isOpened()) {
            std::cout << "Error opening video stream or file" << std::endl;
            return -1;
        }
    } else {
        inputM = imread( input );
        if ( inputM.channels() == 3 )
            cv::cvtColor( inputM, inputM, COLOR_BGR2GRAY );
        inputSave = inputM.clone(); 
    }

    int alpha = 60;
    createTrackbar( "alpha", "TP2", nullptr, 255,  NULL);
    setTrackbarPos( "alpha", "TP2", alpha );

    createTrackbar( "T", "TP2", nullptr, 100, NULL);
    setTrackbarPos( "T", "TP2", 20 );

    createTrackbar( "t", "TP2", nullptr, 100, NULL);
    setTrackbarPos( "t", "TP2", 35 );

    createTrackbar( "l", "TP2", nullptr, 1000, NULL);
    setTrackbarPos( "l", "TP2", 300 );

    bool moyenagePerso = false;
    bool moyenageOpenCV = false;
    bool rehaussement = false;
    bool derivX = false;
    bool derivY = false;
    bool grad = false;
    bool marr = false;
    bool esquisseBool = false;
    
    while ( true ) {
        if (input.compare("video") == 0) {
            Mat frame;
            cap >> frame;
            cv::cvtColor( frame, inputM, COLOR_BGR2GRAY );
        }
        int keycode = waitKey( 50 );
        int asciicode = keycode & 0xff;
        if ( asciicode == 'q' ) break;
        else if ( asciicode == 'a' ) {
            moyenagePerso = !moyenagePerso;
        }
        else if ( asciicode == 'm' ) {
            moyenageOpenCV = !moyenageOpenCV;
        }
        else if ( asciicode == 's' ) {
            rehaussement = !rehaussement;
        }
        else if (asciicode == 'x'){
            derivX = !derivX;
        }
        else if (asciicode == 'y'){
            derivY = !derivY;
        }
        else if (asciicode == 'g'){
            grad = !grad;
        }
        else if (asciicode == 'c'){
            marr = !marr;
        }
        else if (asciicode == 'e'){
            esquisseBool = !esquisseBool;
        }
        else if (asciicode == 'r' && input.compare("video")){
            inputM = inputSave.clone();
        }

        if ( moyenagePerso ) {
            inputM = filtreM( inputM, m);
        }
        if ( moyenageOpenCV ) {
            blur( inputM, inputM, Size( 3, 3 ) );
        }
        if ( rehaussement ) {
            alpha = getTrackbarPos( "alpha", "TP2" ) / 255 * 100;
            Mat R = d0 - alpha*L;
            inputM = filtreM( inputM, R );
        }
        if ( derivX ) {
            inputM = filtreM( inputM, Sx, 128 );
        }
        if ( derivY ) {
            inputM = filtreM( inputM, Sy, 128 );
        }
        if ( grad ) {
            inputM = gradient(inputM);
        }
        if ( marr ) {
            Mat R  = filtreM(inputM,L, 128);

            //calcule de G
            Mat G = gradient(inputM);
            int T = getTrackbarPos("T", "TP2");
            inputM = MarrHildreth(G, R, T);
        }
        if ( esquisseBool ) {
            inputM = esquisse(inputM);
        }
        
        if ( input.compare("video")) {
            // reset all bools
            moyenagePerso = false;
            moyenageOpenCV = false;
            rehaussement = false;
            derivX = false;
            derivY = false;
            grad = false;
            marr = false;
            esquisseBool = false;
        }

        imshow( "TP2", inputM );
    }
    imwrite("result-" + input + ".png", inputM );
}