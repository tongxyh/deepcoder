#ifndef FUNC_H
#define FUNC_H 
#include <iostream>
//#include <Eigen/Dense>
#include <vector>
#include <string>
#include "main.h"
#include <cmath>
#include <fstream>
#include <sstream>
#include <bitset>
#include <assert.h>

#include "opencv2/opencv.hpp"


using namespace cv;
using namespace std;
//using Eigen::MatrixXd;


extern Table<string> table_0,table_1,table_2,table_3,table_r,table_z;
extern int zg_table[16][2];



int readtable(ifstream & fin,const string key,Table<string> &tb);
string enc_cavlc_unit(vector<int> data,unsigned int nL,unsigned int nU);
vector<int>  dec_cavlc_unit(string& bit,uint32_t& idx,unsigned int nL,unsigned int nU);
vector<int>  dct_zigzag(Mat &mb);
Mat  rev_dct_zigzag(vector<int>& data);
void mat2d_show(Mat &m);
double getPSNR(const Mat& I1, const Mat& I2);


















#endif
