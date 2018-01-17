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
#include "func.h"

#include "opencv2/opencv.hpp"


using namespace cv;
using namespace std;
//using Eigen::MatrixXd;

Table<string> table_0,table_1,table_2,table_3,table_r,table_z;
//int zg_table[16][2]={0,0,0,1,1,0,2,0,1,1,0,2,0,3,1,2,2,1,3,0,3,1,2,2,1,3,2,3,3,2,3,3};
int zg_table[16][2]={0,0,0,1,0,2,0,3,1,0,1,1,1,2,1,3,2,0,2,1,2,2,2,3,3,0,3,1,3,2,3,3};
float dc[4][4]={{1,1,1,1},{2,1,-1,-2},{1,-1,-1,1},{1,-2,2,-1}};
//float di[4][4]={{1,1,1,1},{1,0.5,-0.5,-1},{1,-1,-1,1},{0.5,-1,1,-0.5}};

Mat dct_c(4,4,CV_32FC1,dc);
//Mat dct_i(4,4,CV_32FC1,di);

int codeframe(Mat& img,string out_file)
{


	/*
	 * 处理符号位
	Mat residual=img_o_32-img_b_32;

	Mat img(residual.size(),CV_32S);

	for(int i=0;i<residual.rows;++i)
	{
		for(int j=0;j<residual.cols;++j)
		{
			if(*residual.ptr<int>(i,j)<0)
				*img.ptr<int>(i,j)=0;
			else 
				*img.ptr<int>(i,j)=1;
					
		}
	}

	*/

	/*
	 * 对整副图像做dct变换
	Mat dct_img(img.size(),CV_64FC1);
	dct(Mat_<double>(img),dct_img);
	img=Mat_<int>(dct_img);
	*/
//	Mat img(512,512,CV_32SC1,Scalar(1));
//	cout<<"ori data"<<endl;
//	mat2d_show(img);
	//namedWindow( "显示图片", CV_WINDOW_AUTOSIZE  ); 
//	imshow("img",img);
//	waitKey(0);


	
	int im_rows=img.rows;
	int im_cols=img.cols;
	const int data_size=4;

	string bit;
	for(int i=0;i<im_rows;i=i+data_size)
	{
			Range row_range(i,i+data_size);
		for(int j=0;j<im_cols;j=j+data_size)
		{
			Range col_range(j,j+data_size);

			Mat MacroBlock(img,row_range,col_range);

			vector<int> data=dct_zigzag(MacroBlock);
			bit+=enc_cavlc_unit(data,2,2);

		}

	}


//	cout<<"encode data: "<<bit<<endl;
	string bitsave=bit;
	ofstream out(out_file,ios::binary);
	auto btsz=bit.size();
	for(uint32_t i=0;i<bit.size();i+=8)
	{
		bitset<8> temp(0);
		for(int j=0;j<8&&j+i<bit.size();++j)
		{
			if(bit[i+j]=='0')
				temp.set(j,0);
			else 
				temp.set(j);
		}
		out.write((char*)&temp,1);
		
	}


	out.close();
	
	bit.clear();
	
	ifstream in(out_file,ios::binary);

	uint8_t in_temp;
	while(in.read((char*)&in_temp,1))
	{
		for(int j=0;j<8;++j)
		{
			bool flag=(in_temp>>j)&1;
			if(flag)
			{
				bit+="1";
			}
			else 
				bit+="0";

		}
	}

	in.close();

	if(bit.size()>btsz)
	{
		bit=bit.substr(0,btsz);
	}

	if(bit==bitsave)
		cout<<"bit equal bitori"<<endl;
	else 
		return 0;

	
	Mat dec_img(im_rows,im_cols,CV_32SC1);


	uint32_t m=0;
	while(m<bit.size())
	{
		for(int i=0;i<im_rows;i+=data_size)
		{
			for(int j=0;j<im_cols;j+=data_size)
			{
				auto dec_data=dec_cavlc_unit(bit,m,2,2);

				Mat tmp=rev_dct_zigzag(dec_data);


				for(int ii=0;ii<4;++ii){
					for(int jj=0;jj<4;++jj)
					{
						*(dec_img.ptr<int>(i+ii,j+jj))=*(tmp.ptr<int>(ii,jj));
					}
				}

			}
		}

	}

	/*
	 * 图像的反dct变换
	Mat idct_img(dec_img.size(),CV_64FC1);
	idct(Mat_<double>(dec_img),idct_img);
	dec_img=Mat_<int>(idct_img);
	*/
//	Mat tmp=img-dec_img; 
	cout<<"psnr is "<<getPSNR(Mat_<uint8_t> (img),Mat_<uint8_t>(dec_img))<<endl;
	//mat2d_show(dec_img);
//    mat2d_show(tmp);

	return 0;

} 

int main(int argc,char** argv)
{

	if(argc!=4)
		return -1;

	ifstream fin("table");
	if(!fin)
		{cout<<"no such file"<<endl;return -1;}

	readtable(fin,"table_coeff0",table_0);
	readtable(fin,"table_coeff1",table_1);
	readtable(fin,"table_coeff2",table_2);
	readtable(fin,"table_coeff3",table_3);
	readtable(fin,"table_run",table_r);
	readtable(fin,"table_zeros",table_z);
	fin.close();

	const string  img_ori(argv[1]);
//	const string  img_back="/home/dong/Documents/MATLAB/ml_v7/8bit_img/ori/1-1.bmp";
	const string  img_back(argv[2]);

	const string  out_file(argv[3]);
	const int channel=8;
	vector<Mat> spec_cube;
	for(int i=1;i<=channel;++i)
	{
		string img1_name=img_ori+to_string(i)+".bmp";
		string img2_name=img_back+to_string(i)+".bmp";
		Mat img_o =imread(img1_name,IMREAD_UNCHANGED);
		if(img_o.empty())
		{
			cerr<<"can not load img "<<img_ori<<endl;
			return -1;
		}
		Mat img_b =imread(img2_name,IMREAD_UNCHANGED);
	
		if(img_b.empty())
		{
			cerr<<"can not load img "<<img_back<<endl;
			return -1;
		}
		

		Mat img_o_32,img_b_32;
		img_o.convertTo(img_o_32,CV_32SC1);
		img_b.convertTo(img_b_32,CV_32SC1);
		Mat residual=img_o_32-img_b_32;
		spec_cube.push_back(residual);

	}



	for(int i=0;i<channel;++i)
	{
		string out_name=out_file+to_string(i+1)+".dat";
		if(i==0)
			codeframe(spec_cube[i],out_name);
		else
		{
			Mat residu=spec_cube[i]-spec_cube[i-1];
			codeframe(residu,out_name);
		}		
	}
	Mat big_img(512*8,512,CV_32SC1);
	for(int i=0;i<8;++i)
	{
		Mat tmp=spec_cube[i];
		for(int j=0;j<512;++j)
		{
			for(int k=0;k<512;++k)
			{
				*big_img.ptr<int>(j+i*512,k)=*tmp.ptr<int>(j,k);
			}

		}
	}
	string whole_name=out_file+"whole.dat";
	codeframe(big_img,whole_name);

//	vector<int> data{200,-1,-7,-1,-4,-3,1,-5,-6,1,-1,2,2,4,20,1};
//	vector<int> data{0 ,0 ,1 ,0,0 ,0 ,0 ,0,1, 0 ,0, 1,-1, 0, 0, 0};
//	vector<int> data{0,3,0,1,-1,-1,0,1,0,0,0,0,0,0,0,0};
//	vector<int> data{-2,4,3,-3,0,0,-1,0,0,0,0,0,0,0,0,0};
/*
	cout<<"ori data: ";
	for(auto c:data)
	{
		cout<<c<<' ';
	}
	cout<<endl;
*/
// 4*4 test


	return 0;

} 
