#include "func.h"

extern Mat dct_c;
int readtable(ifstream & fin,const string key,Table<string> &tb)
{
	vector<vector<string>> vec_str;
	string line;
	while(getline(fin,line)&&line!=key)
	{
		istringstream sstr(line);
		string word;
		vector<string> temp;
		while(sstr>>word)
		{
			if(word.size()<2){
				cout<<key<<" wrong input"<<endl;
				return -1;
			}
			temp.push_back(word.substr(1,word.size()-2));
		}
		vec_str.push_back(temp);
	}
	tb.modify(vec_str);	
	return 0;
}
/*
 * 实现十进制转换为二进制存储在string中,用指定长度的length表示,如果length表示不下则用实际长度进行表示
 */
string dec2bin(int);
string dec2bin_length(int num,int length)
{
	assert(num>=0&&length>=1);
	string res;
	res=dec2bin(num);
	int  len=res.size();
	if(len>length)
		res=res.substr(len-length,length);
	else 
	{
		string head(length-len,'0');
		res=head+res;
	}
	return res;

}

string dec2bin(int num)
{
	if(num==0)
		return "0";
	if(num==1)
		return "1";
	return dec2bin(num>>1)+to_string(num%2);
}
string enc_cavlc_unit(vector<int> data,unsigned int nL,unsigned int nU)
{

	uint8_t i_total=0;										 //非零系数的个数
	uint8_t i_total_zeros=0;	//最后一个非空系数前0的个数
	uint8_t i_trailing=0;		//拖尾系数的个数,最多为3个
	string sign;  //拖尾符号组合,
	vector<int32_t> run,level;//非零系数前面截止到另一个非零系数前面中间0的个数,非零系数
	string bit; //最终返回值


	int i_last=data.size()-1; //针对4*4的小块进行处理

	while((i_last>=0)&&(data[i_last]==0))
		{--i_last;} //定位到第一个非零系数

	int idx=0;

	/*
	 * 找到level,拖尾系数,run,和total_zeros
	 */

	while(i_last>=0&&abs(data[i_last])==1&&i_trailing<3)
	{
		level.push_back(data[i_last]);
		++i_total;
		++i_trailing;

		if(data[i_last]==-1)
			sign+="1";
		else 
			sign+="0";

		run.push_back(0);
		--i_last;

		
		while(i_last>=0&&data[i_last]==0)
		{
			run[idx]++;
			i_total_zeros++;
			i_last--;
		}
		++idx;

	}

	while(i_last>=0)
	{
		level.push_back(data[i_last]);
		++i_total;

		run.push_back(0);
		--i_last;
		while(i_last>=0&&data[i_last]==0)
		{
			run[idx]=++run[idx];
			++i_total_zeros;
			--i_last;
		}
		++idx;
	}

	/*
	 * encode coeff_token
	 */
	
	int32_t n;
	if(nL>0&&nU>0)
		n=(nL+nU)/2;
	else if(nL>0||nU>0)
		n=nL+nU;
	else 
		n=0;
	Table<string> table_coeff;
	if(n>=0&&n<2)
		table_coeff=table_0;
	else if(n>=2&&n<4)
		table_coeff=table_1;
	else if(n>=4&&n<8)
		table_coeff=table_2;
	else 
		table_coeff=table_3;


	string  coeff_token=table_coeff[i_total][i_trailing];
	bit+=coeff_token;

	if(i_total==0)
		return bit;

	/*
	 * 如果拖尾系数不为0,则需要编码拖尾系数的符号
	 */

	if(i_trailing>0)
		bit+=sign;

	/*
	 * 编码除了拖尾系数的非零系数
	 */

	//initial suffix length 

	uint32_t i_sufx_len =0;

	if(i_total>10&&i_trailing<=1)
		i_sufx_len=1;

	for(int i=i_trailing;i<i_total;++i)
	{
		int i_level_code;
		if(level[i]<0)
			i_level_code=-2*level[i]-1;
		else 
			i_level_code=2*level[i]-2;
			

		int level_prfx,level_sufx;
			//处理前缀
		level_prfx=i_level_code/(1<<i_sufx_len);
		level_sufx=i_level_code%(1<<i_sufx_len);

		while(level_prfx>0)
			{bit+="0";--level_prfx;}
		bit+="1";
			//处理后缀
			
		if(i_sufx_len>0)
		{
			bit+=dec2bin_length(level_sufx,i_sufx_len);
		}
	

	if(i_sufx_len==0)
		++i_sufx_len;
	else if (abs(level[i])>(3<<(i_sufx_len-1))&&i_sufx_len<6)
		++i_sufx_len;
	}

	// encode total_zeros 
	
	if(i_total<data.size())
	{
		// i_total>0, 如果等于0在之前就返回了
		bit+=table_z[i_total-1][i_total_zeros];
	}


	//encode each run of zeros 
	
	int i_zero_left=i_total_zeros;

	if(i_zero_left>=1)
	{
		for(int i=0;i<i_total;++i)
		{
			if(i_zero_left>0&&(i==i_total-1))
				break;
			if(i_zero_left>=1)
			{
				int i_zl=min(i_zero_left-1,6);
				string run_before=table_r[run[i]][i_zl];
				bit+=run_before;
				i_zero_left -= run[i];
			}
		}
	}


	return bit;
}

vector<int>  dec_cavlc_unit(string& bit,uint32_t& idx,unsigned int nL,unsigned int nU)
{

	uint8_t i_total=0;										 //非零系数的个数
	uint8_t i_total_zeros=0;	//最后一个非空系数前0的个数
	uint8_t i_trailing=0;		//拖尾系数的个数,最多为3个
	vector<int32_t> level;//非零系数前面截止到另一个非零系数前面中间0的个数,非零系数

	int run[16]={0};
	vector<int> data(16,0); //最终返回值



	/*
	 * decode coeff_token
	 */
	
	int32_t n;
	if(nL>0&&nU>0)
		n=(nL+nU)/2;
	else if(nL>0||nU>0)
		n=nL+nU;
	else 
		n=0;
	Table<string> table_coeff;
	if(n>=0&&n<2)
		table_coeff=table_0;
	else if(n>=2&&n<4)
		table_coeff=table_1;
	else if(n>=4&&n<8)
		table_coeff=table_2;
	else 
		table_coeff=table_3;



	bool flag;

	for(uint32_t i=1;i<=bit.length()-idx;++i)
	{

		flag=table_coeff.search(bit.substr(idx,i),i_total,i_trailing);
		if(flag)
		{
			idx=i+idx;
			break;
		}
	}

	assert(flag);
	if(i_total==0)
	{
		return vector<int>(16,0);
	}


	//解码非零系数
	
	uint8_t k=0;
	uint8_t m=i_trailing;

	/*
	 * 如果拖尾系数不为0,则需要编码拖尾系数的符号
	 */

	while(m>0)
	{
		if(bit[idx]=='0')
			level.push_back(1);
		else if(bit[idx]=='1')
			level.push_back(-1);
		else 
		{}
		++k;
		--m;
		++idx;
	}

	int i_sufx_len=0;
	if(i_total>10&&(i_trailing<=1))
		i_sufx_len=1;
	
	for(int i=k;i<i_total;++i)
	{

		int level_prfx=0;
		int level_sufx=0;
		while (idx<bit.length())
		{
			if(bit[idx]=='0')
				{++idx;++level_prfx;}
			else 
				{++idx;break;}
		}
		
		for(int j=0;j<i_sufx_len;++j)
		{
			level_sufx=level_sufx*2+(bit[idx]-'0');
			++idx;
		}
		int i_level_code=(1<<i_sufx_len)*level_prfx+level_sufx;

		if(i_level_code%2==0)
			level.push_back((i_level_code+2)/2);
		else 
			level.push_back((i_level_code+1)/-2);
		if(i_sufx_len==0)
			++i_sufx_len;
		else if (abs(level[i])>(3<<(i_sufx_len-1))&&i_sufx_len<6)
			++i_sufx_len;
	}

	
	// decode total_zeros
		
	
	if(i_total<data.size())
	{
		// i_total>0, 如果等于0在之前就返回了
		string tmp;
		pair<int32_t,int32_t> pos;
		while(idx<bit.length())
		{
			tmp=tmp+bit[idx];

			pair<uint8_t,uint8_t> pos;
			auto flag=table_z.search_v2(tmp,pos,i_total-1,table_row);
			++idx;
			if(flag)
				{i_total_zeros=pos.second;break;}
		}
	
	}

	


	//decode each run of zeros 
	
	auto i_zero_left=i_total_zeros;

	string ss;
	int run_i=0;
	if(i_zero_left>0)
	{
		while(run_i<i_total-1&&i_zero_left>0)
		{
			ss+=bit[idx];
			int i_zl=min(i_zero_left-1,6);
			pair<uint8_t,uint8_t> pos;
			auto flag=table_r.search_v2(ss,pos,i_zl,table_col);
			++idx;
			if(flag)
			{

				run[run_i]=pos.first;
				i_zero_left -= run[run_i];
				++run_i;
				ss="";
			}

		}
		if(i_zero_left>0)
		{
			run[run_i]=i_zero_left;
			i_zero_left=0;
		}
	}

	for(int i=i_total+i_total_zeros-1;i>=0;)
	{
		for(uint8_t j=0;j<level.size();++j)
		{
			data[i]=level[j];
			i--;
			i -= run[j];

		}
	}
	return data;
}


void mat2d_show(Mat &m);

vector<int>  dct_zigzag(Mat &mb)
{
	assert(mb.depth()==CV_32S);
	
	/*
	 * dct transform 
	 */
	/* 
	Mat dct_mb(mb.size(),CV_64FC1);
	dct(Mat_<double>(mb),dct_mb);
	mb=Mat_<int>(dct_mb);	
	*/
	/**/
//	mb=Mat_<int>(dct_c*Mat_<float>(mb)*dct_c.t());
	vector<int> res;
	for(int i=0;i<16;++i)
	{
		int num=*(mb.ptr<int>(zg_table[i][0],zg_table[i][1]));
		if(num!=-1&&num!=1&&num!=0)
		{
		//	res.push_back(0);
			res.push_back(num);
		}
		else 
			res.push_back(0);
	//	res.push_back(num);

	}
	
	return res;
}
Mat  rev_dct_zigzag(vector<int>& data)
{
	Mat res(4,4,CV_32SC1);
	for(int i=0;i<16;++i)
	{
		*(res.ptr<int>(zg_table[i][0],zg_table[i][1]))=data[i];
	}

	/*
	 * dct 逆变换处理
	 */
	/*
	Mat idct_mb(res.size(),CV_64FC1);
	idct(Mat_<double>(res),idct_mb);
	res=Mat_<int>(idct_mb);
    */
	/**/
//	res=Mat_<int>(dct_c.inv()*Mat_<float>(res)*dct_c.t().inv());	
	return res;
}

void mat2d_show(Mat &m)
{

	assert(m.dims==2);
	int row=m.rows;
	int col=m.cols;

	for(int i=0;i<row;++i)
	{
		for(int j=0;j<col;++j)
		{
			cout<<*m.ptr<int>(i,j)<<" ";
		}
		cout<<endl;
	}


}

//输入格式是Mat类型，I1，I2代表是输入的两幅图像
double getPSNR(const Mat& I1, const Mat& I2)
{
	assert(I1.channels()==I2.channels());
    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|AbsDiff函数是 OpenCV 中计算两个数组差的绝对值的函数
    s1.convertTo(s1, CV_32F);  // 这里我们使用的CV_32F来计算，因为8位无符号char是不能进行平方计算
    s1 = s1.mul(s1);           // |I1 - I2|^2

    Scalar s = sum(s1);         //对每一个通道进行加和

	double sse = 0; // sum channels
	for(int i=0;i<I1.channels();++i)
		sse += s.val[i]; // sum channels

    if( sse <= 1e-10) // 对于非常小的值我们将约等于0
        return 0;
    else
    {
        double  mse =sse /(double)(I1.channels() * I1.total());//计算MSE
		double max_value;
		if(I1.depth()==CV_8U)
			max_value=255;
		else if(I1.depth()==CV_16U)
			max_value=65535;
		else if(I1.depth()==CV_16U)
			max_value=65535;
		else 
		{
			cerr<<"no permit value type"<<endl;
			return -1;
		}
		
        double psnr = 10.0*log10((max_value*max_value)/mse);
        return psnr;//返回PSNR
    }
}
