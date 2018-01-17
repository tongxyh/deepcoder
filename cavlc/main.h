#ifndef MAIN_H
#define MAIN_H
#include <vector>
#include <string>
#include <iostream>
#include <assert.h>
#include <utility>
using namespace std;


enum Row_Or_Col{table_blank,table_row,table_col};
template<typename T>
class Table{
public:
	Table()
	{
		content=new vector<vector<T>>();
	}
	Table(vector<vector<T>> tb)
	{
		content=new vector<vector<T>>(tb);
	}
	Table(typename vector<T>::size_type m,typename vector<T>::size_type n)
	{
		content=new vector<vector<T>>(m,vector<T>(n));
	}
	Table(const Table& tb)
	{
		if(&tb==this)
			return;
		else 
		{
			if(content)
				delete content; 
			content=new vector<vector<T>>(*(tb.content));
		}
	}
	void modify(vector<vector<T>> &vec)
	{
		if(content)
			delete content;
		content=new vector<vector<T>> (vec);
	}
	Table & operator = (const Table & rtb)
	{
		if(&rtb==this)
			return *this;
		else
		{
			if(content)
				delete content;
			content=new vector<vector<T>>(*(rtb.content));
		}
		return *this;
	}

	vector<T> &  operator [](uint8_t k)
	{
		if(k>=this->getrow())
		{
			cout<<"col overflow error"<<endl;
		}
		return (*content)[k];
	}
	typename vector<T>::size_type getrow()
	{
		if(content)
			return content->size();
		return -1;
	}
	typename vector<T>::size_type getcol()
	{
		if(content)
			return content->begin()->size();
		return -1;
	}
	void display()
	{
		for(auto beg=content->begin();beg!=content->end();++beg)
		{
			for(auto jbeg=beg->begin();jbeg!=beg->end();++jbeg)
				cout<<*jbeg<<" ";
			cout<<endl;
		}
	}

	bool search(T,uint8_t &row,uint8_t &col);
	~Table()
	{
		if(content)
			delete content;
	}

	bool search_v2(T &,pair<uint8_t,uint8_t> &, uint8_t pos=0,Row_Or_Col type=table_blank);


private:
	vector<vector<T>> *content;
};

template <typename T>

bool Table<T>::search(T obj,uint8_t &row,uint8_t &col)
{

	auto selfrow=this->getrow();
	auto selfcol=this->getcol();

	for(uint32_t i=0;i<selfrow;++i)
	{
		for(uint32_t j=0;j<selfcol;++j)
		{
			if((*content)[i][j]==obj)
			{
				row=i;
				col=j;
				return true;
			}
		}
	}


	return false;

}


template <typename T>
bool Table<T>::search_v2(T& obj,pair<uint8_t,uint8_t> &res, uint8_t pos,Row_Or_Col type)
{
	auto selfrow=this->getrow();
	auto selfcol=this->getcol();
	if(type==table_row)
	{
		for(uint32_t j=0;j<selfcol;++j)
		{
			if((*content)[pos][j]==obj)
			{
				res.first=pos;
				res.second=j;
				return true;
			}
		}
		
	}
	else if(type==table_col)
	{
		for(uint32_t i=0;i<selfrow;++i)
		{
			if((*content)[i][pos]==obj)
			{
				res.first=i;
				res.second=pos;
				return true;
			}
		}
	}
	else 
	{
		if(search(obj,res.first,res.second))
		{
			return true;
		}
	}
	return false;
}


#endif
