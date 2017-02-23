/*
 *  Coord.h
 *  Distribution
 *
 *  Created by Andrey on 13/12/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef COORD_H_
#define COORD_H_

//#include "CoordinateVector.h"

#include <vector>
#include <exception>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

using namespace std;

//typedef int CoordinateType;
#define MAX_DIMS 4

class OutOfBoundError: public exception
{
public:
	OutOfBoundError(const char * filename, const int line)
	{
		char msg_str[] = "Coordinate index out of bounds on: %s:%d";
		m_what_str = (char *)malloc(strlen(msg_str) + strlen(filename) + 15);
		sprintf(m_what_str, "Coordinate index out of bounds on: %s:%d\n", filename, line);
	}
	virtual const char* what() const throw()
	{
		return m_what_str;
	}
	
private:
	char * m_what_str;
};

class SizeError: public exception
{
	virtual const char* what() const throw()
	{
		return "Vector size is not equal to Dims";
	}
};

template < class CoordinateType >
class Coord {
public:
	Coord(CoordinateType *);
	Coord(const Coord &);
	Coord(CoordinateType=0, CoordinateType=0, CoordinateType=0, CoordinateType=0);
	Coord(const vector<CoordinateType> v)	{
		if (v.size() != Coord::GetDefDims()) {
			throw OutOfBoundError(__FILE__, __LINE__);
		}
		for (int dim = 0; dim < Coord::GetDefDims(); ++dim) {
			mCV[dim] = v[dim];
		}
	}
	
	virtual ~Coord();
	
	CoordinateType GetCoord(int) const
#ifdef BOUNDS_CHECK
	throw(OutOfBoundError)
#endif
    ;
	void SetCoord(int, CoordinateType)
#ifdef BOUNDS_CHECK
	throw(OutOfBoundError)
#endif
    ;
	void SetPosition(CoordinateType, CoordinateType, CoordinateType=0, CoordinateType=0);
	
	Coord operator+ (const Coord &) const;
	Coord operator% (const Coord &) const;
	Coord operator- (const Coord &) const;
	Coord operator* (const Coord &) const;
	
	Coord operator+ (const CoordinateType) const;
	Coord operator% (const CoordinateType) const;
	Coord operator- (const CoordinateType) const;
	Coord operator/ (const CoordinateType) const;
	Coord operator* (const CoordinateType) const;
	
	bool operator== (const Coord &) const;
	bool operator< (const Coord &) const;
	bool operator<= (const Coord &) const;
	bool operator> (const Coord &) const;
	bool operator>= (const Coord &) const;
	bool operator!= (const Coord &) const;
	
	Coord & operator= (const Coord & rhs);
	
	static int GetDefDims()	{	return	mDefDims;	}
	static void SetDefDims(int dims)
	{
		if (2 <= dims || dims <= MAX_DIMS)
			mDefDims = dims;
	}
	
	CoordinateType& operator[](const int ind)
#ifdef BOUNDS_CHECK
	throw(OutOfBoundError)
#endif
    ;
    
	CoordinateType operator[](const int ind) const
#ifdef BOUNDS_CHECK
	throw(OutOfBoundError)	
#endif
    {return GetCoord(ind); }
	vector<CoordinateType> * to_vec() const;
private:
	CoordinateType mCV[MAX_DIMS];
	int mDims;
	
	bool CheckBounds(int) const;
	
	static int mDefDims;
	static int instances;
};

typedef Coord<int> iCoord;
typedef Coord<double> dCoord;
typedef vector<iCoord> CoordVec;

template < class CoordinateType >
Coord<CoordinateType>::Coord(CoordinateType X, CoordinateType Y, CoordinateType Z, CoordinateType V)
{
	Coord<CoordinateType>::instances += 1;
	
	mDims = Coord<CoordinateType>::mDefDims;
	
	SetPosition(X,Y,Z,V);
}

template < class CoordinateType >
Coord<CoordinateType>::Coord(CoordinateType * cv)
{
	Coord<CoordinateType>::instances += 1;
	
	mDims = Coord<CoordinateType>::mDefDims;
	
	for (int i=0; i < Coord<CoordinateType>::mDefDims; i++)
	{
		mCV[i] = cv[i];
	}
}

template < class CoordinateType >
Coord<CoordinateType>::Coord(const Coord & c)
{
	
	Coord<CoordinateType>::instances += 1;
	
	mDims = Coord<CoordinateType>::mDefDims;
	
	//mDims = c.mDims;
	for (int i=0; i < Coord<CoordinateType>::mDefDims; i++)
	{
		mCV[i] = c.GetCoord(i);
	}
}

template < class CoordinateType >
Coord<CoordinateType>::~Coord()
{
	Coord<CoordinateType>::instances -= 1;
};

template < class CoordinateType >
inline bool Coord<CoordinateType>::CheckBounds(int index) const
{
	return (0 <= index && index < Coord<CoordinateType>::mDefDims);
}

template < class CoordinateType >
inline CoordinateType Coord<CoordinateType>::GetCoord(int num) const
#ifdef BOUNDS_CHECK
throw(OutOfBoundError)
#endif
{
#ifdef BOUNDS_CHECK
	if (this->CheckBounds(num))
#endif
		return mCV[num];
#ifdef BOUNDS_CHECK
	throw OutOfBoundError(__FILE__, __LINE__);
#endif
}

template < class CoordinateType >
inline void Coord<CoordinateType>::SetCoord(int num, CoordinateType val) 
#ifdef BOUNDS_CHECK
throw(OutOfBoundError)
#endif
{
#ifdef BOUNDS_CHECK
	if (this->CheckBounds(num)) {
#endif
		this->mCV[num] = val;
#ifdef BOUNDS_CHECK
		return;
	}
	throw OutOfBoundError(__FILE__, __LINE__);
#endif
}

template < class CoordinateType >
inline void Coord<CoordinateType>::SetPosition(CoordinateType X, CoordinateType Y, CoordinateType Z, CoordinateType V)
{
	mCV[0] = X;
	mCV[1] = Y;
	mCV[2] = Z;
	mCV[3] = V;
}

template < class CoordinateType >
Coord<CoordinateType> Coord<CoordinateType>::operator+ (const Coord<CoordinateType> &rhs) const
{
//	Coord res(mCV[0]+rhs.mCV[0], mCV[1]+rhs.mCV[1], mCV[2]+rhs.mCV[2], mCV[3]+rhs.mCV[3]);
//	for (int i = 0; i < Coord<CoordinateType>::mDefDims; ++i)
//		res.SetCoord(i, this->GetCoord(i) + rhs.GetCoord(i));
	return Coord(mCV[0]+rhs.mCV[0], mCV[1]+rhs.mCV[1], mCV[2]+rhs.mCV[2], mCV[3]+rhs.mCV[3]);
}

template < class CoordinateType >
Coord<CoordinateType> Coord<CoordinateType>::operator% (const Coord<CoordinateType> & rhs) const
{
	Coord res;
	for (int i = 0; i < Coord<CoordinateType>::mDefDims; ++i)
		res.SetCoord(i, this->GetCoord(i) % ((rhs.GetCoord(i) != 0)?rhs.GetCoord(i):1));
	return res;
}

template < class CoordinateType >
Coord<CoordinateType> Coord<CoordinateType>::operator- (const Coord<CoordinateType> & rhs) const
{
	Coord res;
	for (int i = 0; i < Coord<CoordinateType>::mDefDims; ++i)
		res.SetCoord(i, this->GetCoord(i) - rhs.GetCoord(i));
	return res;
}

template < class CoordinateType >
Coord<CoordinateType> Coord<CoordinateType>::operator* (const Coord<CoordinateType> & rhs) const
{
	Coord res;
	for (int i = 0; i < Coord<CoordinateType>::mDefDims; ++i)
		res.SetCoord(i, this->GetCoord(i) * rhs.GetCoord(i));
	return res;
}

template < class CoordinateType >
Coord<CoordinateType> Coord<CoordinateType>::operator/ (const CoordinateType divide) const
{
	Coord<CoordinateType> res(*this);
	
	for (int i = 0; i < Coord<CoordinateType>::mDefDims; ++i)
		res.SetCoord(i, this->GetCoord(i) / divide);
	
	return res;
}

template < class CoordinateType >
Coord<CoordinateType> Coord<CoordinateType>::operator+ (const CoordinateType rhs) const
{
	Coord<CoordinateType> tmp(rhs,rhs,rhs, rhs);
	Coord<CoordinateType> res = *this + tmp;
	return res;
}

template < class CoordinateType >
Coord<CoordinateType> Coord<CoordinateType>::operator% (const CoordinateType rhs) const
{
	Coord<CoordinateType> tmp(rhs,rhs,rhs, rhs);
	Coord<CoordinateType> res = *this % tmp;
	return res;
}

template < class CoordinateType >
Coord<CoordinateType> Coord<CoordinateType>::operator- (const CoordinateType rhs) const
{
	Coord<CoordinateType> tmp(rhs,rhs,rhs, rhs);
	Coord<CoordinateType> res = *this - tmp;
	return res;
}

template < class CoordinateType >
Coord<CoordinateType> Coord<CoordinateType>::operator* (const CoordinateType rhs) const
{
	Coord<CoordinateType> tmp(rhs,rhs,rhs, rhs);
	Coord<CoordinateType> res = *this * tmp;
	return res;
}

template < class CoordinateType >
bool Coord<CoordinateType>::operator== (const Coord<CoordinateType> & rhs) const
{
	if (mDims != rhs.mDims)	return false;
	bool res = true;
	for (int i = 0; (i < Coord<CoordinateType>::mDefDims) && res; ++i)
		res = res && (rhs.GetCoord(i) == this->GetCoord(i));
	return res;
}

template < class CoordinateType >
bool Coord<CoordinateType>::operator< (const Coord<CoordinateType> & rhs) const
{
	bool res = false;
	for (int i = 0; i < Coord<CoordinateType>::mDefDims; ++i)
	{
		if (this->GetCoord(i) < rhs.GetCoord(i)) {
			// less case - return true
			res = true;
			break;
		}
		if (this->GetCoord(i) == rhs.GetCoord(i))
			// equal case - continue check
			continue;
		// greater case - return false
		break;
	}
	return res;
}

template < class CoordinateType >
bool Coord<CoordinateType>::operator<= (const Coord<CoordinateType> & rhs) const
{
	bool res = (*this < rhs || *this == rhs);
	return res;
}

template < class CoordinateType >
bool Coord<CoordinateType>::operator> (const Coord<CoordinateType> & rhs) const
{
	bool res = !(*this <= rhs);
	return res;
}

template < class CoordinateType >
bool Coord<CoordinateType>::operator>= (const Coord<CoordinateType> & rhs) const
{
	bool res = !(*this < rhs);
	return res;
}

template < class CoordinateType >
bool Coord<CoordinateType>::operator!= (const Coord & rhs) const
{
	bool res = !(*this == rhs);
	return res;
}

template < class CoordinateType >
Coord<CoordinateType> & Coord<CoordinateType>::operator= (const Coord<CoordinateType> & rhs)
{
	if (this != &rhs)
	{
		for (int i = 0; i<Coord<CoordinateType>::mDefDims; i++)
		{
			this->mCV[i] = rhs.mCV[i];
		}
	}
	return *this;
}

template < class CoordinateType >
vector<CoordinateType> * Coord<CoordinateType>::to_vec() const
{
	vector<CoordinateType> * res = new vector<CoordinateType>(Coord<CoordinateType>::mDefDims);
	for (int i = 0; i < Coord<CoordinateType>::mDefDims; i++)
	{
		res[0][i] = this->mCV[i];
	}
	return res;
}

//template < class CoordinateType >
//Coord<CoordinateType> operator= (const vector<CoordinateType> & rhs)
//{
//	if (rhs.size() == Coord<CoordinateType>::mDefDims)
//	{
//		for (int i = 0; i<Coord<CoordinateType>::mDefDims; i++)
//		{
//			this->mCV[i] = rhs[i];
//		}
//	}	else {
//		throw SizeError();
//	}
//	
//	return *this;
//}

template <class CoordinateType>
int Coord<CoordinateType>::mDefDims = 2;

template <class CoordinateType>
int Coord<CoordinateType>::instances = 0;

template < class CoordinateType >
ostream& operator <<(ostream& stream, const Coord<CoordinateType> & c)
{
	//stream << "(";
	stream << c.GetCoord(0);
	for (int d = 1; d < Coord<CoordinateType>::GetDefDims(); ++d)
	{
		stream << "\t" << c.GetCoord(d);
	}
	//stream << ")";
	return stream;
}


template < class CoordinateType >
CoordinateType& Coord<CoordinateType>::operator[](const int index)
#ifdef BOUNDS_CHECK
throw(OutOfBoundError)
#endif
{
	if (this->CheckBounds(index)) {
		return mCV[index];
	}
	throw OutOfBoundError(__FILE__, __LINE__);
}


#endif