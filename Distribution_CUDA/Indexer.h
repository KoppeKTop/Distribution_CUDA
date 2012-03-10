/*
 *  Indexer.h
 *  Distribution
 *
 *  Created by Andrey on 20/12/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */
#ifndef _INDEXER_H_
#define _INDEXER_H_

#include <vector>
using std::vector;

class Indexer {
public:
	Indexer(vector<int>);
	vector<int> next();
	vector<int> curr()	const
	{	return _curr;	}
	void to_begin();
	bool is_last()	const
	{	return _max_idx == _curr_idx;	}
private:
	vector<int> _curr;
    size_t _curr_idx;
    size_t _max_idx;
	vector<int> _size;
	bool _is_last;
};

#endif