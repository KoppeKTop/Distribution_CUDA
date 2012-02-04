/*
 *  Indexer.cpp
 *  Distribution
 *
 *  Created by Andrey on 20/12/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#include "Indexer.h"

#include <vector>
#include <algorithm>

using std::vector;

Indexer::Indexer(vector<int> size)
{
	_size = size;
	_curr = vector<int>(size.size(), 0);
	_is_last = false;
    _curr_idx = 0;
    
    _max_idx = 1;
    for (int idx = 0; idx < size.size(); ++idx) {
        _max_idx *= size[idx];
    }
}

vector<int> Indexer::next()
{
    _curr_idx ++;
	for (int d = _size.size() - 1; d >= 0; --d) {
		if (++_curr[d] == _size[d]) {
			_curr[d] = 0;
		}	else {
			break;
		}
	}
	return _curr;
}

void Indexer::to_begin()
{
	_curr = vector<int>(_size.size(), 0);
    _curr_idx = 0;
}

