/*
 *  Rect.h
 *  Distribution
 *
 *  Created by Andrey on 15/12/10.
 *  Copyright 2010 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef RECT_H_
#define RECT_H_

#include "Coord.h"

struct Rect {
	double minCoord[3], maxCoord[3];
};

#endif