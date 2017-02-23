/*
 *  CalcPlan.h
 *  Distribution
 *
 *  Created by Andrey on 10.02.11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef _CALC_PLAN_H_
#define _CALC_PLAN_H_
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

using namespace std;

#define REQUIRED_PARAMETERS_CNT 6
#define _TEST_PLAN_

struct TCalcPlan
{
	int Init(int argc, char ** argv)
	{
		FILE * ifile = NULL;
		int req_params_nums[REQUIRED_PARAMETERS_CNT];
		char usage_string[] = "Usage: %s points_file division_number field_size min_r max_r step [-float] [-GPU=num] [-maps=maps.maps]\n";
		int result = 0;
		// 1. Must be at least 3 parameters
		if (argc < 3) {
			result = 20;
			goto wrong_args;
		}

		is_float = false;
		gpu = 0;
		maps_fn = "";

		int param_idx, req_param_idx;
		for (param_idx = 1, req_param_idx = 0; param_idx < argc; param_idx++ )
		{
			char * pch;
			pch = strtok (argv[param_idx], "-=");
			if (strcmp(pch, "float") == 0)	{
				is_float = true;
			}	else if (strcmp(pch, "GPU") == 0)	{
				pch = strtok (NULL, "-=");
				gpu = atoi(pch);
			}	else if (strcmp(pch, "maps") == 0) {
				pch = strtok (NULL, "-=");
				maps_fn = std::string(pch);
			}	else if (req_param_idx == REQUIRED_PARAMETERS_CNT)	{
				fprintf(stderr, "Too much parameters\n");
				result = 23;
				goto wrong_args;
			}	else	{
				// required parameter
				req_params_nums[req_param_idx] = param_idx;
				req_param_idx++;
			}
		}
		
		// 2. File must exist
		filename = argv[req_params_nums[0]];
		ifile = fopen(filename, "rb");
		if (!ifile) {
			fprintf(stderr, "No such file: %s\n", filename);
			
			result = 21;
			goto wrong_args;
		}
		fclose(ifile);
		
		// 3. Divisions number must be valid
		divisions = atoi(argv[req_params_nums[1]]);
		if (divisions <= 0) {
			result = 22;
			goto wrong_args;
		}

		field_size = atof(argv[req_params_nums[2]]);
		if (field_size <= 0)
		{
			result = 24;
			goto wrong_args;
		}

		min_r = atof(argv[req_params_nums[3]]);
		if (min_r <= 0)
		{
			result = 25;
			goto wrong_args;
		}

		max_r = atof(argv[req_params_nums[4]]);
		if (max_r <= 0)
		{
			result = 26;
			goto wrong_args;
		}

		step = atof(argv[req_params_nums[5]]);
		if (step <= 0)
		{
			result = 27;
			goto wrong_args;
		}

		is_initialized = true;

	#ifdef _TEST_PLAN_
		cout << filename << " " << divisions << " " << field_size << " " 
			<< min_r << " " << max_r << " " << step << " " << is_float << " " << gpu << endl;
	#endif
		return result;
wrong_args:
		fprintf(stderr, usage_string, argv[0]);
		fprintf(stderr, "Error #%d\n", result);
		return result;
	}
	// Required
	char * filename;
	int divisions;
	double field_size;
	double min_r;
	double max_r;
	double step;
	// optional
    bool is_float;
    int gpu;
    std::string maps_fn;
	// internal
	bool is_initialized;
};

#endif 
