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
using namespace std;

#define REQUIRED_PARAMETERS_CNT 3
#define _TEST_PLAN_

struct TCalcPlan
{
	int Init(int argc, char ** argv)
	{
		// TODO: parse YAML config instead argv
		FILE * ifile = NULL;
		int req_params_nums[REQUIRED_PARAMETERS_CNT];
		char usage_string[] = "Usage: %s points_file division_number field_size [-float] [-GPU=num]\n";
		int result = 0;
		// 1. Must be at least 3 parameters
		if (argc < 3) {
			result = 20;
			goto wrong_args;
		}

		is_float = false;
		gpu = 0;

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
		is_initialized = true;

	#ifdef _TEST_PLAN_
		cout << filename << " " << divisions << " " << is_float << " " << gpu << endl;
	#endif
wrong_args:
		fprintf(stderr, usage_string, argv[0]);
		fprintf(stderr, "Error #%d\n", result);
		return result;
	}
	// Required
	char * filename;
	int divisions;
	double field_size;
	// optional
    bool is_float;
    int gpu;
	// internal
	bool is_initialized;
};

#endif 
