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
	TCalcPlan(int argc, char ** argv)
	{
		// TODO: parse YAML config instead argv
		FILE * ifile;
		int req_params_nums[REQUIRED_PARAMETERS_CNT];
		char usage_string[] = "Usage: %s points_file division_number field_size [-float] [-GPU=num]\n";
		
		// 1. Must be at least 3 parameters
		if (argc < 3) {
			fprintf(stderr, usage_string, argv[0]);
			exit(20);
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
				fprintf(stderr, usage_string, argv[0]);
				exit(23);
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
			fprintf(stderr, usage_string, argv[0]);
			exit(21);
		}
		fclose(ifile);
		
		// 3. Divisions number must be valid
		divisions = atoi(argv[req_params_nums[1]]);
		if (divisions <= 0) {
			fprintf(stderr, usage_string, argv[0]);
			exit(22);
		}

		field_size = atof(argv[req_params_nums[2]]);
		if (field_size <= 0)
		{
			fprintf(stderr, usage_string, argv[0]);
			exit(24);
		}

	#ifdef _TEST_PLAN_
		cout << filename << " " << divisions << " " << is_float << " " << gpu << endl;
	#endif
	}
	// Required
	char * filename;
	int divisions;
	double field_size;
	// optional
    bool is_float;
    int gpu;
};

#endif 
