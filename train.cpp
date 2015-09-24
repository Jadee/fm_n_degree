// Copyright (c) 2014 Baidu Corporation
// @file:   train.cpp
// @brief:  tool for training n-degree FM
// @author: Li Changcheng (lichangcheng@baidu.com)
// @date:   2014-12-23

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fm_n_degree.h"

const int MAX_FILE_NAME_LEN = 1024;

// Function declaration
void print_help();
int parse_command_line(fm_n_degree::FM* fm, int argc, char** argv, char* trainFile, char* modelFile);

int main(int argc, char** argv)
{
	char trainFile[MAX_FILE_NAME_LEN];
	char modelFile[MAX_FILE_NAME_LEN];
	fm_n_degree::FM* fm = new fm_n_degree::FM();

	if (parse_command_line(fm, argc, argv, trainFile, modelFile) != 0) {
		print_help();
		return -1;
	}

//	printf("%d\t%d\t%f\t%f\t%s\t%s\n", fm->m_degree, fm->m_factSize, fm->m_regFactor, fm->m_learnRate, trainFile, modelFile);

	fm->read_data(trainFile);
	
/*	for (int i = 0; i < fm->m_dataNum; ++i) {
		printf("%d", fm->m_data[i].y);
		for (int j = 0; j < fm->m_featNum; ++j) {
			printf("\t%f", fm->m_data[i].x[j]);
		}
		printf("\n");
	}
	getchar(); */
	
	fm->train();
	fm->save_model(modelFile);

//	printf("%f\t%f\t%f\n", fm->predict(fm->m_data), fm->predict(fm->m_data + 1), fm->predict(fm->m_data + 2));

	delete fm;

	return 0;
}

// Print help information
void print_help()
{
    printf(
            "Usage: ./train [options] training_file [model_file]\n"
            "options:\n"
            "   -d fm degree (default 2)\n"
            "   -k factor size (default 3)\n"
            "   -c regularization coefficient (default 0)\n"
            "   -l learning rate (default 0.01)\n"
            "   -p partial FM flag (0 or 1, default 0)\n"
            "   -v initialization standard deviation (default 0.1)\n"
            "   -b mini_batch (default 200)\n"
            "   -i iterations num (default 200) \n"
            "   -n regularization term (1 - L1, 2 - L2, default 2)\n\n"
            "training_file format: \n"
            "   label index1:x1 index2:x2 ...\n"
    );
}

// Parse command 
int parse_command_line(fm_n_degree::FM* fm, int argc, char** argv, char* trainFile, char* modelFile)
{
	// Set default parameters
	fm->set_fm_degree(2);
	fm->set_factor_size(3);
	fm->set_regular_factor(0.0f);
	fm->set_learn_rate(0.01f);
	fm->set_partial_fm_flag(0);
	fm->set_init_std_dev(0.1f);
	fm->set_regular_term(2);
    fm->set_mini_batch(200);
    fm->set_iterations_num(200);
	
	// parse options
	int i = 0;
	for (i = 1; i < argc; ++i) {
		if (argv[i][0] != '-') {
			break;
		}

		if (++i >= argc) {
			return -1;
		}

		switch (argv[i-1][1]) {
			case 'd':
				int degree = atoi(argv[i]);
				if (degree < 1 || degree > 10) {
					printf("[ERROR] Invalid -d value, should be in [2, 10]!\n");
					return -1;
				}
				fm->set_fm_degree(degree);
				break;

			case 'k':
				int factorSize = atoi(argv[i]);
				if (factorSize <= 0) {
					printf("[ERROR] Invalid -k value (should be > 0)!\n");
					return -1;
				}
				fm->set_factor_size(factorSize);
				break;

			case 'c':
				float regFactor = atof(argv[i]);
				if (regFactor < 0) {
					printf("[ERROR] Invalid -c value (should be > 0)!\n");
					return -1;
				}
				fm->set_regular_factor(regFactor);
				break;

			case 'l':
				float learnRate = atof(argv[i]);
				if (learnRate < 0) {
					printf("[ERROR] Invalid -l value (should be > 0)\n");
					return -1;
				}				
				fm->set_learn_rate(learnRate);
				break;
			
			case 'p':
				int partialFmFlag = atoi(argv[i]);
				if (partialFmFlag != 0 && partialFmFlag != 1) {
					printf("[ERROR] Invalid -p value (should be 0 or 1)\n");
					return -1;
				}				
				fm->set_partial_fm_flag(partialFmFlag);
				break;
				
			case 'v':
				float initStdDev = atof(argv[i]);
				if (initStdDev < 0) {
					printf("[ERROR] Invalid -v value (should be > 0)\n");
					return -1;
				}				
				fm->set_init_std_dev(initStdDev);
				break;

			case 'n':
				int regularTerm = atoi(argv[i]);
				if (regularTerm != 1 && regularTerm != 2) {
					printf("[ERROR] Invalid -n value (should be 1 or 2)\n");
					return -1;					
				}
				fm->set_regular_term(regularTerm);
				break;

            case 'b':
                int mini_batch = atoi(argv[i]);
                if (mini_batch < 0) {
                    printf("[ERROR] Invalid -b value (should be > 0)\n");
                    return -1;
                }
                fm->set_mini_batch(mini_batch);
                break;

            case 'i':
                int iter_num = atoi(argv[i]);
                if (iter_num < 0) {
                    printf("[ERROR] Invalid -b value (should be > 0)\n");
                    return -1;
                }
                fm->set_iterations_num(iter_num);
                break;
				
			default:
				printf("[ERROR] Unknown option: -%c\n", argv[i-1][1]);
				return -1;
		}
	}

	if (i >= argc) {
		return -1;
	}

	// Parse input file name
	snprintf(trainFile, MAX_FILE_NAME_LEN, "%s", argv[i]);

	// Parse model file
	if (i < argc - 1) {
		snprintf(modelFile, MAX_FILE_NAME_LEN, "%s", argv[i+1]);
	}
	else {						// Default model file name
		char* ptr = strrchr(argv[i], '/');
		if (ptr == NULL) {
			ptr = argv[i];		// argv[i] is the input file name
		}
		else {
			++ptr;				// Ignor path
		}
		sprintf(modelFile, "%s.model", ptr);
	}

	return 0;
}

