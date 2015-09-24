// Copyright (c) 2014 Baidu Corporation
// @file:   test.cpp
// @brief:  tool for testing n-degree fm
// @author: Li Changcheng (lichangcheng@baidu.com)
// @date:   2014-12-23

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fm_n_degree.h"

const int MAX_FILE_NAME_LEN = 1024;

// Function declaration
void print_help();
int parse_command_line(int argc, char** argv, char* testFile, char* modelFile);

int main(int argc, char** argv)
{
	char testFile[MAX_FILE_NAME_LEN];
	char modelFile[MAX_FILE_NAME_LEN];
	fm_n_degree::FM* fm = new fm_n_degree::FM(); 

	if (parse_command_line(argc, argv, testFile, modelFile) != 0) {
		print_help();
		return -1;
	}

	fm->test(testFile, modelFile);
	
	delete fm;

	return 0;
}

// Print help information
void print_help()
{
	printf(
		"Usage: ./test test_file model_file\n"
		"test_file format: label index1:x1 index2:x2 ...\n"
	);
}

// Parse command 
int parse_command_line(int argc, char **argv, char *testFile, char *modelFile)
{
	if (argc != 3) {
		return -1;
	}

	snprintf(testFile, MAX_FILE_NAME_LEN, "%s", argv[1]);
	snprintf(modelFile, MAX_FILE_NAME_LEN, "%s", argv[2]);

	return 0;
}

