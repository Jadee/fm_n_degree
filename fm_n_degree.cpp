// Copyright (c) 2014 Baidu Corporation
// @file:   fm_n_degree.cpp
// @brief:  Source file, n-degree factorization machine
// @author: Li Changcheng (lichangcheng@baidu.com)
// @date:   2014-12-21

#include "fm_n_degree.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef MAX
#define MAX(a,b) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef MIN
#define MIN(a,b) ( ((a) < (b)) ? (a) : (b) )
#endif

namespace fm_n_degree {

const int FM::S_MAX_STOP_ITER_NUM = 200;
const int FM::S_MINI_BATCH_SIZE = 800;

FM::FM() : m_featNum(0), m_dataNum(0), m_data(NULL), m_degree(0), m_factSize(0), m_w0(0.0f), m_w(NULL), 
		   m_v(NULL), m_regFactor(0.0f), m_learnRate(0.0f), m_gradW0(0.0f), m_gradW(NULL), m_gradV(NULL), 
		   m_sumGrad2(0.0f), m_momentumW0(0.0f), m_momentumW(NULL), m_momentumV(NULL), m_partialFmFlag(0), 
		   m_fmFeatFlag(NULL), m_maxLabel(0), m_minLabel(0), m_initStdDev(0.0f), m_norm(2), m_sumW0(0.0f), 
		   m_sumW(NULL), m_sumV(NULL)
{
}

FM::~FM()
{
	// Free data
	if (m_data != NULL) {
		for (int i = 0; i < m_dataNum; ++i) {
			if (m_data[i].x != NULL) {
				delete m_data[i].x;
				m_data[i].x = NULL;
			}
		}
		delete m_data;
		m_data = NULL;
	}

	// Free model
	if (m_w != NULL) {
		delete m_w;
		m_w = NULL;
	}

	if (m_v != NULL) {
		for (int i = 0; i < m_degree; ++i) {
			if (m_v[i] != NULL) {
				delete m_v[i];
				m_v[i] = NULL;
			}
		}
		delete m_v;
		m_v = NULL;
	}
	
	if (m_sumW != NULL) {
		delete m_sumW;
		m_sumW = NULL;
	}

	if (m_sumV != NULL) {
		for (int i = 0; i < m_degree; ++i) {
			if (m_sumV[i] != NULL) {
				delete m_sumV[i];
				m_sumV[i] = NULL;
			}
		}
		delete m_sumV;
		m_sumV = NULL;
	}

	// Free gradients
	if (m_gradW != NULL) {
		delete m_gradW;
		m_gradW = NULL;
	}

	if (m_gradV != NULL) {
		for (int i = 0; i < m_degree; ++i) {
			if (m_gradV[i] != NULL) {
				delete m_gradV[i];
				m_gradV[i] = NULL;
			}
		}
		delete m_gradV;
		m_gradV = NULL;
	}

	// Free momentum
	if (m_momentumW != NULL) {
		delete m_momentumW;
		m_momentumW= NULL;
	}

	if (m_momentumV != NULL) {
		for (int i = 0; i < m_degree; ++i) {
			if (m_momentumV[i] != NULL) {
				delete m_momentumV[i];
				m_momentumV[i] = NULL;
			}
		}
		delete m_momentumV;
		m_momentumV = NULL;
	}

	// Free sparseFlag
	if (m_fmFeatFlag != NULL) {
		delete m_fmFeatFlag;
		m_fmFeatFlag = NULL;
	}		
}

void FM::set_regular_factor(float regFactor)
{
	m_regFactor = regFactor;
}

void FM::set_learn_rate(float learnRate)
{
	m_learnRate = learnRate;
}

void FM::set_fm_degree(int degree)
{
	m_degree = degree;
}

void FM::set_mini_batch(int mini_batch)
{
    m_mini_batch = mini_batch;
}

void FM::set_iterations_num(int iter_num)
{
    m_iter_num = iter_num;
}

void FM::set_factor_size(int factSize)
{
	m_factSize = factSize;
}

void FM::set_partial_fm_flag(int flag)
{
	m_partialFmFlag = flag;
}

void FM::set_init_std_dev(float stdDev)
{
	m_initStdDev = stdDev;
}

void FM::set_regular_term(int regularTerm)
{
	m_norm = regularTerm;
}

int FM::read_data(const char* fileName)
{
	// Format: y(-1/0, 1) \t x1 \t x2 \t, ...
	FILE* fp = fopen(fileName, "r");
	if (fp == NULL) {
		printf("[ERROR] Cannot open %s! Reading data failed!\n", fileName);
		return -1;
	}

	const int MAX_LINE_DATA_LEN = 100000;
	char buf[MAX_LINE_DATA_LEN];

	// Count data number, and allocate data memory
	m_dataNum = 0;
	m_featNum = 0;
	
	while (fgets(buf, MAX_LINE_DATA_LEN, fp) != NULL) {
		++m_dataNum;

		// Count feature number
		char* ptr = buf + strlen(buf);
		while (*ptr != ' ' && *ptr != '\t' && ptr != buf) {
			if (*ptr == ':') {
				*ptr = '\0';
			}
			--ptr;
		}

		int maxFeatNum = static_cast<int>(strtol(ptr + 1, NULL, 10));
		if (maxFeatNum < 1) {
			printf("[ERROR] Invalid data format, wrong feature index!\n");
			fclose(fp);
			return -1;
		}
		
		m_featNum = MAX(m_featNum, maxFeatNum);
	}

	if (m_dataNum < 1) {
		printf("[ERROR] No data in the file!\n");
		fclose(fp);
		return -1;
	}

	// Allocate memory and initialize data
	m_data = new Data[m_dataNum];
	for (int i = 0; i < m_dataNum; ++i) {
		m_data[i].y = 0;
		m_data[i].x = NULL;
		m_data[i].score = 0.0f;
		m_data[i].sumVX = 0.0f;
	}
	
	rewind(fp);				// Back to the head of the file
	
	// Parse data
	int lineNum = 0;
	m_dataNum = 0;			// Reset data number, drop invalid data
	while (fgets(buf, MAX_LINE_DATA_LEN, fp) != NULL) {
		++lineNum;
		if (parse_line(buf, m_data + m_dataNum) != 0) {
			printf("[WARNING] Parsing line %d failed!\n", lineNum);
			continue;
		}

		++m_dataNum;
	}
	
	fclose(fp);
	return 0;
}

int FM::parse_line(char* buf, Data* ptrData)
{
	// Parse label
	char* ptr = strtok(buf, "\t ");
	if(ptr == NULL) {
		printf("[WARNING] Parsing label failed!\n");
		return -1;
	}
	
	ptrData->y = static_cast<int>(strtol(ptr, NULL, 10));

	// Get maxLabel and minLabel
	m_maxLabel = MAX(m_maxLabel, ptrData->y);
	m_minLabel = MIN(m_minLabel, ptrData->y);

	// Allocate data memory
	ptrData->x = new float[m_featNum];
	for (int i = 0; i < m_featNum; i++) {
		ptrData->x[i] = 0.0f;
	}
	
	// Parse feature
	ptr = strtok(NULL, ":");
	while (ptr != NULL) {
		int index =  static_cast<int>(strtol(ptr, NULL, 10));
		if (index > m_featNum or index < 1) {
			printf("[WARNING] Invalid feature index!\n");
			delete ptrData->x;
			ptrData->x = NULL;
			return -1;
 		}

		ptr = strtok(NULL, "\t ");		
		ptrData->x[index - 1] = strtof(ptr, NULL);
		ptr = strtok(NULL, ":");
	}
	
	return 0;
}

int FM::initialize()
{	
	// Initialize w0	
	m_w0 = 0.0f;
	m_momentumW0 = 0.0f;

	m_sumW0 = 0.0f;
		
	// Allocate memory for weights and gradients
	if (m_featNum < 0) {
		printf("[ERROR] Invalid feature number!\n");
		return -1;
	}   
	
	m_w = new float[m_featNum];
	m_gradW = new float[m_featNum];
	m_momentumW = new float[m_featNum];
	m_sumW = new float[m_featNum];
	
	// Initialize w
	for (int i = 0; i < m_featNum; ++i) {
		m_w[i] = 0.0f;
		m_gradW[i] = 0.0f;
		m_momentumW[i] = 0.0f;
		m_sumW[i] = 0.0f;
	}
	
	// Allocate memory for factors and gradients
	m_v = new float* [m_degree];
	m_gradV = new float* [m_degree];
	m_momentumV = new float* [m_degree];
	m_sumV = new float* [m_degree];	
	
	srand(time(0));
	for (int i = 0; i < m_degree; ++i) {
		m_v[i] = new float[m_factSize * m_featNum];
		m_gradV[i] = new float[m_factSize * m_featNum];
		m_momentumV[i] = new float[m_factSize * m_featNum];
		m_sumV[i] = new float[m_factSize * m_featNum];

		// Initialize
		for (int j = 0; j < m_factSize * m_featNum; ++j) {
			m_v[i][j] = get_normal_rand(m_initStdDev);
			m_gradV[i][j] = 0.0f;
			m_momentumV[i][j] = 0.0f;
			m_sumV[i][j] = 0.0f;
		}
	}

	m_sumGrad2 = 0.0f;
	m_partialFmFlag = 0;

	// Allocate memory for sparse flags
	m_fmFeatFlag = new int[m_featNum];
	for (int i = 0; i < m_featNum; ++i) {
		m_fmFeatFlag[i] = 0;
	}

	calculate_fm_feat_flags();
/*	for (int i = 0; i < m_featNum; ++i) {
		if (m_fmFeatFlag[i] == 1) {
			printf("%d:1\t", i);
		}
	}
	getchar();
*/	
	return 0;
}

float FM::get_normal_rand(float stdDev)
{
	const int RAND_NUM = 25;
	float x = 0.0f;
	
	for (int i = 0; i < RAND_NUM; ++i) {
		x += (float)rand() / RAND_MAX;
	}

	x -= RAND_NUM / 2.0f;
	x /= sqrt(RAND_NUM / 12.0f);
	x *= stdDev;
	
	return x;
}

int FM::calculate_fm_feat_flags()
{
	const int ZERO_NUM_THRESHOLD = 2;
	const float ZERO_RATIO_THRESHOLD = 0.99f;
	
	for (int i = 0; i < m_featNum; ++i) {
		int zeroNum = 0;
		for (int j = 0; j < m_dataNum; ++j) {
			if (fabs(m_data[j].x[i]) < 1e-6) {
				++zeroNum;
			}
		}
		
		if (zeroNum > ZERO_NUM_THRESHOLD && zeroNum > ZERO_RATIO_THRESHOLD * m_dataNum) {
			m_fmFeatFlag[i] = 1;
		}
	}
	
	return 0;
}

int FM::train()
{
	if (initialize() != 0) {
		printf("[ERROR] Initialize failed!\n");
		return -1;
	}
	
	// Calculate scores for all data
	for (int i = 0; i < m_dataNum; ++i) {
		m_data[i].score = predict(m_data + i);
	}
  
	float loss = calculate_loss();
	float preLoss = 0.0f;

	printf("------------------------------------------------------------------------\n");
	printf("Iteration Process... [%d iterations in total]\n", m_iter_num);
	printf("Total Data Number: %d\t\tFeature Number: %d\n", m_dataNum, m_featNum);
	printf("------------------------------------------------------------------------\n");
   
	// Iteration
	int iterNum = 0;
	int smoothNum = 0;
	
	while (iterNum < m_iter_num) {
		printf("Iter[%d] \t\tLoss[%.0f]\t\tW0[%.2f]\n", ++iterNum, loss, m_w0);
		
		shuffle_data();

		// Mini-batch SGD
		int indexBegin = 0;
		int indexEnd = MIN(indexBegin + m_mini_batch, m_dataNum);

		while(indexEnd <= m_dataNum) {
			run_mini_batch_sgd(indexBegin, indexEnd);
			indexBegin = indexEnd;
			indexEnd = indexBegin + m_mini_batch;
		}

		preLoss = loss;
		loss = calculate_loss();

		// Calculate sum weights for smoothing
		if ( iterNum > 10) {
			for (int i = 0; i < m_featNum; ++i) {
				m_sumW[i] += m_w[i];
			}

			for (int i = 1; i < m_degree; ++i) {
				for (int j = 0; j < m_factSize * m_featNum; ++j) {
					m_sumV[i][j] += m_v[i][j];
				}
			}
		}
		++smoothNum;
	}

	// Smooth weights
	for (int i = 0; i < m_featNum; ++i) {
		m_w[i] = m_sumW[i] / smoothNum;
	}

	for (int i = 1; i < m_degree; ++i) {
		for (int j = 0; j < m_factSize * m_featNum; ++j) {
			m_v[i][j] = m_sumV[i][j] / smoothNum;
		}
	}
	
	return 0;
}

float FM::calculate_loss()
{
	float loss = 0.0f;
	
	// Calculate loss
	for (int i = 0; i < m_dataNum; ++i) {
		float error = m_data[i].score - m_data[i].y;  
		loss += error * error;
	}

	// Regularization terms
	float regLoss = 0.0f;

	if (m_norm == 1) {
		regLoss += fabs(m_w0);
	} else {
		regLoss += m_w0 * m_w0;
	}
	
	for (int i = 0; i < m_featNum; ++i) {
		if (m_norm == 1) {
			regLoss += fabs(m_w[i]);
		} else {
			regLoss += m_w[i] * m_w[i];
		}
	}
	
	for (int i = 1; i < m_degree; ++i) {
		for (int j = 0; j < m_factSize; ++j) {
			for (int k = 0; k < m_featNum; ++k) {
				if (m_partialFmFlag != 0 && m_fmFeatFlag[k] == 0) {
					continue;
				}
		
				int index = j * m_featNum + k;
				if (m_norm == 1) {
					regLoss += fabs(m_v[i][index]);
				} else {
					regLoss += m_v[i][index] * m_v[i][index];
				}
			}
		}
	}
	
	loss += regLoss * m_regFactor;

	return loss;
}

int FM::shuffle_data()
{	
	for(int i = 0; i < m_dataNum; ++i) {
		// Get a random index
		int index = rand() % (i + 1);

		// Swap data[i] with data[index]
		if (index != i) {
			int y = m_data[i].y;
			m_data[i].y = m_data[index].y;
			m_data[index].y = y;

			float* x = m_data[i].x;
			m_data[i].x = m_data[index].x;
			m_data[index].x = x;

			float score = m_data[i].score;
			m_data[i].score = m_data[index].score;
			m_data[index].score = score;
			
			float sumVX = m_data[i].sumVX;
			m_data[i].sumVX = m_data[index].sumVX;
			m_data[index].sumVX = sumVX;
		}
	}
	
	return 0;
}

int FM::run_mini_batch_sgd(int begin, int end)
{
	const float MOMENTUM_FACTOR = 0.0f;

	// Set gradients to 0 at the begining of mini-batch SGD
	m_gradW0 = 0.0f;
	for (int i = 0; i < m_featNum; ++i) {
		m_gradW[i] = 0.0f;
	}
	for (int i = 1; i < m_degree; ++i) {
		for (int k = 0; k < m_featNum; ++k) {
			if (m_partialFmFlag != 0 && m_fmFeatFlag[k] == 0) {
				continue;
			}
			for (int j = 0; j < m_factSize; ++j) {
				m_gradV[i][j * m_featNum + k] = 0.0f;
			}
		}
	}

	// Calculate scores and gradients for mini-batch data
	for (int i = begin; i < end; ++i) {
		m_data[i].score = predict(m_data + i);
		calculate_gradients(m_data + i);
	}

	float step = m_learnRate;

	// Update weights
	if (m_norm == 1) {
		m_w0 = proximal_operator_L1(m_w0 - step * m_gradW0);
	} else {
		m_gradW0 += 2 * m_regFactor * m_w0;
		//	m_sumGrad2 += m_gradW0 * m_gradW0;
		m_momentumW0 = MOMENTUM_FACTOR * m_momentumW0 - step * m_gradW0;
		m_w0 += m_momentumW0;
	}

	for (int i = 0; i < m_featNum; ++i) {
		if (m_norm == 1) {
			m_w[i] = proximal_operator_L1(m_w[i] - step * m_gradW[i]);
		} else {
			m_gradW[i] += 2 * m_regFactor * m_w[i];
			//m_sumGrad2 += m_gradW[i] * m_gradW[i];
			m_momentumW[i] = MOMENTUM_FACTOR * m_momentumW[i] - step * m_gradW[i];
			m_w[i] += m_momentumW[i];
		}
	}

	// Update factors
	int index = 0;
	for (int i = 1; i < m_degree; ++i) {
		for (int k = 0; k < m_featNum; ++k) {
			if (m_partialFmFlag != 0 && m_fmFeatFlag[k] == 0) {
				continue;
			}

			for (int j = 0; j < m_factSize; ++j) {
				index = j * m_featNum + k;
				if (m_norm == 1) {
					m_v[i][index] = proximal_operator_L1(m_v[i][index] - step * m_gradV[i][index]);
				} else {
					m_gradV[i][index] += 2 * m_regFactor * m_v[i][index];
					//				m_sumGrad2 += m_gradV[i][index] * m_gradV[i][index];
					m_momentumV[i][index] = MOMENTUM_FACTOR * m_momentumV[i][index] - step * m_gradV[i][index];
					m_v[i][index] += m_momentumV[i][index];
				}
			}
		}
	}
	
	return 0;
}

int FM::save_model(const char* modelName)
{
	FILE* fp = fopen(modelName, "w");
	if (fp == NULL) {
		printf("[ERROR] Cannot open %s! Saving model failed!\n", modelName);
		return -1;
	}

	// Print model size and w0
	fprintf(fp, "%d\n%d\n%d\n%f\n", m_degree, m_factSize, m_featNum, m_w0);

	// Print weights
	for (int i = 0; i < m_featNum; ++i) {
		fprintf(fp, "%f\n", m_w[i]);
	}

	// Print factors
	for (int i = 1; i < m_degree; ++i) {
		for (int j = 0; j < m_factSize * m_featNum; ++j) {
			fprintf(fp, "%f\n", m_v[i][j]);
		}
	}

	fclose(fp);
	return 0;
}

float FM::proximal_operator_L1(float weight)
{
	float t = m_regFactor * m_learnRate;
	if (weight >= t) {
		return weight - t;
	}
	else if (weight <= -t) {
		return weight + t;
	}
	else {
		return 0;
	}
}

int FM::calculate_gradients(const Data* ptrData)
{
	float score = ptrData->score;
	int y = ptrData->y;
	float error = score - y;

	m_gradW0 += 2 * error;
	
	// Calculate the gradients of weights
	for (int i = 0; i < m_featNum; ++i) {
		m_gradW[i] += ptrData->x[i] * 2 * error;
	}
		
	// Calculate the gradients of factors
	for (int i = 1; i < m_degree; ++i) {
		for (int j = 0; j < m_factSize; ++j) {
			// Pre-calculate items
			float sum = ptrData->sumVX;
			float sumSquare = 0.0f;
			float squareSum = 0.0f;
			
			if (i == 2) {
				for (int k = 0; k < m_featNum; ++k) {
					if (m_partialFmFlag != 0 && m_fmFeatFlag[k] == 0) {
						continue;
					}

					float x = ptrData->x[k];
					if (x < 1e-6 && x > -1e-6) {
						continue;
					}

					float tempScore = m_v[i][j * m_featNum + k] * x;
					squareSum += tempScore * tempScore;
				}
			
				sumSquare = sum * sum;
			}
			
			for (int k = 0; k < m_featNum; ++k) {
				if (m_partialFmFlag != 0 && m_fmFeatFlag[k] == 0) {
					continue;
				}
				
				float x = ptrData->x[k];
				if (x < 1e-6 && x > -1e-6) {
					continue;
				}
				
				int index = j * m_featNum + k;
				
				// Calculate item gradient, degree = i + 1
				float gradItem = 0.0f;
				float item = m_v[i][index] * x;
		
				if (i == 1) {
					gradItem =  x * (sum - item);
				} else if (i == 2) {
					gradItem = x * (0.5 * sumSquare - sum * item - 0.5 * squareSum + item * item);
				}

				m_gradV[i][index] += 2 * error * gradItem;
			}
		}
	}
	
	return 0;
}

int FM::test(const char* fileName, const char* modelName)
{
	if (load_model(modelName) != 0) {
		printf("[ERROR] Load model %s failed!\n", modelName);
		return -1;
	}

	int featNum = m_featNum;

	if (read_data(fileName) != 0) {
		printf("[ERROR] Read test data %s failed!\n", fileName);
		return -1;
	}

	// Check feature size in the data with the model
	if (m_featNum > featNum) {
		printf("[ERROR] Invalid feature index in test_file!\n");
		return -1;
	}

	const int MAX_FILE_NAME_LEN = 1024;
	char resFileName[MAX_FILE_NAME_LEN];
	snprintf(resFileName, MAX_FILE_NAME_LEN, "%s.res", fileName);

	FILE* fp = fopen(resFileName, "w");
	if (fp == NULL) {
		printf("[ERROR] Cannot open %s!\n", resFileName);
		return -1;
	}

	for (int i = 0; i < m_dataNum; ++i) {
		float score = predict(m_data + i);
		int label = m_data[i].y;

		fprintf(fp, "%f\t%d\t%d\n", score, m_maxLabel, label);
	}

	printf("[NOTICE] Predict results are saved in %s\n", resFileName);
	fclose(fp);
	
	return 0;
}

float FM::predict(Data* ptrData)
{
	float score = m_w0;

	for (int i = 0; i < m_featNum; ++i) {
		score += m_w[i] * ptrData->x[i];
	}

	for (int i = 1; i < m_degree; ++i) {
		for (int j = 0; j < m_factSize; ++j) {
			float sum = 0.0f;
			float sumSquare = 0.0f;
			float squareSum = 0.0f;
			float sumCube = 0.0f;
			float cubeSum = 0.0f;
			
			for (int k = 0; k < m_featNum; ++k) {
				if (m_partialFmFlag != 0 && m_fmFeatFlag[k] == 0) {
					continue;
				}
			
				float x = ptrData->x[k];
				if (x < 1e-6 && x > -1e-6) {
					continue;
				}
				
				float tempScore = m_v[i][j * m_featNum + k] * x;
				sum += tempScore;
				if (i == 1) {
					squareSum += tempScore * tempScore;
				} else if (i == 2) {
					cubeSum += tempScore * tempScore * tempScore;
				}
			}

			ptrData->sumVX = sum;
			sumSquare = sum * sum;
			sumCube = sumSquare * sum;
			
			if (i == 1) {
				score += 0.5 * (sumSquare - squareSum);
			} else if (i == 2) {
				score += 1.0f / 6 * (sumCube - 3 * squareSum * sum + 2 * cubeSum);
			}
		}
	}

	// Truncate
	score = MAX(score, m_minLabel);
	score = MIN(score, m_maxLabel);

	ptrData->score = score;
	
	return score;	
}

/*
int FM::calculate_factorial(int n)
{
	const int MAX_FACTORIAL_N = 10;
	
	if (n > MAX_FACTORIAL_N or n < 0) {
		printf("[WARNING] Invalid degree for factorial!\n");
		return 1;
	}
	
	int factorial = 1;
	for (int i = 2; i <= n; ++i) {
		factorial *= i;
	}

	return factorial;
}
*/

int FM::load_model(const char* modelName)
{
	FILE* fp = fopen(modelName, "r");
	if (fp == NULL) {
		printf("[ERROR] Cannot open %s! Loading model failed!\n", modelName);
		return -1;
	}

	const int MAX_MODEL_LINE_LEN = 1024;
	char buf[MAX_MODEL_LINE_LEN];
	int lineNum = 1;

	while (fgets(buf, MAX_MODEL_LINE_LEN, fp) != NULL) {
		switch(lineNum) {
		case 1: {
			m_degree = static_cast<int>(strtol(buf, NULL, 10));
			if (m_degree < 1) {
				printf("[ERROR] Invalid factorization degree!\n");
				fclose(fp);
				return -1;
			}
			
			break;
		}
		case 2: {
			m_factSize = static_cast<int>(strtol(buf, NULL, 10));
			if (m_factSize < 1) {
				printf("[ERROR] Invalid factor size!\n");
				fclose(fp);
				return -1;
			}
		
			break;
		}
		case 3: {
			m_featNum = static_cast<int>(strtol(buf, NULL, 10));
			if (m_featNum < 1) {
				printf("[ERROR] Invalid feature number!\n");
				fclose(fp);
				return -1;
			}		   

			// Allocate memory for weights and factors
			m_w = new float[m_featNum];
			m_v = new float* [m_degree];
			for (int i = 0; i < m_degree; ++i) {
				m_v[i] = new float[m_factSize * m_featNum];
			}
			
			break;
		}
		case 4: {
			m_w0 = strtof(buf, NULL);
			break;
		}		   
		default: {
			// Read weights and factors
			if (lineNum <= m_featNum + 4) {
				m_w[lineNum - 5] = strtof(buf, NULL);
			} else if (lineNum <= (m_degree - 1) * m_factSize * m_featNum + m_featNum + 4) {
				int index = lineNum - m_featNum - 5;
				int i = static_cast<int> (index / (m_factSize * m_featNum)) + 1;
				int j = index % (m_factSize * m_featNum);
				m_v[i][j] = strtof(buf, NULL);
			} else {
				// Do nothing
			} 
		}
		}
		
		++lineNum;
	}

	fclose(fp);
	return 0;
}

} // namespace fm_n_degree

