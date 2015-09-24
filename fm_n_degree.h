// Copyright (c) 2014 Baidu Corporation
// @file:   fm_n_degree.h
// @brief:  Header file, n-degree factorization machine
// @author: Li Changcheng (lichangcheng@baidu.com)
// @date:   2014-12-21

namespace fm_n_degree {

// Data struct
struct Data {
	int y;						// Label
	float* x;					// Feature vector
	float score;				// Predicted score
	float sumVX;				// sum of vi * xi
};

class FM {
public:
	FM();
	~FM();

	// Member functions for setting variables
	void set_fm_degree(int degree);
	void set_factor_size(int factSize);
	void set_regular_factor(float regFactor);
	void set_learn_rate(float learnRate);
	void set_partial_fm_flag(int flag);
	void set_init_std_dev(float stdDev);
	void set_regular_term(int regularTerm);

    void set_mini_batch(int mini_batch);
    void set_iterations_num(int iter_num);

	// Member functions for reading data
	int read_data(const char* fileName);
	int parse_line(char* buf, Data* ptrData);
	
	// Member functions for training
	int initialize();
	int train();
	float calculate_loss();
	int shuffle_data();
	int run_mini_batch_sgd(int begin, int end);

	// Member functions for calculating gradients
	float proximal_operator_L1(float weight);
	int calculate_gradients(const Data* ptrData);
	
	// Member fucntions for testing
	int test(const char* fileName, const char* modelName);
	float predict(Data* ptrData);
	int load_model(const char* modelName);
	
	// Other member functions	
	int calculate_fm_feat_flags();
	float get_normal_rand(float stdDev);
	int save_model(const char* modelName);
//	int calculate_factorial(int n);

//private:
public: // For debugging
	static const int S_MAX_STOP_ITER_NUM;			// Max iteration number
	static const int S_MINI_BATCH_SIZE;				// Mini-batch size
	
	// Member variables for data
	int m_maxLabel;				// Max label
	int m_minLabel;				// Min label
	int m_featNum;				// Feature number
	int m_dataNum;				// Data number
	Data* m_data;				// Data
	
	// Member variables for FM
	int m_degree;				// Degree of FM
	int m_factSize;				// Factor size

    int m_mini_batch;           // MINI_BATCH
    int m_iter_num;             // ITERATIONS_NUM

	// Member variables for model
	float m_w0;					// Bias w0
	float* m_w;					// Weights, size = m_featNum
	float** m_v;				// Factors, size = (m_degree - 1) * (m_featNum * m_factSize)
	float m_sumW0;				// Sum of w0 for smoothing
	float* m_sumW;				// Sum of w
	float** m_sumV;				// Sum of v

	// Member variables for parameters
	float m_regFactor;			// Regularization factor
	float m_learnRate;			// Learning rate
	float m_initStdDev;			// Initialization standard deviation
	int m_norm;					// Regularization term: 1 - L1, 2 - L2

	// Member variables for gradients
	float m_gradW0;				// Gradient of w0
	float* m_gradW;				// Gradients of w
	float** m_gradV;			// Gradients of v
	float m_sumGrad2;			// For AdaGrad

	float m_sumGrad2W0;
	float* m_sumGrad2W;
	
	// Member variables for Momentum
	float m_momentumW0;
	float* m_momentumW;
	float** m_momentumV;

	// Member variables for partial FM
	int m_partialFmFlag;		// For partial FM
	int* m_fmFeatFlag;			// Sparse flags for all features
};

} // namespace fm_n_degree

