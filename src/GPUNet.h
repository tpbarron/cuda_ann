/*
 * GPUNet.h
 *
 *  Created on: Jan 5, 2014
 *      Author: trevor
 */

#ifndef GPUNET_H_
#define GPUNET_H_

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

#include "Net.h"
#include "NetData.h"
#include <time.h>
#include <string>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

//defaults
const float GPU_LEARNING_RATE = 0.7;
const float GPU_MOMENTUM = 0.9;
const long GPU_MAX_EPOCHS = 1500;
const int GPU_DESIRED_ACCURACY = 90;
const int GPU_DESIRED_MSE = 0.001;
const int N_GPUS = 1;



class GPUNet {

public:

	enum NetworkStructure {
		STANDARD,
		GPU_ARCH_OPT
	};

	GPUNet();
	GPUNet(int ni, int no, GPUNet::NetworkStructure net_type);
	~GPUNet();

	void init_structure(int ni, int no, GPUNet::NetworkStructure net_type);
	void init_vars();
	void alloc_dev_mem();
	void init_from_net(Net &net, NetData &d);
	void init_net();
	void print_net();
	void write_net(std::string fname);

	int get_num_input();
	int get_num_hidden();
	int get_num_output();

	void set_learning_rate(float lr);
	void set_momentum(float m);
	void set_training_params(float lr, float m);
	void set_max_epochs(int max_epochs);
	void set_desired_accuracy(float acc);
	void set_stopping_conds(int me, float acc);
	void train_net(TrainingDataSet *tset);

	void feed_forward_v1(); //inputs already copied
	void feed_forward_v1_2(float *d_inp);
	void feed_forward_v2(); //inputs already copied
	void feed_forward_v2_2(); //inputs already copied

	void backprop_v1(); //targets already copied
	void backprop_v2(float *d_inp, float *d_tar);

	bool validate_output(float* desired_output);
	bool validate_weights(float *desired_ih_weights, float *desired_ho_weights);

	void test_backprop(Net &net, NetData &d);
	void test_feed_forward(Net &net, NetData &d);
	void test_reduction();

	void run_parallel(Net &net, NetData &d);

	int num_patterns_copyable(TrainingDataSet *tset);
	size_t current_mem_usage(int dev);


private:

	/*
	 * State vars
	 */
	int n_gpus;
	size_t *gpu_mem;

	/*
	 * GPU memory
	 */
	clock_t start, finish;
	int n_input, n_hidden, n_output;
	float *d_input, *d_hidden, *d_output, *d_target;

	float *d_ih_weights, *d_ho_weights;
	float *d_ih_deltas, *d_ho_deltas;
	float *d_hid_err_gradients, *d_out_err_gradients;

	//for validation, copies on host
	float *h_output;
	float *h_ih_weights, *h_ho_weights;

	/*
	 * learning vars
	 */
	long epoch;
	long max_epochs;
	float l_rate;
	float momentum;
	float desired_acc;
	float trainingSetAccuracy;
	float validationSetAccuracy;
	float generalizationSetAccuracy;
	float trainingSetMSE;
	float validationSetMSE;
	float generalizationSetMSE;

	dim3 get_threadsm2l1();
	dim3 get_threadsm2l2();

	void add_gpu_mem(int bytes);
	int get_current_device();
	size_t dataset_size(TrainingDataSet *tset);
	size_t total_dev_mem(int dev);

	void copy_to_device(thrust::host_vector<FeatureVector*> &hv, FeatureVector*** dv);
	void copy_to_device_biased(thrust::host_vector<FeatureVector*> &hv, FeatureVector ***dv);
	void copy_to_device_host_array(thrust::host_vector<FeatureVector*> &hv, FeatureVector ***dv);
	void copy_to_device_host_array_biased(thrust::host_vector<FeatureVector*> &hv, FeatureVector ***dv);
	void copy_to_device_host_array_ptrs_biased(thrust::host_vector<FeatureVector*> &hv, FeatureVector ***dv);

	void get_set_accuracy_mse(thrust::host_vector<FeatureVector*> set, float* s_acc, float* s_mse);
	void get_set_accuracy_mse_dev(FeatureVector **feature_vecs, size_t n_features, float* s_acc, float* s_mse);
	void run_training_epoch(thrust::host_vector<FeatureVector*> feature_vecs);
	void run_training_epoch_dev(FeatureVector **feature_vecs, size_t n_features);

	float* reduce(int n, int len, float* d_sums, float *d_y);
	float* execute_split_reduction(int n, int offset, float *d_x, float *d_y);

};

#endif /* GPUNET_H_ */
