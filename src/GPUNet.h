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
#include "NetIO.h"
#include "GPUNetSettings.h"
#include <time.h>
#include <string>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

class NetIO;
class GPUNet {

public:

	GPUNet(unsigned int ni, unsigned int no, GPUNetSettings::NetworkStructure net_type);
	GPUNet(std::string net_file);
	~GPUNet();

	void init_structure(unsigned int ni, unsigned int no, GPUNetSettings::NetworkStructure net_type);
	void init_vars();
	void alloc_dev_mem();
	void init_from_net(Net &net, NetData &d);
	void init_net();
	void print_net();
	bool write_net(std::string fname);
	bool read_net(std::string fname);

	int get_num_input();
	int get_num_hidden();
	int get_num_output();

	void set_learning_rate(float lr);
	void set_momentum(float m);
	void set_training_params(float lr, float m);
	void set_max_epochs(int max_epochs);
	void set_desired_accuracy(float acc);
	void set_stopping_conds(int me, float acc);
	void calc_dataset_parameters(TrainingDataSet *tset);
	void train_net_sectioned(TrainingDataSet *tset);
	void train_net_sectioned_overlap(TrainingDataSet *tset);

	void feed_forward_v1_2(float *d_inp);
	void feed_forward_v1_3(float *d_inp);
	void feed_forward_v2_2(unsigned int n, float *d_inp, float *d_sums); //inputs already copied

	void backprop_v2(float *d_inp, float *d_tar);

	float* evaluate(float* input);
	float* batch_evaluate(float** inputs);

	bool validate_output(float* desired_output);
	bool validate_weights(float *desired_ih_weights, float *desired_ho_weights);

	void test_backprop(Net &net, NetData &d);
	void test_feed_forward(Net &net, NetData &d);
	void run_parallel(Net &net, NetData &d);

	size_t current_mem_usage(int dev);

	void copy_to_device_host_array_ptrs_biased(thrust::host_vector<FeatureVector*> &hv, FeatureVector ***dv);
	void copy_to_device_host_array_ptrs_biased_section(thrust::host_vector<FeatureVector*> &hv, FeatureVector ***dv,
			int p_start, int p_end, bool allocate);
	void copy_to_device_host_array_ptrs_biased_section_stream(thrust::host_vector<FeatureVector*> &hv, FeatureVector ***dv,
			int p_start, int p_end, bool allocate, cudaStream_t stream);


//private:

	NetIO *nio;

	/*
	 * State vars
	 */
	int n_gpus;
	size_t *gpu_mem;
	GPUNetSettings::NetworkStructure net_type;
	int n_copyable_patterns;
	int n_sections;

	/*
	 * GPU state
	 */
	cudaStream_t err_calc_stream, weight_update_stream, train_stream1, train_stream2;
	cudaEvent_t event1;

	/*
	 * GPU memory
	 */
	clock_t start, finish;
	unsigned int n_input, n_hidden, n_output;
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

	void run_training_epoch_dev(FeatureVector **feature_vecs, size_t n_features);

	void add_gpu_mem(int bytes);
	int get_current_device();
	size_t dataset_size(TrainingDataSet *tset);
	size_t total_dev_mem(int dev);

};

#endif /* GPUNET_H_ */
