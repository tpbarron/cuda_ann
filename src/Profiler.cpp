/*
 * Profiler.cpp
 *
 *  Created on: Feb 19, 2014
 *      Author: trevor
 */

#include <stdio.h>
#include <iostream>
#include "Profiler.h"

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
	}																		\
}


Profiler::Profiler(GPUNet *gnet, Net *net, NetTrainer *nt) {
	Profiler::gnet = gnet;
	Profiler::net = net;
	Profiler::nt = nt;

	Profiler::iterations = DEFAULT_ITERATIONS;
	Profiler::start = 0;
	Profiler::stop = 0;
	Profiler::cu_start = 0;
	Profiler::cu_stop = 0;
}

Profiler::~Profiler() {
	// TODO Auto-generated destructor stub
}


void Profiler::set_iterations(int i) {
	iterations = i;
}

int Profiler::get_iterations() {
	return iterations;
}

void Profiler::cuda_start() {
	cudaEventCreate(&cu_start);
	cudaEventCreate(&cu_stop);
	cudaEventRecord(cu_start);
}

void Profiler::cuda_stop() {
	cudaEventRecord(cu_stop);
	cudaEventSynchronize(cu_stop);
}

/*
 * Bandwidth calculation
 *
 * BW_effective = (Rb + Wb) / (t * 10^9)
 */
float Profiler::profile_feed_forward_v1_2(NetData &d) {
	std::cout << "Profiling feed forward v1.2 over " << iterations << " iterations." << std::endl;

	TrainingDataSet *tset = d.get_training_dataset();
	float *d_training_set;
	gnet->copy_to_device(tset->training_set, tset->n_training, tset->fpp, &d_training_set);

	cuda_start();

	for (int i = 0; i < iterations; ++i) {
		gnet->feed_forward_v1_2(d_training_set, 0);
	}

	cuda_stop();

	//layer1 - > 2 read
	int bytes_total_layer1 = gnet->get_num_hidden() * ((8*gnet->get_num_input()+1)+4) * iterations;
	int bytes_total_layer2 = gnet->get_num_output() * ((8*gnet->get_num_hidden()+1)+4) * iterations;

	int bytes_total = bytes_total_layer1 + bytes_total_layer2;

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, cu_start, cu_stop);
	CUDA_CHECK_RETURN(cudaFree(d_training_set));

	float bw_effective = bytes_total / (milliseconds * 1e6);

	std::cout << milliseconds << " ms" << std::endl;
	std::cout << "Effective bandwidth: " << bw_effective << std::endl;
	return milliseconds;
}


float Profiler::profile_feed_forward_v1_3(NetData &d) {
	std::cout << "Profiling feed forward v1.3 over " << iterations << " iterations." << std::endl;

	FeatureVector **dv;
	//gnet->copy_to_device_host_array_ptrs_biased(d.get_training_dataset()->training_set, &dv);

	cuda_start();

	for (int i = 0; i < iterations; ++i) {
		gnet->feed_forward_v1_3(dv[0]->input);
	}

	cuda_stop();

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, cu_start, cu_stop);

	std::cout << milliseconds << " ms" << std::endl;
	return milliseconds;
}


inline int
pow2roundup (int x)
{
    if (x < 0)
        return 0;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x+1;
}

float Profiler::profile_feed_forward_v2_2(NetData &d) {
	std::cout << "Profiling feed forward v2.2 over " << iterations << " iterations." << std::endl;

	FeatureVector **dv;
	//gnet->copy_to_device_host_array_ptrs_biased(d.get_training_dataset()->training_set, &dv);

	float *d_sums;
	unsigned int n = pow2roundup((net->n_input+1));
	std::cout << "pow2 = " << n << std::endl;
	CUDA_CHECK_RETURN(cudaMalloc((void**)&d_sums, n*(net->n_hidden)*sizeof(float)));
	CUDA_CHECK_RETURN(cudaMemset(d_sums, 0, n*(net->n_hidden)*sizeof(float)));

	cuda_start();

	for (int i = 0; i < iterations; ++i) {
		gnet->feed_forward_v2_2(n, dv[0]->input, d_sums);
	}

	cuda_stop();

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, cu_start, cu_stop);

	std::cout << milliseconds << " ms" << std::endl;
	return milliseconds;
}


float Profiler::profile_backprop_v2(NetData &d) {
	std::cout << "Profiling backprop v2 over " << iterations << " iterations." << std::endl;

	TrainingDataSet *tset = d.get_training_dataset();
	float *d_training_set;
	gnet->copy_to_device(tset->training_set, tset->n_training, tset->fpp, &d_training_set);


	cuda_start();

	for (int i = 0; i < iterations; ++i) {
		gnet->backprop_v2(d_training_set, 0, gnet->n_input+1);
	}


	cuda_stop();
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, cu_start, cu_stop);
	CUDA_CHECK_RETURN(cudaFree(d_training_set));

	std::cout << milliseconds << " ms" << std::endl;
	return milliseconds;
}

float Profiler::profile_cpu_backprop(float *targets) {
	std::cout << "Profiling cpu backprop over " << iterations << " iterations." << std::endl;

	start = clock();

	for (int i = 0; i < iterations; ++i) {
		nt->backprop(targets);
	}

	stop = clock();
	long milliseconds = ((float)stop - start) / CLOCKS_PER_SEC * 1000.0;
	std::cout << milliseconds << " ms" << std::endl;
	return milliseconds;
}

float Profiler::profile_cpu_feedforward(float *input) {
	std::cout << "Profiling cpu feedforward over " << iterations << " iterations." << std::endl;

	start = clock();

	for (int i = 0; i < iterations; ++i) {
		net->feed_forward(input);
	}

	stop = clock();
	long milliseconds = ((float)stop - start) / CLOCKS_PER_SEC * 1000.0;
	std::cout << milliseconds << " ms" << std::endl;
	return milliseconds;
}

float Profiler::profile_mse_acc() {
	return 0;
}

float Profiler::profile_weight_init() {
	return 0;
}
