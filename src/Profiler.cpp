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

double Profiler::profile_feed_forward_v1() {
	std::cout << "Profiling feed forward v1 over " << iterations << " iterations." << std::endl;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	for (int i = 0; i < iterations; ++i) {
		gnet->feed_forward_v1();
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	/*
	 * total_l1->l2 = n_hidden*4*((n_layer1+1)*2 + n_layer2)
	 * total_l2->l3 = n_output*4*((n_layer2+1)*2 + n_layer3)
	 *
	 * total = total_l1->l2 + total_l2->l3;
	 */
	int total_l1_2 = gnet->get_num_hidden()*4*((gnet->get_num_input()+1)*2 + gnet->get_num_hidden());
	int total_l2_3 = gnet->get_num_output()*4*((gnet->get_num_hidden()+1)*2 + gnet->get_num_output());
	int total = (total_l1_2 + total_l2_3) * iterations;

	std::cout << milliseconds << " ms" << std::endl;
	std::cout << "Effective Bandwidth (GB/s): " << total/milliseconds/1e6 << std::endl << std::endl;
	return milliseconds;
}


double Profiler::profile_feed_forward_v1_2() {
	std::cout << "Profiling feed forward v1.2 over " << iterations << " iterations." << std::endl;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	for (int i = 0; i < iterations; ++i) {
		//gnet->feed_forward_v1_2();
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	std::cout << milliseconds << " ms" << std::endl;
	return milliseconds;
}


double Profiler::profile_feed_forward_v2() {
	std::cout << "Profiling feed forward v2 over " << iterations << " iterations." << std::endl;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);


	for (int i = 0; i < iterations; ++i) {
		gnet->feed_forward_v2();
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	std::cout << milliseconds << " ms" << std::endl;
	return milliseconds;
}

double Profiler::profile_feed_forward_v2_2() {
	std::cout << "Profiling feed forward v2_2 over " << iterations << " iterations." << std::endl;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);


	for (int i = 0; i < iterations; ++i) {
		gnet->feed_forward_v2_2();
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	std::cout << milliseconds << " ms" << std::endl;
	return milliseconds;
}


double Profiler::profile_backprop_v1() {
	std::cout << "Profiling backprop v1 over " << iterations << " iterations." << std::endl;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	for (int i = 0; i < iterations; ++i) {
		gnet->backprop_v1();
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	std::cout << milliseconds << " ms" << std::endl;
	return milliseconds;
}

double Profiler::profile_backprop_v2() {
	std::cout << "Profiling backprop v2 over " << iterations << " iterations." << std::endl;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	for (int i = 0; i < iterations; ++i) {
		//gnet->backprop_v2();
		CUDA_CHECK_RETURN(cudaDeviceSynchronize());
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	std::cout << milliseconds << " ms" << std::endl;
	return milliseconds;
}

double Profiler::profile_cpu_backprop(float *targets) {
	std::cout << "Profiling cpu backprop over " << iterations << " iterations." << std::endl;

	start = clock();

	for (int i = 0; i < iterations; ++i) {
		nt->backprop(targets);
		nt->update_weights();
	}

	stop = clock();
	long milliseconds = ((double)stop - start) / CLOCKS_PER_SEC * 1000.0;
	std::cout << milliseconds << " ms" << std::endl;
	return milliseconds;
}

double Profiler::profile_cpu_feedforward(float *input) {
	std::cout << "Profiling cpu feedforward over " << iterations << " iterations." << std::endl;

	start = clock();

	for (int i = 0; i < iterations; ++i) {
		net->feed_forward(input);
	}

	stop = clock();
	long milliseconds = ((double)stop - start) / CLOCKS_PER_SEC * 1000.0;
	std::cout << milliseconds << " ms" << std::endl;
	return milliseconds;
}

double Profiler::profile_mse_acc() {
	return 0;
}

double Profiler::profile_weight_init() {
	return 0;
}
