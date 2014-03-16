/*
 * Profiler.h
 *
 *  Created on: Feb 19, 2014
 *      Author: trevor
 */

#ifndef PROFILER_H_
#define PROFILER_H_

#include <time.h>
#include "GPUNet.h"
#include "Net.h"
#include "NetTrainer.h"
#include "NetData.h"

const int DEFAULT_ITERATIONS = 1000;

class Profiler {
public:
	Profiler(GPUNet *gnet, Net *net, NetTrainer *nt);
	~Profiler();

	void set_iterations(int i);
	int get_iterations();

	void cuda_start();
	void cuda_stop();

	float profile_feed_forward_v1();
	float profile_feed_forward_v1_2(NetData &d);
	float profile_feed_forward_v2();
	float profile_feed_forward_v2_2();

	float profile_backprop_v1();
	float profile_backprop_v2(NetData &d);
	float profile_cpu_backprop(float *targets);
	float profile_cpu_feedforward(float *targets);
	float profile_mse_acc();
	float profile_weight_init();


private:

	GPUNet *gnet;
	Net *net;
	NetTrainer *nt;
	int iterations;
	clock_t start, stop;
	cudaEvent_t cu_start, cu_stop;

};

#endif /* PROFILER_H_ */
